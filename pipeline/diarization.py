"""Stage 4: count speakers and label who spoke when (pyannote.audio).

Needs a one-time free HuggingFace token to DOWNLOAD the model (accept terms
at BOTH hf.co/pyannote/speaker-diarization-3.1 AND
hf.co/pyannote/segmentation-3.0). After download it runs offline.
If no token is available, falls back to energy-based segmentation (1 speaker)
and returns a LOUD warning so the report/UI can surface the degradation.
"""
from __future__ import annotations

import os
from pathlib import Path

from .utils import log, free_cuda, guard_speechbrain_lazy


def diarize(voice_wav: Path, cfg: dict, device: str, models=None) -> dict:
    if models is None:
        from .models import MANAGER as models

    # pyannote -> lightning -> torch._dynamo scans sys.modules; SpeechBrain's
    # lazy k2/wordemb stubs raise ImportError there unless patched first.
    guard_speechbrain_lazy()

    if not cfg.get("enabled", True):
        return _fallback(voice_wav, reason="disabled")

    token = (cfg.get("hf_token") or os.environ.get("HF_TOKEN")
             or os.environ.get("HUGGINGFACE_TOKEN") or _token_file())
    if not token:
        log("No HF token found (config.diarization.hf_token or HF_TOKEN env). "
            "Falling back to single-speaker segmentation.")
        return _fallback(
            voice_wav, reason="no_token",
            warning=("Speaker detection is DEGRADED: no Hugging Face token is "
                     "set, so everything was labeled as one speaker. Accept "
                     "the model terms at hf.co/pyannote/speaker-diarization-3.1 "
                     "and hf.co/pyannote/segmentation-3.0, create a read token "
                     "at hf.co/settings/tokens, and paste it in the settings."))

    try:
        import torch

        model_id = cfg.get("model", "pyannote/speaker-diarization-3.1")

        def loader():
            import inspect
            from pyannote.audio import Pipeline
            log("Loading pyannote diarization pipeline ...")
            # pyannote 4.x renamed use_auth_token= to token=
            params = inspect.signature(Pipeline.from_pretrained).parameters
            kw = "token" if "token" in params else "use_auth_token"
            return Pipeline.from_pretrained(model_id, **{kw: token})

        pipe = models.get(f"pyannote:{model_id}", loader, device)
        if device == "cuda":
            pipe.to(torch.device("cuda"))

        kwargs = {}
        if cfg.get("min_speakers") is not None:
            kwargs["min_speakers"] = int(cfg["min_speakers"])
        if cfg.get("max_speakers") is not None:
            kwargs["max_speakers"] = int(cfg["max_speakers"])

        # Pass a preloaded waveform: pyannote 4.x decodes files via torchcodec,
        # whose DLLs often fail to load on Windows. In-memory audio skips it.
        import soundfile as sf
        wav, sr = sf.read(str(voice_wav), dtype="float32", always_2d=True)
        audio = {"waveform": torch.from_numpy(wav.T).contiguous(),
                 "sample_rate": int(sr)}

        annotation = pipe(audio, **kwargs)
        # pyannote 4.x pipelines may wrap the Annotation in an output object.
        if not hasattr(annotation, "itertracks"):
            annotation = getattr(annotation, "speaker_diarization", annotation)
        free_cuda()

        segments = [
            {"start": round(turn.start, 2), "end": round(turn.end, 2), "speaker": spk}
            for turn, _, spk in annotation.itertracks(yield_label=True)
        ]
        segments.sort(key=lambda s: s["start"])
        speakers = sorted({s["speaker"] for s in segments})
        log(f"Detected {len(speakers)} speaker(s) across {len(segments)} turns.")
        return {"segments": segments, "num_speakers": len(speakers),
                "speakers": speakers, "method": "pyannote",
                "speech_regions": _speech_regions(segments)}

    except Exception as e:
        log(f"WARNING: diarization failed ({e}); using fallback.")
        return _fallback(
            voice_wav, reason=f"error: {e}",
            warning=f"Speaker detection is DEGRADED (pyannote failed: {e}). "
                    f"Everything was labeled as one speaker.")


def _token_file() -> str:
    """Read the HF token from an untracked hf_token.txt at the project root
    (kept out of git so the public repo never contains a credential)."""
    try:
        p = Path(__file__).resolve().parent.parent / "hf_token.txt"
        return p.read_text(encoding="utf-8").strip() if p.exists() else ""
    except Exception:
        return ""


def _speech_regions(segments: list, pad: float = 0.3) -> list:
    """Merged union of all speech turns → [{start, end}], gap-merged."""
    if not segments:
        return []
    ivals = sorted((max(0.0, s["start"] - pad), s["end"] + pad) for s in segments)
    merged = [list(ivals[0])]
    for a, b in ivals[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [{"start": round(a, 2), "end": round(b, 2)} for a, b in merged]


def _fallback(voice_wav: Path, reason: str, warning: str | None = None) -> dict:
    """Split on silence, assign everything to a single speaker."""
    import librosa
    y, sr = librosa.load(str(voice_wav), sr=16000, mono=True)
    intervals = librosa.effects.split(y, top_db=30)
    segments = [
        {"start": round(s / sr, 2), "end": round(e / sr, 2), "speaker": "SPEAKER_00"}
        for s, e in intervals
    ]
    if not segments:
        segments = [{"start": 0.0, "end": round(len(y) / sr, 2), "speaker": "SPEAKER_00"}]
    out = {"segments": segments, "num_speakers": 1,
           "speakers": ["SPEAKER_00"], "method": f"fallback ({reason})",
           "speech_regions": _speech_regions(segments)}
    if warning:
        out["warning"] = warning
    return out
