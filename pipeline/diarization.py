"""Stage 4: count speakers and label who spoke when.

Diarization runs in an ISOLATED subprocess environment (pyannote.audio 4 +
numpy 2), because pyannote 4 and ClearVoice cannot share one environment —
see docs/DIARIZATION_RUNTIME_ARCHITECTURE.md. This module never imports
pyannote; it calls the worker through workers.diarization_worker.client.

INPUT BRANCH (`diarization.input_branch`, default `raw_16k`)
-----------------------------------------------------------
Measured on a two-speaker fixture with an identical model, revision and
offline path:

    raw_16k       (audio_16k_mono.wav)  → 2 speakers, 2 overlap regions
    enhanced_16k  (voice_16k.wav)       → 1 speaker,  0 overlap regions

ClearVoice enhancement removes the cues pyannote needs to separate talkers,
so diarization defaults to the raw compatibility track. Enhancement remains
in use for transcription and for the listenable voice deliverable.

There is NO silent fallback between branches: if the configured branch is
unavailable the stage reports a structured `routing_unavailable` failure and
degrades to the honest single-speaker fallback, exactly as any other failure.

Honesty contract: `method` is `pyannote4_isolated` ONLY when the worker
returned genuine pyannote output. Every other path reports
`fallback_single_speaker` with a specific `failure_stage`, and the report and
UI must not describe it as successful diarization.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from .utils import free_cuda, log

DEFAULT_INPUT_BRANCH = "raw_16k"
SUPPORTED_INPUT_BRANCHES = ("raw_16k", "enhanced_16k")

# Human-readable explanation per failure stage, surfaced in report warnings.
_STAGE_HELP = {
    "worker_env_missing": ("the isolated diarization environment is not "
                           "installed - run ./setup_diarization.ps1"),
    "import_failure": "the diarization worker environment failed to import its packages",
    "model_not_cached": ("the diarization model is not in the local cache and "
                         "the worker is running offline"),
    "token_absent": ("no Hugging Face token is available for the gated "
                     "diarization model"),
    "access_denied": ("the Hugging Face account has not been granted access to "
                      "the gated diarization model"),
    "model_load_failure": "the diarization pipeline could not be constructed",
    "cuda_incompatible": "the worker's PyTorch build cannot run on this GPU",
    "gpu_out_of_memory": "the GPU ran out of memory during diarization",
    "timeout": "diarization exceeded its configured timeout",
    "process_crash": "the diarization worker process crashed",
    "malformed_response": "the diarization worker returned an unreadable response",
    "no_speakers_detected": "the diarization model found no speaker turns",
    "cancelled": "diarization was cancelled",
    "audio_unreadable": "the diarization input audio could not be read",
    "routing_unavailable": ("the configured diarization input branch is not "
                            "available; no silent switch to another branch is "
                            "performed"),
    "unknown": "diarization failed for an unclassified reason",
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return ""


def resolve_input_branch(branches: dict, cfg: dict) -> tuple[str, Path | None, str | None]:
    """(branch_name, path_or_None, error_or_None) — never substitutes a branch."""
    name = str(cfg.get("input_branch", DEFAULT_INPUT_BRANCH)).lower().strip()
    if name not in SUPPORTED_INPUT_BRANCHES:
        return name, None, (
            f"unsupported diarization.input_branch '{name}'; supported values "
            f"are {', '.join(SUPPORTED_INPUT_BRANCHES)}")
    path = branches.get(name)
    if path is None:
        return name, None, f"branch '{name}' was not produced by this pipeline run"
    path = Path(path)
    if not path.is_file():
        return name, None, f"branch '{name}' file is missing: {path.name}"
    return name, path, None


def diarize(audio_input, cfg: dict, device: str, models=None,
            audio_sha256: str = "", should_cancel=None,
            full_cfg: dict | None = None) -> dict:
    """`audio_input` is either a dict of {branch_name: path} or a single path
    (single path = legacy callers; treated as the configured branch).

    Returns the legacy dict shape plus explicit status and routing fields.
    """
    branches = (dict(audio_input) if isinstance(audio_input, dict)
                else {str(cfg.get("input_branch", DEFAULT_INPUT_BRANCH)).lower():
                      audio_input})
    # A usable path for the honest fallback even when routing itself failed.
    any_path = next((Path(p) for p in branches.values()
                     if p and Path(p).is_file()), None)

    if not cfg.get("enabled", True):
        out = _fallback(any_path, reason="disabled", failure_stage="disabled",
                        warning="Speaker detection is DISABLED in the "
                                "configuration; everything was labeled as one "
                                "speaker.")
        out["input_branch"] = None
        return out

    branch, audio_path, routing_error = resolve_input_branch(branches, cfg)
    if routing_error is not None:
        # NO silent substitution: an unavailable branch is a reported failure.
        log(f"WARNING: diarization routing failed ({routing_error}); "
            f"falling back to single-speaker segmentation.")
        out = _fallback(
            any_path, reason="routing_unavailable",
            failure_stage="routing_unavailable",
            warning=(f"Speaker detection is DEGRADED: {routing_error}. "
                     f"Everything was labeled as ONE speaker - this is NOT "
                     f"real diarization."))
        out["input_branch"] = branch
        out["input_file"] = None
        out["input_sha256"] = None
        return out

    # Assemble the config view the client expects (diarization.* plus the
    # worker interpreter, which may be overridden per install).
    client_cfg = dict(full_cfg or {})
    client_cfg["diarization"] = dict(cfg)
    input_sha = _sha256_file(audio_path)
    log(f"Diarization input branch: {branch} ({audio_path.name})")

    token = (cfg.get("hf_token") or os.environ.get("HF_TOKEN")
             or os.environ.get("HUGGINGFACE_TOKEN") or _token_file() or None)

    # VRAM policy: never hold enhancement and diarization models at once.
    if models is not None:
        try:
            models.release_gpu()
        except Exception:
            pass
    free_cuda()

    from workers.diarization_worker.client import run_diarization
    resp = run_diarization(audio_path, audio_sha256, client_cfg,
                           token=token, should_cancel=should_cancel)
    routing = {"input_branch": branch, "input_file": audio_path.name,
               "input_sha256": input_sha}

    if resp.genuine_pyannote and resp.state == "ok" and resp.turns:
        segments = [{"start": t["start"], "end": t["end"], "speaker": t["speaker"]}
                    for t in resp.turns]
        log(f"Detected {resp.num_speakers} speaker(s) across {len(segments)} "
            f"turns [pyannote {resp.pyannote_version}, isolated worker, "
            f"{resp.device_used}, {resp.processing_sec}s].")
        out = {
            "segments": segments,
            "num_speakers": resp.num_speakers,
            "speakers": resp.speakers,
            "method": "pyannote4_isolated",
            "state": "ok",
            "genuine_pyannote": True,
            "fallback_used": False,
            "failure_stage": None,
            "speech_regions": resp.speech_regions or _speech_regions(segments),
            "exclusive_segments": resp.exclusive_turns,
            "overlap_regions": resp.overlap_regions,
            "model_id": resp.model_id,
            "model_revision": resp.model_revision,
            "pyannote_version": resp.pyannote_version,
            "worker_torch": resp.torch_version,
            "device_used": resp.device_used,
            "processing_sec": resp.processing_sec,
            "peak_vram_mb": resp.peak_vram_mb,
            "warnings_worker": resp.warnings,
            **routing,
        }
        return out

    stage = (resp.failures[0]["stage"] if resp.failures else "unknown")
    detail = (resp.failures[0].get("message", "") if resp.failures else "")
    help_text = _STAGE_HELP.get(stage, _STAGE_HELP["unknown"])
    log(f"WARNING: real diarization did NOT run ({stage}: {detail[:160]}); "
        f"falling back to single-speaker segmentation.")
    fb = _fallback(
        audio_path, reason=stage, failure_stage=stage,
        warning=(f"Speaker detection is DEGRADED: {help_text}. Everything was "
                 f"labeled as ONE speaker - this is NOT real diarization."))
    fb["state"] = resp.state if resp.state in ("failed", "cancelled", "degraded") else "failed"
    fb["worker_failures"] = resp.failures
    fb["pyannote_version"] = resp.pyannote_version
    fb.update(routing)
    return fb


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


def _fallback(voice_wav: Path | None, reason: str,
              failure_stage: str | None = None,
              warning: str | None = None) -> dict:
    """Split on silence, assign everything to a single speaker.

    This is NOT diarization. `genuine_pyannote` is always False here.
    `voice_wav` may be None when routing failed before any audio was chosen.
    """
    segments: list[dict] = []
    load_error: str | None = None
    if voice_wav is not None and Path(voice_wav).is_file():
        try:
            import librosa
            y, sr = librosa.load(str(voice_wav), sr=16000, mono=True)
            intervals = librosa.effects.split(y, top_db=30)
            segments = [
                {"start": round(s / sr, 2), "end": round(e / sr, 2),
                 "speaker": "SPEAKER_00"}
                for s, e in intervals
            ]
            if not segments:
                segments = [{"start": 0.0, "end": round(len(y) / sr, 2),
                             "speaker": "SPEAKER_00"}]
        except Exception as e:
            # The fallback itself must never take the stage down: report an
            # empty timeline rather than raising out of the diarization stage.
            load_error = f"{type(e).__name__}: {' '.join(str(e).split())[:150]}"
            log(f"  (fallback segmentation could not read the audio: {load_error})")
    out = {
        "segments": segments,
        "num_speakers": 1,
        "speakers": ["SPEAKER_00"],
        "method": "fallback_single_speaker",
        "state": "failed",
        "genuine_pyannote": False,
        "fallback_used": True,
        "failure_stage": failure_stage or reason,
        "speech_regions": _speech_regions(segments),
        "exclusive_segments": [],
        "overlap_regions": [],
    }
    if load_error:
        out["fallback_load_error"] = load_error
    if warning:
        out["warning"] = warning
    return out
