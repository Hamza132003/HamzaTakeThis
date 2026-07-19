"""Multi-speaker handling.

Builds one full-length CLEANED audio track per speaker:
  - where only that speaker talks -> the isolated voice
  - where two speakers overlap    -> a separation model (ClearVoice
    MossFormer2_SS_16K, SpeechBrain SepFormer as fallback) un-mixes them,
    and each separated stream is assigned to the right speaker by ECAPA-TDNN
    speaker-embedding similarity.

Transcription then reads each turn from its own speaker's track, so
simultaneous speech no longer garbles both transcripts.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from .utils import log, free_cuda, guard_speechbrain_lazy

SR = 16000


def _load16k(path: Path) -> np.ndarray:
    import librosa
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    return y.astype(np.float32)


def _save(path: Path, y: np.ndarray) -> None:
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(y, dtype=np.float32), SR)


def detect_overlaps(segments: list) -> list:
    """Return time regions where two different speakers are active at once."""
    overlaps = []
    for i in range(len(segments)):
        a = segments[i]
        for j in range(i + 1, len(segments)):
            b = segments[j]
            if a["speaker"] == b["speaker"]:
                continue
            s, e = max(a["start"], b["start"]), min(a["end"], b["end"])
            if e - s > 0.15:
                overlaps.append({"start": round(s, 2), "end": round(e, 2),
                                 "speakers": sorted({a["speaker"], b["speaker"]})})
    overlaps.sort(key=lambda o: o["start"])
    return overlaps


def build_speaker_tracks(voice_wav: Path, segments: list, out_dir: Path,
                         cfg: dict, device: str, models=None) -> dict:
    if models is None:
        from .models import MANAGER as models
    guard_speechbrain_lazy()   # incl. Windows no-symlink patch for fetching

    y = _load16k(voice_wav)
    n = len(y)
    speakers = sorted({s["speaker"] for s in segments})
    buffers = {spk: np.zeros(n, dtype=np.float32) for spk in speakers}

    # Base: copy the isolated voice into each speaker's own turns.
    for s in segments:
        a, b = int(s["start"] * SR), min(n, int(s["end"] * SR))
        buffers[s["speaker"]][a:b] = y[a:b]

    overlaps = detect_overlaps(segments)
    method = "mask-only"

    if overlaps and cfg.get("separate_overlap", True):
        try:
            embed = _ecapa_embedder(cfg, device, models)
            # One reference embedding per speaker from their NON-overlapped
            # audio (>= 2 s needed for a stable voiceprint).
            profiles = {}
            for spk in speakers:
                ref = _non_overlap_audio(buffers[spk], segments, overlaps, spk)
                if len(ref) >= 2 * SR:
                    profiles[spk] = _embedding(embed, ref)

            unmix, unmix_name = _load_separator(cfg, device, models, out_dir)
            skipped_3plus = 0
            for ov in overlaps:
                if len(ov["speakers"]) != 2:
                    skipped_3plus += 1
                    continue  # separator handles exactly 2 talkers
                spkA, spkB = ov["speakers"]
                if spkA not in profiles or spkB not in profiles:
                    continue  # not enough clean audio to match streams safely
                a, b = int(ov["start"] * SR), min(n, int(ov["end"] * SR))
                if b - a < int(0.3 * SR):
                    continue
                streams = unmix(y[a:b])
                if streams is None:
                    continue
                e0, e1 = _embedding(embed, streams[0]), _embedding(embed, streams[1])
                straight = _cos(e0, profiles[spkA]) + _cos(e1, profiles[spkB])
                swapped = _cos(e0, profiles[spkB]) + _cos(e1, profiles[spkA])
                if straight >= swapped:
                    buffers[spkA][a:b], buffers[spkB][a:b] = streams[0], streams[1]
                else:
                    buffers[spkA][a:b], buffers[spkB][a:b] = streams[1], streams[0]
            method = unmix_name
            if skipped_3plus:
                method += f" (skipped {skipped_3plus} region(s) with 3+ speakers)"
            free_cuda()
        except Exception as e:
            log(f"WARNING: overlap un-mix failed ({e}); mask-only.")
            method = "mask-only (unmix failed)"

    tracks = {}
    for spk, buf in buffers.items():
        p = out_dir / "speakers" / f"{spk}.wav"
        _save(p, buf)
        tracks[spk] = str(p)

    log(f"Built {len(tracks)} speaker track(s); {len(overlaps)} overlap region(s) [{method}].")
    return {"tracks": tracks, "overlaps": overlaps, "method": method}


# ------------------------------------------------------------------ helpers
def _non_overlap_audio(buf: np.ndarray, segments: list, overlaps: list,
                       spk: str) -> np.ndarray:
    """Concatenate this speaker's turns, excluding overlapped regions."""
    n = len(buf)
    keep = np.zeros(n, dtype=bool)
    for s in segments:
        if s["speaker"] == spk:
            keep[int(s["start"] * SR):min(n, int(s["end"] * SR))] = True
    for ov in overlaps:
        if spk in ov["speakers"]:
            keep[int(ov["start"] * SR):min(n, int(ov["end"] * SR))] = False
    return buf[keep]


def _ecapa_embedder(cfg: dict, device: str, models):
    model_id = cfg.get("embedding_model", "speechbrain/spkrec-ecapa-voxceleb")

    def loader():
        from speechbrain.inference.speaker import EncoderClassifier
        savedir = Path.home() / ".cache" / "voice_isolator" / model_id.split("/")[-1]
        log(f"Loading speaker embedder '{model_id}' ...")
        return EncoderClassifier.from_hparams(
            source=model_id, savedir=str(savedir), run_opts={"device": device})

    return models.get(f"ecapa:{model_id}", loader, device)


def _embedding(embedder, y: np.ndarray) -> np.ndarray:
    import torch
    sig = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32)).unsqueeze(0)
    with torch.no_grad():
        emb = embedder.encode_batch(sig)
    return emb.squeeze().detach().cpu().numpy().astype(np.float32)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _load_separator(cfg: dict, device: str, models, out_dir: Path):
    """Return (unmix_fn, name). unmix_fn(y_16k) -> [stream0, stream1] | None."""
    try:
        model_name = cfg.get("separator", "MossFormer2_SS_16K")

        def loader():
            from clearvoice import ClearVoice
            log(f"Loading ClearVoice separator '{model_name}' ...")
            return ClearVoice(task="speech_separation", model_names=[model_name])

        cv = models.get(f"clearvoice_ss:{model_name}", loader, "cuda")
        tmp_dir = out_dir / "_unmix"

        def unmix(y: np.ndarray):
            tmp_dir.mkdir(parents=True, exist_ok=True)
            tmp = tmp_dir / "seg.wav"
            _save(tmp, y)
            try:
                out = cv(input_path=str(tmp), online_write=False)
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            out = np.asarray(out)
            out = np.squeeze(out)
            if out.ndim != 2:
                return None
            # Normalize orientation to (n_src, time).
            if out.shape[0] > out.shape[1]:
                out = out.T
            if out.shape[0] < 2:
                return None
            return [_fit(out[k], len(y)) for k in range(2)]

        return unmix, "mossformer2_ss"
    except Exception as e:
        log(f"  (ClearVoice separator unavailable: {e}; using SepFormer)")
        mid = cfg.get("sepformer_model", "speechbrain/sepformer-whamr16k")

        def loader_sb():
            from speechbrain.inference.separation import SepformerSeparation
            savedir = Path.home() / ".cache" / "voice_isolator" / mid.split("/")[-1]
            log(f"Loading SepFormer un-mixer '{mid}' ...")
            return SepformerSeparation.from_hparams(
                source=mid, savedir=str(savedir), run_opts={"device": device})

        model = models.get(f"sepformer:{mid}", loader_sb, device)

        def unmix_sb(y: np.ndarray):
            import torch
            sig = torch.from_numpy(np.ascontiguousarray(y)).unsqueeze(0)
            if device == "cuda":
                sig = sig.to("cuda")
            with torch.no_grad():
                est = model.separate_batch(sig)          # (1, time, n_src)
            est = est.squeeze(0).detach().cpu().numpy()
            if est.shape[1] < 2:
                return None
            return [_fit(est[:, k], len(y)) for k in range(2)]

        return unmix_sb, "sepformer"


def _fit(s: np.ndarray, n: int) -> np.ndarray:
    s = s.astype(np.float32)
    s = np.pad(s, (0, max(0, n - len(s))))[:n]
    peak = np.max(np.abs(s)) + 1e-9
    return s / peak * 0.97
