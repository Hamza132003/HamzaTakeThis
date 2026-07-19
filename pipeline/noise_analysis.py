"""Stage 3: identify WHAT the noise is and WHEN it happens.

Uses PANNs (a CNN trained on Google AudioSet, 527 sound classes) in
sound-event-detection mode to get frame-level predictions, then groups
active frames into timestamped events.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .utils import log, free_cuda

PANNS_SR = 32000  # PANNs models expect 32 kHz

# Speech-family AudioSet classes: we want the NOISE report to describe
# non-speech sounds, so any residual voice here is filtered out.
SPEECH_LABELS = {
    "Speech", "Male speech, man speaking", "Female speech, woman speaking",
    "Child speech, kid speaking", "Conversation", "Narration, monologue",
    "Speech synthesizer", "Babbling", "Whispering", "Shout", "Yell",
    "Children shouting", "Screaming", "Chatter", "Hubbub, speech noise, speech babble",
}


def analyse_noise(noise_wav: Path, cfg: dict, device: str) -> dict:
    if not cfg.get("enabled", True):
        return {"events": [], "top_tags": [], "method": "disabled"}

    try:
        import librosa
        ckpts = _ensure_panns_assets(need_tagging=True)
        from panns_inference import AudioTagging, SoundEventDetection, labels

        y, _ = librosa.load(str(noise_wav), sr=PANNS_SR, mono=True)
        if y.size == 0:
            return {"events": [], "top_tags": [], "method": "empty"}

        min_prob = float(cfg.get("min_prob", 0.15))
        top_k = int(cfg.get("top_k", 8))

        # --- Clip-level tagging (Cnn14): high-confidence "what is it" ---------
        log("Loading PANNs audio tagger (clip-level) ...")
        at = AudioTagging(checkpoint_path=ckpts["tagging"], device=device)
        clipwise = at.inference(y[None, :])[0][0]     # (527,)
        del at
        free_cuda()

        order = [int(i) for i in np.argsort(clipwise)[::-1]
                 if labels[i] not in SPEECH_LABELS]
        top_idx = [i for i in order if clipwise[i] >= min_prob][:top_k]
        if not top_idx:                                # nothing above threshold
            top_idx = order[:min(3, top_k)]
        top_tags = [{"label": labels[i], "confidence": round(float(clipwise[i]), 3)}
                    for i in top_idx]

        # --- Frame-level detection (Cnn14_DecisionLevelMax): "when" -----------
        log("Loading PANNs sound-event detector (timeline) ...")
        sed = SoundEventDetection(checkpoint_path=ckpts["sed"], device=device)
        framewise = sed.inference(y[None, :])[0]       # (frames, 527)
        free_cuda()

        n_frames = framewise.shape[0]
        sec_per_frame = (len(y) / PANNS_SR) / max(1, n_frames)
        gap = int(0.3 / sec_per_frame) + 1

        events = []
        for ci in top_idx:
            thr = max(min_prob, 0.5 * float(clipwise[ci]))   # class-adaptive
            active = framewise[:, ci] >= thr
            for s_f, e_f in _runs(active, gap_tol=gap):
                events.append({
                    "label": labels[ci],
                    "start": round(s_f * sec_per_frame, 2),
                    "end": round((e_f + 1) * sec_per_frame, 2),
                    "confidence": round(float(framewise[s_f:e_f + 1, ci].max()), 3),
                })
        events.sort(key=lambda e: e["start"])
        return {"events": events, "top_tags": top_tags, "method": "panns (tag+sed)"}

    except Exception as e:
        log(f"WARNING: noise analysis failed ({e}).")
        return {"events": [], "top_tags": [], "method": f"error: {e}"}


def _ensure_panns_assets(need_tagging: bool = False) -> dict:
    """panns_inference downloads its files by shelling out to `wget`, which
    Windows doesn't have. Fetch the label map and checkpoint(s) with urllib
    into the ~/panns_data folder panns expects, and return their paths so
    panns skips its own download."""
    import urllib.request
    from pathlib import Path

    data_dir = Path.home() / "panns_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    def fetch(path: Path, url: str, note: str, min_size: float):
        if not path.exists() or path.stat().st_size < min_size:
            log(f"Downloading PANNs {note} (one-time) ...")
            urllib.request.urlretrieve(url, str(path))

    fetch(data_dir / "class_labels_indices.csv",
          "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/"
          "class_labels_indices.csv", "label map", 1000)

    sed = data_dir / "Cnn14_DecisionLevelMax.pth"
    fetch(sed, "https://zenodo.org/record/3987831/files/"
          "Cnn14_DecisionLevelMax_mAP%3D0.385.pth?download=1",
          "detection checkpoint (~327 MB)", 3e8)
    out = {"sed": str(sed)}

    if need_tagging:
        tag = data_dir / "Cnn14_mAP=0.431.pth"
        fetch(tag, "https://zenodo.org/record/3987831/files/"
              "Cnn14_mAP%3D0.431.pth?download=1",
              "tagging checkpoint (~327 MB)", 3e8)
        out["tagging"] = str(tag)
    return out


def _runs(mask: np.ndarray, gap_tol: int = 1):
    """Yield (start_idx, end_idx) of True runs, bridging gaps <= gap_tol."""
    idx = np.where(mask)[0]
    if idx.size == 0:
        return
    start = prev = idx[0]
    for i in idx[1:]:
        if i - prev <= gap_tol:
            prev = i
        else:
            yield start, prev
            start = prev = i
    yield start, prev
