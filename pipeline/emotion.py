"""Stage 7: sentiment from vocal tone (language-agnostic).

Runs a speech-emotion model on each speaker turn. We report the emotion
label plus a coarse sentiment polarity, since translating sentiment across
Farsi/Hebrew->Arabic on distant audio would be unreliable; tone is not.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .utils import log, free_cuda, guard_speechbrain_lazy

SR = 16000

# Map common SER labels to a coarse polarity.
POLARITY = {
    "hap": "positive", "happy": "positive", "joy": "positive",
    "ang": "negative", "angry": "negative", "sad": "negative",
    "disgust": "negative", "fear": "negative", "fearful": "negative",
    "neu": "neutral", "neutral": "neutral", "calm": "neutral",
}


def analyse_emotion(voice_wav: Path, segments: list, cfg: dict, device: str) -> list:
    if not cfg.get("enabled", True) or not segments:
        return segments

    try:
        guard_speechbrain_lazy()
        import librosa
        from transformers import pipeline

        y, _ = librosa.load(str(voice_wav), sr=SR, mono=True)
        model_id = cfg.get("model", "superb/wav2vec2-base-superb-er")
        log(f"Loading speech-emotion model '{model_id}' ...")
        clf = pipeline("audio-classification", model=model_id,
                       device=0 if device == "cuda" else -1)

        for seg in segments:
            a = int(seg["start"] * SR)
            b = min(len(y), int(seg["end"] * SR))
            clip = y[a:b].astype(np.float32)
            if clip.size < int(0.2 * SR):
                seg["emotion"], seg["sentiment"] = "unknown", "unknown"
                continue
            preds = clf(clip, top_k=1)
            label = preds[0]["label"].lower()
            seg["emotion"] = label
            seg["emotion_score"] = round(float(preds[0]["score"]), 3)
            seg["sentiment"] = POLARITY.get(label, "neutral")

        free_cuda()
        log("Emotion/sentiment analysis complete.")
        return segments

    except Exception as e:
        log(f"WARNING: emotion analysis failed ({e}).")
        for seg in segments:
            seg.setdefault("emotion", "unknown")
            seg.setdefault("sentiment", "unknown")
        return segments
