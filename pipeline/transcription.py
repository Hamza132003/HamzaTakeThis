"""Stage 5: transcribe with faster-whisper (word timestamps + anti-hallucination).

For noisy, distant recordings the best transcription source is not obvious:
the original preserves natural cues Whisper likes, but when noise dominates
the enhanced voice is far better. So we try BOTH and keep whichever finds
more real speech.

Robustness:
- Decoding is GATED to the speech regions diarization found
  (clip_timestamps), so Whisper never decodes pure-noise spans — the main
  source of hallucinated text.
- Per-segment rejection on no_speech_prob / avg_logprob / compression ratio /
  mean word confidence / n-gram repetition (all configurable).
- Segments whose English translation length is wildly inconsistent with the
  source text are FLAGGED (quality.flagged) rather than silently kept.
- Overlapping-speech regions are re-decoded from each speaker's separated
  track so simultaneous talkers get their own words.
- Word-level timestamps + confidence are attached to every segment.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

from .utils import log, free_cuda, check_cancel, JobCancelled

SR = 16000
WHISPER_KEY = "whisper"   # ModelManager cache key (runner evicts before NLLB)

# Segments that are just a non-speech tag Whisper emits on music/noise.
_JUNK = {"music", "musik", "музыка", "موسیقی", "موسیقى", "الموسيقى",
         "applause", "noise", "silence", "thank you", "thanks for watching",
         "you", "bye", "."}

# Biases decoding toward clean, punctuated, formal text in the target
# language (notably improves Arabic output style and reduces gibberish).
_INITIAL_PROMPT = {
    "ar": "فيما يلي نصٌّ باللغة العربية الفصحى، مكتوبٌ بعلامات الترقيم الصحيحة.",
    "fa": "در ادامه متنی به زبان فارسی روان و با علائم نگارشی صحیح آمده است.",
    "he": "להלן תמליל בעברית תקנית עם סימני פיסוק נכונים.",
    "en": "The following is a clear, well-punctuated English transcript.",
}


def transcribe(audio_wavs, diar_segments: list, cfg: dict, device: str,
               speaker_tracks: dict | None = None,
               speech_regions: list | None = None,
               overlaps: list | None = None,
               models=None) -> list:
    if not cfg.get("enabled", True):
        return []
    if models is None:
        from .models import MANAGER as models
    if isinstance(audio_wavs, (str, Path)):
        audio_wavs = [audio_wavs]

    try:
        import librosa

        model = _load_model(cfg, device, models)

        forced = str(cfg.get("language", "auto")).lower()
        clip_ts = _clip_timestamps(model, cfg, speech_regions)

        # Decode each candidate source; keep the one with the most real speech.
        best = None
        for wav in audio_wavs:
            y, _ = librosa.load(str(wav), sr=SR, mono=True)
            if y.size < int(0.2 * SR):
                continue
            if forced not in ("", "auto"):
                candidates = [forced]
            else:
                candidates = _detect_language(model, y, cfg, clip_ts)
            segs, lang, score = _decode_best(model, y, candidates, cfg, clip_ts)
            if segs and (best is None or score > best[3]):
                best = (y, segs, lang, score)

        if best is None:
            log("No intelligible speech found in any source.")
            return []
        y, wsegs, lang, _ = best

        eng = _decode_task(model, y, lang, "translate", cfg, clip_ts)

        results = []
        for w in wsegs:
            results.append(_make_segment(w, lang, diar_segments, eng))

        # Re-decode overlapped regions from each speaker's separated track.
        if speaker_tracks and overlaps:
            results = _redecode_overlaps(model, results, overlaps,
                                         speaker_tracks, lang, cfg)

        _flag_inconsistent(results)
        free_cuda()
        n_flag = sum(1 for r in results if r["quality"]["flagged"])
        log(f"Transcribed {len(results)} segment(s) [{lang}] + English"
            + (f" ({n_flag} flagged low-confidence)" if n_flag else "") + ".")
        return results

    except JobCancelled:
        raise
    except Exception as e:
        log(f"WARNING: transcription failed ({e}).")
        return []


# ------------------------------------------------------------------ loading
def _load_model(cfg: dict, device: str, models):
    compute_type = cfg.get("compute_type",
                           "int8_float16" if device == "cuda" else "int8")
    model_name = cfg.get("model", "large-v3")

    def loader():
        from faster_whisper import WhisperModel
        log(f"Loading faster-whisper '{model_name}' ({device}, {compute_type}) ...")
        return WhisperModel(model_name, device=device, compute_type=compute_type)

    return models.get(WHISPER_KEY, loader, device)


def _clip_timestamps(model, cfg: dict, speech_regions: list | None):
    """Flattened [s, e, s, e, ...] from diarization speech regions, if both
    the config and the installed faster-whisper support it."""
    if not cfg.get("gate_with_diarization", True) or not speech_regions:
        return None
    try:
        if "clip_timestamps" not in inspect.signature(model.transcribe).parameters:
            return None
    except Exception:
        return None
    flat = []
    for r in speech_regions:
        s, e = max(0.0, float(r["start"]) - 0.25), float(r["end"]) + 0.25
        if flat and s <= flat[-1]:
            flat[-1] = max(flat[-1], e)
        else:
            flat.extend([s, e])
    return flat if flat else None


def _detect_language(model, y, cfg: dict, clip_ts) -> list:
    """Auto language: trust Whisper's detector when it lands on a configured
    candidate; otherwise fall back to scoring every candidate."""
    candidates = list(cfg.get("languages") or []) or [None]
    try:
        _, info = _whisper(model, y, None, "transcribe", True, cfg, clip_ts)
        detected = getattr(info, "language", None)
        prob = float(getattr(info, "language_probability", 0.0) or 0.0)
        if detected in candidates and prob >= 0.5:
            log(f"Language auto-detected: {detected} (p={prob:.2f})")
            return [detected]
    except JobCancelled:
        raise
    except Exception as e:
        log(f"  (language detection failed: {e})")
    return candidates


# ------------------------------------------------------------------ decoding
def _whisper(model, y, language, task, vad, cfg, clip_ts=None):
    kwargs = dict(
        language=language, task=task,
        beam_size=int(cfg.get("beam_size", 8)),
        vad_filter=vad, vad_parameters=dict(min_silence_duration_ms=500),
        condition_on_previous_text=False,      # avoids repetition loops
        no_speech_threshold=float(cfg.get("max_no_speech_prob", 0.6)),
        compression_ratio_threshold=float(cfg.get("max_compression_ratio", 2.4)),
        no_repeat_ngram_size=3,
        temperature=[0.0, 0.2, 0.4, 0.6],
        word_timestamps=bool(cfg.get("word_timestamps", True)),
    )
    # Style prompt only for same-language transcription (not translate).
    if (task == "transcribe" and language
            and cfg.get("use_initial_prompt", True)
            and language in _INITIAL_PROMPT):
        kwargs["initial_prompt"] = _INITIAL_PROMPT[language]
    if clip_ts:
        kwargs["clip_timestamps"] = clip_ts
    segs, info = model.transcribe(y, **kwargs)
    # faster-whisper decodes lazily as the generator is consumed, so a
    # per-segment cancel check here interrupts the actual decode work.
    out = []
    for s in segs:
        check_cancel()
        out.append(s)
    return out, info


def _is_junk(text: str) -> bool:
    s = text.strip().strip(".!?،؛…-–—[](){}\"'♪ ").lower()
    return len(s) <= 1 or s in _JUNK


def _repeated_ngram(text: str, n: int = 3, times: int = 3) -> bool:
    toks = text.split()
    if len(toks) < n * times:
        return False
    grams = [" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    from collections import Counter
    return Counter(grams).most_common(1)[0][1] >= times


def _clean(segs, cfg):
    min_alp = float(cfg.get("min_avg_logprob", -1.0))
    max_nsp = float(cfg.get("max_no_speech_prob", 0.6))
    max_cr = float(cfg.get("max_compression_ratio", 2.4))
    min_wp = float(cfg.get("min_word_prob", 0.35))
    out = []
    for s in segs:
        t = (s.text or "").strip()
        if not t or (s.end - s.start) < 0.3 or _is_junk(t):
            continue
        cr = float(getattr(s, "compression_ratio", 1.0) or 1.0)
        nsp = float(s.no_speech_prob or 0.0)
        alp = float(s.avg_logprob or 0.0)
        words = [{"word": w.word.strip(), "start": round(w.start, 2),
                  "end": round(w.end, 2), "probability": round(w.probability, 3)}
                 for w in (s.words or [])] if getattr(s, "words", None) else []
        mean_wp = (float(np.mean([w["probability"] for w in words]))
                   if words else None)
        # Hard rejects: statistical signatures of hallucination.
        if cr > max_cr or _repeated_ngram(t):
            continue
        if nsp > max_nsp and alp < min_alp:
            continue
        if mean_wp is not None and mean_wp < min_wp:
            continue
        out.append({"start": s.start, "end": s.end, "text": t,
                    "alp": alp, "nsp": nsp, "cr": cr, "words": words})
    return out


def _decode_task(model, y, language, task, cfg, clip_ts=None):
    """One Whisper task with a no-VAD fallback so faint speech isn't lost."""
    try:
        segs, _ = _whisper(model, y, language, task, True, cfg, clip_ts)
        clean = _clean(segs, cfg)
        if not clean:
            segs, _ = _whisper(model, y, language, task, False, cfg, clip_ts)
            clean = _clean(segs, cfg)
        return clean
    except JobCancelled:
        raise
    except Exception as e:
        log(f"  ({task} pass failed: {e})")
        return []


def _decode_best(model, y, candidates, cfg, clip_ts=None):
    """Decode under each candidate language; return (segments, lang, score)."""
    best = None
    for lang in candidates:
        clean = _decode_task(model, y, lang, "transcribe", cfg, clip_ts)
        if not clean:
            continue
        score = sum(c["alp"] * len(c["text"]) for c in clean)
        detected = lang
        if detected is None:
            _, info = _whisper(model, y, None, "transcribe", True, cfg, clip_ts)
            detected = getattr(info, "language", "unknown")
        if best is None or score > best[2]:
            best = (clean, detected, score)
    if best is None:
        return [], (candidates[0] or "unknown"), -1e9
    return best[0], best[1], best[2]


# ------------------------------------------------------------------ assembly
def _make_segment(w: dict, lang: str, diar_segments: list, eng: list,
                  speaker: str | None = None) -> dict:
    return {
        "speaker": speaker or _assign_speaker(w["start"], w["end"], diar_segments),
        "start": round(w["start"], 2),
        "end": round(w["end"], 2),
        "language": lang,
        "text": w["text"].strip(),
        "english": _overlap_text(w["start"], w["end"], eng),
        "words": w.get("words", []),
        "quality": {"avg_logprob": round(w["alp"], 3),
                    "no_speech_prob": round(w["nsp"], 3),
                    "compression_ratio": round(w.get("cr", 1.0), 2),
                    "flagged": False},
    }


def _redecode_overlaps(model, results: list, overlaps: list,
                       speaker_tracks: dict, lang: str, cfg: dict) -> list:
    """Replace mixed-audio segments inside overlap regions with per-speaker
    decodes from the separated tracks."""
    import librosa

    track_audio = {}
    for spk, path in speaker_tracks.items():
        try:
            track_audio[spk], _ = librosa.load(str(path), sr=SR, mono=True)
        except Exception:
            pass

    replaced_any = False
    for ov in overlaps:
        if len(ov["speakers"]) != 2 or (ov["end"] - ov["start"]) < 0.5:
            continue
        if any(spk not in track_audio for spk in ov["speakers"]):
            continue
        new_segs = []
        for spk in ov["speakers"]:
            a = max(0.0, ov["start"] - 0.2)
            b = ov["end"] + 0.2
            y = track_audio[spk][int(a * SR):int(b * SR)]
            if y.size < int(0.3 * SR):
                continue
            segs = _decode_task(model, y, lang, "transcribe", cfg)
            eng = _decode_task(model, y, lang, "translate", cfg) if segs else []
            for s in segs:
                s = dict(s)
                s["start"] += a
                s["end"] += a
                for wd in s.get("words", []):
                    wd["start"] = round(wd["start"] + a, 2)
                    wd["end"] = round(wd["end"] + a, 2)
                eng_off = [{"start": e["start"] + a, "end": e["end"] + a,
                            "text": e["text"]} for e in eng]
                new_segs.append(_make_segment(s, lang, [], eng_off, speaker=spk))
        if not new_segs:
            continue
        # Drop mixed-decode segments that live mostly inside this overlap.
        kept = []
        for r in results:
            mid = (r["start"] + r["end"]) / 2
            inside = min(r["end"], ov["end"]) - max(r["start"], ov["start"])
            frac = inside / max(1e-6, r["end"] - r["start"])
            if frac > 0.5 and ov["start"] <= mid <= ov["end"]:
                continue
            kept.append(r)
        results = kept + new_segs
        replaced_any = True

    if replaced_any:
        results.sort(key=lambda r: r["start"])
        log("Re-decoded overlapping speech from separated speaker tracks.")
    return results


# Signature phrases Whisper emits on music/noise instead of real speech
# (YouTube-training artifacts). Substring match on the English, lowercase.
_HALLUCINATION_EN = ("thank you for watching", "thanks for watching",
                     "please subscribe", "like and subscribe",
                     "see you in the next video", "music playing")

# Subtitle-credit lines Whisper invents on media audio (learned from
# fansubbed content). Substring match on the source text.
_HALLUCINATION_SRC = ("توقيت وترجمة", "ترجمة نانسي", "اشترك في القناة",
                      "زیرنویس", "כתוביות", "subtitles by", "subtitled by",
                      "translated by")


def _flag_inconsistent(results: list) -> None:
    """Flag hallucination signatures (kept visible, greyed out in the UI,
    rather than silently dropped): signature filler phrases in the English,
    junk-tag source text, or a wild english/source length mismatch."""
    for r in results:
        text = (r.get("text") or "").strip()
        eng = (r.get("english") or "").strip().lower()
        t, e = len(text), len(eng)
        reason = None
        if any(p in text.lower() or p in text for p in _HALLUCINATION_SRC):
            reason = "subtitle-credit hallucination"
        elif any(p in eng for p in _HALLUCINATION_EN):
            reason = "signature noise phrase in English"
        elif any(j in text.lower() for j in ("موسیقی", "موسيقى", "music")) and t < 30:
            reason = "source text is a music/noise tag"
        elif t > 20 and e > 0 and (e > 3 * t or e * 3 < t):
            reason = "english/source length mismatch"
        if reason:
            r["quality"]["flagged"] = True
            r["quality"]["reason"] = reason


def _overlap_text(start, end, eng_segs):
    parts = [e["text"] for e in eng_segs
             if min(end, e["end"]) - max(start, e["start"]) > 0]
    return " ".join(parts).strip()


def _assign_speaker(start, end, diar_segments):
    if not diar_segments:
        return "SPEAKER_00"
    best, best_ov = None, 0.0
    for d in diar_segments:
        ov = min(end, d["end"]) - max(start, d["start"])
        if ov > best_ov:
            best_ov, best = ov, d["speaker"]
    return best or diar_segments[0]["speaker"]
