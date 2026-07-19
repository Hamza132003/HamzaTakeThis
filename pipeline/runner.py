"""End-to-end orchestration for a single recording.

Used by both the CLI (main.py) and the web dashboard. Returns a rich report
dict and writes report.json / report.md plus audio + spectrogram artifacts.

Models are shared through pipeline.models.MANAGER so a batch of files only
pays the multi-GB load cost once. VRAM rule (6 GB): one heavy model on the
GPU at a time — Whisper is evicted before NLLB loads.
"""
from __future__ import annotations

from pathlib import Path

from . import (audio, separation, noise_analysis, diarization, speakers,
               transcription, translation, emotion, summary, spectrogram,
               report)
from .models import MANAGER
from .utils import (log, stage, free_cuda, set_cancel_check, check_cancel,
                    JobCancelled)


def _rel(p, out_dir: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(out_dir.resolve())).replace("\\", "/")
    except Exception:
        return str(p)


def process_file(input_path, cfg: dict, device: str, progress=None,
                 should_cancel=None) -> dict:
    in_path = Path(input_path)
    out_dir = Path(cfg.get("output_dir", "outputs")) / in_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # Long stages (Whisper decode, NLLB batches) poll this via check_cancel(),
    # so cancellation lands mid-stage, not just at stage boundaries.
    set_cancel_check(should_cancel)

    steps = ["Extract audio", "Separate voice/noise", "Analyse noise",
             "Diarize", "Un-mix speakers", "Transcribe", "Translate",
             "Sentiment", "AI summary", "Spectrograms", "Report"]
    total = len(steps)

    def tick(i, frac=0.0):
        check_cancel()
        if progress:
            try:
                progress(steps[i], i + 1, total, frac)
            except TypeError:            # older 3-arg callbacks
                progress(steps[i], i + 1, total)

    warnings = []

    # 1. audio
    tick(0)
    with stage("Extract audio"):
        au = audio.extract_audio(in_path, out_dir,
                                 cfg["audio"]["sample_rate"],
                                 cfg["audio"].get("hq_sample_rate", 48000))
        dur = audio.duration_seconds(au["work_wav"])

    # 2. separation (48 kHz enhancement; also emits voice_16k.wav)
    tick(1)
    with stage("Separate voice / noise"):
        sep = separation.separate(au, out_dir, cfg["separation"], device,
                                  models=MANAGER)
    voice16 = sep.get("voice_16k", sep["voice"])

    # 3. noise analysis
    tick(2)
    with stage("Analyse noise (what & when)"):
        noise = noise_analysis.analyse_noise(sep["noise"], cfg["noise"], device)

    # 4. diarization (16 kHz)
    tick(3)
    with stage("Diarize (count speakers)"):
        diar = diarization.diarize(voice16, cfg["diarization"], device,
                                   models=MANAGER)
    if diar.get("warning"):
        warnings.append(diar["warning"])

    # 5. per-speaker tracks + overlap un-mix (only if >1 speaker)
    tick(4)
    speaker_tracks = None
    overlaps = []
    spk_method = "single-speaker"
    if diar["num_speakers"] > 1:
        with stage("Un-mix overlapping speakers"):
            spk = speakers.build_speaker_tracks(
                voice16, diar["segments"], out_dir,
                cfg.get("speakers", {}), device, models=MANAGER)
            speaker_tracks = spk["tracks"]
            overlaps = spk["overlaps"]
            spk_method = spk["method"]

    # 6. transcription (tries the original AND the isolated voice)
    tick(5)
    with stage("Transcribe speech"):
        src = str(cfg["transcription"].get("source", "both")).lower()
        if src == "voice":
            wavs = [voice16]
        elif src == "original":
            wavs = [au["work_wav"]]
        else:                                    # "both" - pick the better
            wavs = [au["work_wav"], voice16]
        segs = transcription.transcribe(
            wavs, diar["segments"], cfg["transcription"], device,
            speaker_tracks=speaker_tracks,
            speech_regions=diar.get("speech_regions"),
            overlaps=overlaps, models=MANAGER)

    # 7. translation — Whisper (~3.2 GB) and NLLB-1.3B (~2.9 GB) cannot share
    # a 6 GB GPU, so evict Whisper first (reloads from local cache next file).
    tick(6)
    if device == "cuda":
        MANAGER.evict(transcription.WHISPER_KEY)
        MANAGER.release_gpu()
    with stage("Translate to Arabic"):
        segs = translation.translate_segments(segs, cfg["translation"], device,
                                              models=MANAGER)

    # 8. emotion
    tick(7)
    if device == "cuda":
        MANAGER.release_gpu()
    with stage("Sentiment (vocal tone)"):
        segs = emotion.analyse_emotion(voice16, segs, cfg["emotion"], device)

    # 9. AI summary of the whole recording (local LLM; needs the GPU alone)
    tick(8)
    if device == "cuda":
        MANAGER.release_gpu()
    with stage("AI summary"):
        ai = summary.summarize(segs, noise, diar, overlaps, dur,
                               cfg.get("summary", {}), device, models=MANAGER)
    if device == "cuda":
        MANAGER.release_gpu()

    # 10. spectrograms
    tick(9)
    with stage("Render spectrograms"):
        tracks = {
            "original": (au["work_wav"], "Original (voice + noise)", "magma"),
            "voice": (sep["voice"], "Isolated voice", "viridis"),
            "noise": (sep["noise"], "Isolated noise", "inferno"),
        }
        if speaker_tracks:
            for spk_id, wav in speaker_tracks.items():
                tracks[f"speaker_{spk_id}"] = (wav, f"{spk_id}", "cividis")
        specs, waves = spectrogram.render_all(tracks, out_dir)
        timeline = spectrogram.render_speaker_timeline(
            diar["segments"], dur, out_dir / "spectrograms" / "timeline.png")

    # 10. assemble + write report
    tick(9)
    with stage("Write report"):
        rep = {
            "input": str(in_path),
            "filename": in_path.name,
            "duration_sec": round(dur, 2),
            "device": device,
            "language_setting": str(cfg["transcription"].get("language", "auto")),
            "separation_method": sep["method"],
            "sample_rate": sep.get("sample_rate", 16000),
            "diarization_method": diar["method"],
            "num_speakers": diar["num_speakers"],
            "speakers": diar["speakers"],
            "overlap_method": spk_method,
            "overlaps": overlaps,
            "languages_detected": sorted({s.get("language", "?") for s in segs}),
            "warnings": warnings,
            "summary": ai,
            "pipeline": {
                "separation": sep["method"],
                "diarization": diar["method"],
                "overlap": spk_method,
                "asr_model": str(cfg["transcription"].get("model", "large-v3")),
                "translation_model": str(cfg["translation"].get("model", "")),
            },
            "noise": noise,
            "speech": segs,
            "audio": {
                "original": _rel(au["hq_wav"], out_dir),
                "voice": _rel(sep["voice"], out_dir),
                "noise": _rel(sep["noise"], out_dir),
            },
            "speaker_tracks": {k: _rel(v, out_dir)
                               for k, v in (speaker_tracks or {}).items()},
            "spectrograms": {k: _rel(v, out_dir) for k, v in specs.items()},
            "waveforms": {k: _rel(v, out_dir) for k, v in waves.items()},
            "timeline": _rel(timeline, out_dir) if timeline else None,
        }
        report.write(rep, out_dir)
    free_cuda()
    log(f"✅ Done: {out_dir / 'report.md'}")
    return rep
