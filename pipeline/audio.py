"""Stage 1: pull audio out of the video and normalise it.

Uses the ffmpeg binary bundled with imageio-ffmpeg, so no system ffmpeg
install is required.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from .utils import log


def _ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        # Fall back to a system ffmpeg if the user has one on PATH.
        return "ffmpeg"


def extract_audio(input_path: Path, out_dir: Path, sample_rate: int,
                  hq_sample_rate: int = 48000) -> dict:
    """Extract two WAVs from the input:

    - work_wav:  mono @ sample_rate (16k) for the speech models
    - hq_wav:    mono @ hq_sample_rate (48k) for full-band enhancement

    Returns a dict of paths. Works for video *or* audio inputs (including
    browser MediaRecorder .webm/.mp4 blobs).
    """
    ffmpeg = _ffmpeg_exe()
    out_dir.mkdir(parents=True, exist_ok=True)
    work_wav = out_dir / "audio_16k_mono.wav"
    hq_wav = out_dir / "audio_48k_mono.wav"

    for target, args in (
        (work_wav, ["-ac", "1", "-ar", str(sample_rate)]),
        (hq_wav, ["-ac", "1", "-ar", str(hq_sample_rate)]),
    ):
        cmd = [ffmpeg, "-y", "-i", str(input_path), "-vn",
               "-acodec", "pcm_s16le", *args, str(target)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not target.exists():
            raise RuntimeError(
                f"ffmpeg failed to extract audio to {target.name}.\n"
                f"{proc.stderr[-800:]}"
            )

    log(f"Extracted audio → {work_wav.name}, {hq_wav.name}")
    return {"work_wav": work_wav, "hq_wav": hq_wav}


def duration_seconds(wav_path: Path) -> float:
    import soundfile as sf
    info = sf.info(str(wav_path))
    return info.frames / float(info.samplerate)
