"""Channel-preserving extraction (Phase 1).

Evidence rule: the original file is untouched; every derived track is an
additive artifact under `<out_dir>/ingest/`.

Derived assets per source:
- `channel_N.wav`  one lossless track per source channel (N ≥ 2 sources);
- `mono_mix.wav`   deterministic original-mix mono = arithmetic mean of all
                   channels, scale 1/C (documented; prevents clipping because
                   |mean| ≤ max|ch| for same-sign peaks and float storage
                   cannot wrap);
- stereo only:
  - `mid.wav`  = (L + R) / 2
  - `side.wav` = (L - R) / 2
  (scaling convention: the /2 factor bounds |mid|,|side| ≤ max(|L|,|R|) so the
  matrix cannot introduce clipping; recorded on each contract's `transform`.)
- >2 channels: per-channel tracks plus the documented compatibility downmix
  (`mono_mix.wav`, equal-weight mean). Individual tracks are never discarded.
- mono sources: `mono_mix.wav` only — NO fake mid/side is fabricated.

PCM format: 32-bit float WAV (`pcm_f32le`). Rationale: (a) lossless for every
numpy float32 operation used here — no requantization error on mid/side math;
(b) no clipping on write (float WAV holds |x|>1); (c) universally readable by
the existing stack (soundfile/librosa/ffmpeg). Trade-off: 2x the bytes of
16-bit PCM; recorded in docs/INGEST.md's disk-expansion notes.

Decoding uses the bundled ffmpeg (imageio-ffmpeg) to convert the source to a
single multichannel float32 WAV at the NATIVE sample rate and channel count
(no -ac/-ar flags), then numpy splits/matrixes channels. No gain or
normalization is ever applied (gain_adjusted=False on all contracts).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from pipeline.contracts import DerivedAudio
from pipeline.ingest.immutable import content_id, stream_hashes, utc_now_iso


def _ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def _decode_native(src: Path, tmp_wav: Path) -> None:
    """Source → multichannel float32 WAV at native rate/channels (no resample,
    no downmix, no gain)."""
    cmd = [_ffmpeg_exe(), "-y", "-v", "error", "-i", str(src),
           "-map", "0:a:0",                       # documented selection rule
           "-c:a", "pcm_f32le", str(tmp_wav)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not tmp_wav.exists():
        raise RuntimeError(f"ffmpeg decode failed: {proc.stderr.strip()[:400]}")


def _write_track(path: Path, data: np.ndarray, sr: int) -> tuple[str, int]:
    """Atomic float32 WAV write → (sha256, byte_size)."""
    import io

    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, np.asarray(data, dtype=np.float32), sr,
             format="WAV", subtype="FLOAT")
    payload = buf.getvalue()
    from pipeline.ingest.immutable import atomic_write_bytes
    sha = atomic_write_bytes(path, payload)
    return sha, len(payload)


def _contract(parent_id: str, parent_sha: str, role: str, transform: str,
              rel_path: str, sha: str, size: int, sr: int,
              duration: float) -> DerivedAudio:
    return DerivedAudio(
        asset_id=content_id(sha, "drv"), parent_asset_id=parent_id,
        parent_sha256=parent_sha, role=role, transform=transform,
        tool="ffmpeg(imageio-ffmpeg)+numpy", rel_path=rel_path, sha256=sha,
        byte_size=size, created_utc=utc_now_iso(), sample_rate=sr, channels=1,
        sample_format="pcm_f32le", duration_sec=round(duration, 6),
        gain_adjusted=False)


def build_channel_assets(src: Path, out_dir: Path, parent_id: str,
                         parent_sha: str) -> tuple[list[DerivedAudio], np.ndarray, int]:
    """Create per-channel/mono/mid/side assets under out_dir/ingest/.

    Returns (contracts, native_audio[C,N] float32, sample_rate) so diagnostics
    can reuse the decoded signal without re-decoding.
    """
    import soundfile as sf

    ingest_dir = out_dir / "ingest"
    ingest_dir.mkdir(parents=True, exist_ok=True)
    tmp = ingest_dir / "_native_decode.wav"
    _decode_native(src, tmp)
    try:
        data, sr = sf.read(str(tmp), dtype="float32", always_2d=True)  # (N, C)
    finally:
        tmp.unlink(missing_ok=True)
    audio = np.ascontiguousarray(data.T)                                # (C, N)
    n_ch, n = audio.shape
    dur = n / float(sr)
    out: list[DerivedAudio] = []

    def emit(role: str, transform: str, y: np.ndarray, fname: str) -> None:
        sha, size = _write_track(ingest_dir / fname, y, sr)
        out.append(_contract(parent_id, parent_sha, role, transform,
                             f"ingest/{fname}", sha, size, sr, dur))

    if n_ch == 1:
        emit("mono_mix", "identity (source is mono; no matrixing applied)",
             audio[0], "mono_mix.wav")
    else:
        for c in range(n_ch):
            emit(f"channel_{c}", f"source channel {c}, unchanged (format "
                 f"conversion to pcm_f32le only)", audio[c], f"channel_{c}.wav")
        emit("mono_mix", f"mean(channel_0..channel_{n_ch - 1}) * (1/{n_ch}) "
             f"(equal-weight compatibility downmix; no gain)",
             audio.mean(axis=0), "mono_mix.wav")
        if n_ch == 2:
            emit("mid", "(L + R) / 2  [L=channel_0, R=channel_1; /2 prevents "
                 "matrix clipping]", (audio[0] + audio[1]) / 2.0, "mid.wav")
            emit("side", "(L - R) / 2  [same convention]",
                 (audio[0] - audio[1]) / 2.0, "side.wav")
    return out, audio, sr
