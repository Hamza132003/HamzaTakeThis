"""Channel-preserving extraction with BOUNDED MEMORY (Phase 1 repair).

Evidence rule: the original file is untouched; every derived track is an
additive artifact written into the canonical artifact store.

Memory model
------------
The decoded recording is NEVER materialized as one array. ffmpeg decodes the
source to a native-rate/native-channel temp WAV on disk (an intermediate file,
not RAM), then `soundfile.blocks()` streams fixed-size blocks
(`BLOCK_FRAMES`, default 1 s at 48 kHz = 48000 frames) through the channel
split and mid/side matrix straight into per-track streaming writers. Peak RAM
is therefore O(BLOCK_FRAMES × channels), independent of duration.

Derived assets per source (unchanged from the first Phase 1 implementation):
- `channel_N.wav`  one lossless track per source channel (multichannel only);
- `mono_mix.wav`   equal-weight mean of all channels, scale 1/C;
- stereo only: `mid.wav` = (L+R)/2, `side.wav` = (L-R)/2 — the /2 convention
  bounds |mid|,|side| ≤ max(|L|,|R|) so matrixing cannot introduce clipping;
- mono sources get `mono_mix.wav` only; NO fake mid/side is fabricated;
- >2 channels keep every channel plus the documented compatibility downmix.

PCM format: 32-bit float WAV (`pcm_f32le`) via the deterministic streaming
writer in `wavio.py` — lossless for the float matrix math, cannot clip on
write, byte-reproducible (see that module for why libsndfile/scipy are not
used). No gain or normalization is ever applied.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from pipeline.contracts import DerivedAudio
from pipeline.ingest.immutable import content_id, stream_hashes, utc_now_iso
from pipeline.ingest.wavio import DeterministicWavWriter

# One second of 48 kHz audio per block. Bounded and deterministic: the block
# grid is a pure function of this constant, so artifact bytes do not depend on
# file length or machine state.
BLOCK_FRAMES = 48_000


def _ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def decode_native(src: Path, tmp_wav: Path) -> None:
    """Source → multichannel float32 WAV at native rate/channels (no resample,
    no downmix, no gain). Streams to disk; does not buffer audio in RAM."""
    cmd = [_ffmpeg_exe(), "-y", "-v", "error", "-i", str(src),
           "-map", "0:a:0",                       # documented selection rule
           "-c:a", "pcm_f32le", str(tmp_wav)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not tmp_wav.exists():
        raise RuntimeError(f"ffmpeg decode failed: {proc.stderr.strip()[:400]}")


def _contract(parent_id: str, parent_sha: str, role: str, transform: str,
              rel_path: str, sha: str, size: int, sr: int,
              duration: float) -> DerivedAudio:
    return DerivedAudio(
        asset_id=content_id(sha, "drv"), parent_asset_id=parent_id,
        parent_sha256=parent_sha, role=role, transform=transform,
        tool="ffmpeg(imageio-ffmpeg)+numpy(streaming)", rel_path=rel_path,
        sha256=sha, byte_size=size, created_utc=utc_now_iso(), sample_rate=sr,
        channels=1, sample_format="pcm_f32le", duration_sec=round(duration, 6),
        gain_adjusted=False)


def build_channel_assets(src: Path, assets_dir: Path, parent_id: str,
                         parent_sha: str,
                         block_frames: int = BLOCK_FRAMES) -> list[DerivedAudio]:
    """Stream per-channel/mono/mid/side assets into `assets_dir`.

    Returns the DerivedAudio contracts. Peak memory is O(block_frames × C);
    the decoded recording is never held whole in RAM.
    """
    import soundfile as sf

    assets_dir.mkdir(parents=True, exist_ok=True)
    tmp = assets_dir / "_native_decode.wav"
    decode_native(src, tmp)
    try:
        info = sf.info(str(tmp))
        sr, n_ch, n_frames = int(info.samplerate), int(info.channels), int(info.frames)

        # Plan the output tracks before touching audio.
        plan: list[tuple[str, str, str]] = []          # (role, transform, filename)
        if n_ch == 1:
            plan.append(("mono_mix",
                         "identity (source is mono; no matrixing applied)",
                         "mono_mix.wav"))
        else:
            for c in range(n_ch):
                plan.append((f"channel_{c}",
                             f"source channel {c}, unchanged (format conversion "
                             f"to pcm_f32le only)", f"channel_{c}.wav"))
            plan.append(("mono_mix",
                         f"mean(channel_0..channel_{n_ch - 1}) * (1/{n_ch}) "
                         f"(equal-weight compatibility downmix; no gain)",
                         "mono_mix.wav"))
            if n_ch == 2:
                plan.append(("mid", "(L + R) / 2  [L=channel_0, R=channel_1; "
                                    "/2 prevents matrix clipping]", "mid.wav"))
                plan.append(("side", "(L - R) / 2  [same convention]", "side.wav"))

        writers = {role: DeterministicWavWriter(assets_dir / fname, sr)
                   for role, _, fname in plan}
        try:
            # (frames, channels) blocks; always_2d keeps mono uniform.
            for block in sf.blocks(str(tmp), blocksize=block_frames,
                                   dtype="float32", always_2d=True):
                if block.size == 0:
                    continue
                b = block.T                                    # (C, frames)
                if n_ch == 1:
                    writers["mono_mix"].write(b[0])
                else:
                    for c in range(n_ch):
                        writers[f"channel_{c}"].write(b[c])
                    writers["mono_mix"].write(b.mean(axis=0))
                    if n_ch == 2:
                        writers["mid"].write((b[0] + b[1]) / 2.0)
                        writers["side"].write((b[0] - b[1]) / 2.0)
            for w in writers.values():
                w.close()                                       # atomic rename
        except BaseException:
            for w in writers.values():
                w.abort()
            raise
    finally:
        tmp.unlink(missing_ok=True)

    dur = n_frames / float(sr)
    out: list[DerivedAudio] = []
    for role, transform, fname in plan:
        path = assets_dir / fname
        sha, _, size = stream_hashes(path)          # hashed AFTER final creation
        out.append(_contract(parent_id, parent_sha, role, transform,
                             f"assets/{fname}", sha, size, sr, dur))
    return out
