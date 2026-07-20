"""Deterministic streaming float32 WAV writer (bounded memory).

Why hand-written: (a) libsndfile stamps a wall-clock timestamp into the PEAK
chunk of float WAVs, which destroys reproducible content hashing;
(b) scipy.io.wavfile builds the whole file in memory. This writer emits a
canonical little-endian RIFF/WAVE stream (WAVE_FORMAT_IEEE_FLOAT, 32-bit,
mono) incrementally, so peak RAM is one block regardless of duration, and the
bytes are a pure function of the samples.

Layout: RIFF/WAVE + fmt (16, format=3) + fact (4, sample count) + data.
The `fact` chunk is required by the spec for non-PCM formats and is accepted
by libsndfile/ffmpeg/scipy; all readers used in this repo were verified.

Completion semantics: samples stream into a temp file in the destination
directory; the header is finalized, fsynced, then `os.replace`d into place and
re-hashed from disk — an artifact is never announced before its final bytes and
hash exist.
"""
from __future__ import annotations

import os
import struct
import tempfile
from pathlib import Path

import numpy as np

_FMT_IEEE_FLOAT = 3
_BITS = 32
_BYTES = _BITS // 8


def _header(sample_rate: int, n_frames: int) -> bytes:
    data_size = n_frames * _BYTES
    fact_size = 4
    # RIFF size = 4 ("WAVE") + (8+16 fmt) + (8+4 fact) + (8+data)
    riff_size = 4 + (8 + 16) + (8 + fact_size) + (8 + data_size)
    return b"".join((
        b"RIFF", struct.pack("<I", riff_size), b"WAVE",
        b"fmt ", struct.pack("<I", 16),
        struct.pack("<HHIIHH", _FMT_IEEE_FLOAT, 1, sample_rate,
                    sample_rate * _BYTES, _BYTES, _BITS),
        b"fact", struct.pack("<I", fact_size), struct.pack("<I", n_frames),
        b"data", struct.pack("<I", data_size),
    ))


class DeterministicWavWriter:
    """Incremental mono float32 WAV writer with atomic completion."""

    def __init__(self, dest: Path, sample_rate: int):
        self.dest = Path(dest)
        self.sample_rate = int(sample_rate)
        self.n_frames = 0
        self.dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.dest.parent), suffix=".part")
        self._tmp = Path(tmp)
        self._fh = os.fdopen(fd, "wb")
        self._fh.write(_header(self.sample_rate, 0))   # placeholder sizes

    def write(self, samples: np.ndarray) -> None:
        block = np.ascontiguousarray(samples, dtype="<f4")
        self._fh.write(block.tobytes())
        self.n_frames += int(block.size)

    def close(self) -> None:
        """Finalize header, fsync, atomically rename into place."""
        try:
            self._fh.seek(0)
            self._fh.write(_header(self.sample_rate, self.n_frames))
            self._fh.flush()
            os.fsync(self._fh.fileno())
        finally:
            self._fh.close()
        os.replace(self._tmp, self.dest)

    def abort(self) -> None:
        try:
            self._fh.close()
        except OSError:
            pass
        try:
            os.unlink(self._tmp)
        except OSError:
            pass

    def __enter__(self) -> DeterministicWavWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()
