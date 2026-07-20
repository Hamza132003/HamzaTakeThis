"""Phase 1 ingest benchmark — engineering measurements on THIS machine only,
not universal performance claims. No models, no network.

Run:  .venv/Scripts/python.exe tools/bench_ingest.py
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import psutil  # noqa: E402
from scipy.io import wavfile  # noqa: E402

from pipeline.ingest.immutable import stream_hashes  # noqa: E402
from pipeline.ingest.ingest import ingest_file  # noqa: E402
from pipeline.ingest.media_probe import probe  # noqa: E402


def _mk_stereo_wav(path: Path, seconds: float, sr: int = 48000) -> None:
    t = np.arange(int(sr * seconds)) / sr
    left = (0.4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    right = (0.4 * np.sin(2 * np.pi * 330 * t)
             + 0.01 * np.random.default_rng(0).standard_normal(len(t))
             ).astype(np.float32)
    wavfile.write(str(path), sr, np.stack([left, right], axis=1))


def _dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def main() -> int:
    proc = psutil.Process()
    print("AEGIS-X Phase 1 ingest benchmark (this machine only)")
    print("-" * 60)
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        cfg = {"storage": {"artifact_dir": str(tdp / "store")}}

        # 1. hashing throughput on 256 MiB of random bytes
        blob = tdp / "blob.bin"
        data = np.random.default_rng(1).bytes(256 * 1024 * 1024)
        blob.write_bytes(data)
        t0 = time.perf_counter()
        stream_hashes(blob)
        dt = time.perf_counter() - t0
        print(f"hashing (sha256+sha512 single pass): 256 MiB in {dt:.2f}s "
              f"= {256 / dt:.0f} MiB/s")

        # 2. representative 60 s stereo file
        rep = tdp / "rep.wav"
        _mk_stereo_wav(rep, 60.0)
        src_size = rep.stat().st_size
        t0 = time.perf_counter()
        probe(rep)
        t_probe = time.perf_counter() - t0

        out = tdp / "out_rep"
        t0 = time.perf_counter()
        m = ingest_file(rep, out, cfg)
        t_ingest = time.perf_counter() - t0
        exp = _dir_size(out) / src_size
        assert m.condition_vector is not None, m.failure_reason
        print(f"probe (60s stereo): {t_probe * 1000:.0f} ms")
        print(f"full ingest (60s stereo, {src_size / 1e6:.0f} MB): "
              f"{t_ingest:.2f}s = RTF {t_ingest / 60:.3f} "
              f"({len(m.derived)} assets, "
              f"{len(m.condition_vector.conditions)} conditions)")
        print(f"disk expansion (all derived / source): {exp:.2f}x")

        # 3. long recording (12 min stereo) — windowed diagnostics path
        long_wav = tdp / "long.wav"
        _mk_stereo_wav(long_wav, 720.0)
        t0 = time.perf_counter()
        ml = ingest_file(long_wav, tdp / "out_long", cfg)
        t_long = time.perf_counter() - t0
        assert ml.condition_vector is not None, ml.failure_reason
        print(f"long file (12 min stereo): {t_long:.2f}s = RTF {t_long / 720:.4f}; "
              f"diagnostics mode: {ml.condition_vector.analysis_mode}")

        # 4. duplicate-ingest behaviour
        dup = tdp / "dup.wav"
        dup.write_bytes(rep.read_bytes())
        t0 = time.perf_counter()
        md = ingest_file(dup, tdp / "out_dup", cfg)
        t_dup = time.perf_counter() - t0
        print(f"duplicate ingest: {t_dup:.2f}s; duplicate_of="
              f"{md.source.duplicate_of} (artifacts re-derived in Phase 1; "
              f"reuse arrives with the Phase 12 DAG store)")

        print(f"peak process RAM during run: "
              f"{proc.memory_info().peak_wset / 1e6:.0f} MB"
              if hasattr(proc.memory_info(), "peak_wset")
              else f"current RSS: {proc.memory_info().rss / 1e6:.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
