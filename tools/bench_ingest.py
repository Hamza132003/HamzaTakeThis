"""Phase 1 ingest benchmark — engineering measurements on THIS machine only,
not universal performance claims. No models, no network.

Each ingest is measured in its OWN subprocess so the reported peak working set
is attributable to ingest alone (the parent's fixture generation would
otherwise dominate, and Windows peak_wset is a monotonic high-water mark that
cannot be reset within a process).

Run:  .venv/Scripts/python.exe tools/bench_ingest.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import psutil  # noqa: E402

BLOCK = 48_000


def _peak_mb() -> float:
    info = psutil.Process().memory_info()
    return getattr(info, "peak_wset", info.rss) / 1e6


def _mk_stereo_wav(path: Path, seconds: float, sr: int = 48000) -> None:
    """Stream the fixture to disk in blocks so generating it does not itself
    allocate the whole recording."""
    import soundfile as sf
    rng = np.random.default_rng(0)
    total = int(sr * seconds)
    with sf.SoundFile(str(path), "w", samplerate=sr, channels=2,
                      subtype="FLOAT") as fh:
        written = 0
        while written < total:
            n = min(BLOCK, total - written)
            t = (np.arange(written, written + n) / sr).astype(np.float64)
            left = (0.4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
            right = (0.4 * np.sin(2 * np.pi * 330 * t)
                     + 0.01 * rng.standard_normal(n)).astype(np.float32)
            fh.write(np.stack([left, right], axis=1))
            written += n


def _dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def _child(wav: str, out_dir: str, store: str) -> int:
    """Run one ingest and report its own peak working set."""
    from pipeline.ingest.ingest import ingest_file
    t0 = time.perf_counter()
    m = ingest_file(Path(wav), Path(out_dir), {"storage": {"artifact_dir": store}})
    dt = time.perf_counter() - t0
    print(json.dumps({
        "seconds": round(dt, 3), "peak_mb": round(_peak_mb(), 1),
        "assets": len(m.derived),
        "mode": m.condition_vector.analysis_mode if m.condition_vector else None,
        "store_rel": m.store_relative_dir, "ok": m.ok,
        "duplicate_of": m.source.duplicate_of,
    }))
    return 0


def _run_child(wav: Path, out_dir: Path, store: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([sys.executable, __file__, "--child", str(wav),
                        str(out_dir), str(store)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"child failed: {r.stderr[-400:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def main() -> int:
    from pipeline.ingest.immutable import stream_hashes

    print("AEGIS-X Phase 1 ingest benchmark (this machine only)")
    print("-" * 66)
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        store = tdp / "store"

        # 1. hashing throughput (256 MiB of incompressible bytes)
        blob = tdp / "blob.bin"
        blob.write_bytes(np.random.default_rng(1).bytes(256 * 1024 * 1024))
        t0 = time.perf_counter()
        stream_hashes(blob)
        dt = time.perf_counter() - t0
        print(f"hashing sha256+sha512 (single pass) : 256 MiB in {dt:.2f}s "
              f"= {256 / dt:.0f} MiB/s")
        blob.unlink()

        # 2. probe timing
        rep = tdp / "rep.wav"
        _mk_stereo_wav(rep, 60.0)
        from pipeline.ingest.media_probe import probe
        t0 = time.perf_counter()
        probe(rep)
        print(f"probe (60 s stereo)                 : "
              f"{(time.perf_counter() - t0) * 1000:.0f} ms")

        # 3. representative + long + very long, each in its own process
        cases = [("60 s stereo", rep, 60.0)]
        long_wav = tdp / "long.wav"
        _mk_stereo_wav(long_wav, 720.0)
        cases.append(("12 min stereo", long_wav, 720.0))
        longer = tdp / "longer.wav"
        _mk_stereo_wav(longer, 2400.0)
        cases.append(("40 min stereo", longer, 2400.0))

        for i, (label, wav, secs) in enumerate(cases):
            res = _run_child(wav, tdp / f"out{i}", store)
            src_mb = wav.stat().st_size / 1e6
            canon = store / res["store_rel"]
            exp = _dir_size(canon) / wav.stat().st_size
            print(f"{label:<20} src {src_mb:7.0f} MB : {res['seconds']:6.2f}s "
                  f"RTF {res['seconds'] / secs:.4f} | peak RAM "
                  f"{res['peak_mb']:6.0f} MB | mode {res['mode']:<15} "
                  f"| store expansion {exp:.2f}x")

        # 4. duplicate ingest
        dup = tdp / "dup.wav"
        dup.write_bytes(rep.read_bytes())
        res = _run_child(dup, tdp / "out_dup", store)
        print(f"duplicate ingest                    : {res['seconds']:.2f}s, "
              f"duplicate_of={res['duplicate_of']} "
              f"(artifacts re-derived in Phase 1; content-addressed reuse of "
              f"the canonical dir; skip-work arrives with the Phase 12 DAG)")
        print(f"parent process peak RAM             : {_peak_mb():.0f} MB "
              f"(fixture generation only)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        sys.exit(_child(*sys.argv[2:5]))
    sys.exit(main())
