"""`python -m aegis doctor` — environment, GPU-compatibility, and gate checks.

Exit codes: 0 = healthy; 1 = hard failure (do not run the pipeline).
Performs no downloads and touches no models.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from .gpu import check_torch_cuda

ROOT = Path(__file__).resolve().parent.parent

CORE_MODULES = ["numpy", "librosa", "torch", "faster_whisper", "transformers",
                "flask", "yaml", "soundfile"]


def _have(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def _onedrive_path(p: Path) -> bool:
    return "onedrive" in str(p).lower()


def run() -> int:
    print("AEGIS-X PRIME doctor")
    print("-" * 50)
    hard_fail = False

    print(f"python: {sys.version.split()[0]} @ {sys.executable}")

    missing = [m for m in CORE_MODULES if not _have(m)]
    for m in CORE_MODULES:
        print(f"  [{'MISS' if m in missing else ' OK '}] module {m}")
    if missing:
        hard_fail = True

    gpu = check_torch_cuda()
    print(f"torch: {gpu['torch_version']} (cuda build: {gpu['cuda_version']})")
    if gpu["available"]:
        cap = gpu["capability"]
        print(f"GPU: {gpu['device_name']} (sm_{cap[0]}{cap[1]})")
        print(f"  arch list: {gpu['arch_list']}")
        if gpu["compatible"]:
            print(f"  [ OK ] {gpu['reason']}")
        else:
            print(f"  [FAIL] {gpu['reason']}")
            print("         Fix: install a matching build (see docs/HARDWARE_SETUP.md),")
            print("         e.g. pip install torch torchaudio --index-url "
                  "https://download.pytorch.org/whl/cu128")
            hard_fail = True
    else:
        print("GPU: none detected — CPU mode (functional, slower).")

    # numpy pin required by clearvoice
    try:
        import numpy
        ok = tuple(int(x) for x in numpy.__version__.split(".")[:2]) < (2, 0)
        print(f"  [{' OK ' if ok else 'WARN'}] numpy {numpy.__version__} "
              f"({'<2 as required by clearvoice' if ok else 'clearvoice requires <2.0'})")
    except Exception:
        pass

    # governance gates
    blocked = (ROOT / "RELEASE_BLOCKED.md").exists()
    gate_txt = "BLOCKED (expected - license deferred)" if blocked else "open"
    print(f"  [{'INFO' if blocked else ' OK '}] release gate: {gate_txt}")
    tracked_token = (ROOT / "hf_token.txt").exists()
    print(f"  [INFO] hf_token.txt present locally: {tracked_token} (gitignored either way)")

    # storage / OneDrive
    if _onedrive_path(ROOT):
        print("  [WARN] repository lives inside a OneDrive-synced path (risk R-8).")
        print("         Configure storage.artifact_dir outside OneDrive before Phase 12;")
        print("         for sensitive audio use --outdir on a non-synced path (PRIVACY.md).")

    print("-" * 50)
    print("RESULT:", "FAIL — do not run the pipeline" if hard_fail else "OK")
    return 1 if hard_fail else 0


if __name__ == "__main__":
    sys.exit(run())
