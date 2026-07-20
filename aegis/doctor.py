"""`python -m aegis doctor` — environment, GPU-compatibility, and gate checks.

Exit codes: 0 = healthy; 1 = hard failure (do not run the pipeline).
Performs no downloads and touches no models.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from .gpu import check_torch_cuda

ROOT = Path(__file__).resolve().parent.parent

# pyannote is deliberately NOT here: diarization runs in an isolated worker
# environment (docs/DIARIZATION_RUNTIME_ARCHITECTURE.md), and importing it in
# the main env is neither expected nor supported.
CORE_MODULES = ["numpy", "librosa", "torch", "faster_whisper", "transformers",
                "flask", "yaml", "soundfile", "clearvoice"]


def _have(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def _onedrive_path(p: Path) -> bool:
    return "onedrive" in str(p).lower()


def _load_cfg() -> dict:
    try:
        import yaml
        with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _diarization_section(cfg: dict) -> bool:
    """Report the isolated worker environment. Returns True on hard failure
    (diarization enabled but unusable)."""
    print()
    print("DIARIZATION WORKER (isolated environment)")
    print("-" * 50)
    enabled = bool((cfg.get("diarization") or {}).get("enabled", True))
    try:
        from workers.diarization_worker.client import (
            resolve_worker_python,
            worker_selftest,
        )
    except Exception as e:
        print(f"  [FAIL] worker client unavailable: {type(e).__name__}: {e}")
        return enabled

    worker_py = resolve_worker_python(cfg)
    print(f"  configured interpreter: {worker_py}")
    if not worker_py.exists():
        print("  [FAIL] worker environment NOT INSTALLED")
        print("         run ./setup_diarization.ps1 to create it")
        if enabled:
            print()
            print("  *** DIARIZATION UNAVAILABLE ***")
            print("  The pipeline will still run, but speaker detection will "
                  "fall back to")
            print("  single-speaker segmentation. That is NOT real diarization.")
        return enabled

    st = worker_selftest(cfg)
    for key, label in (("python", "python"), ("numpy", "numpy"),
                       ("torch", "torch"), ("torchaudio", "torchaudio"),
                       ("pyannote.audio", "pyannote.audio"),
                       ("soundfile", "soundfile")):
        val = st.get(key)
        print(f"  [{' OK ' if val else 'FAIL'}] {label:<16} {val or 'unavailable'}")
    # torchcodec is reported explicitly, never hidden, but a broken native
    # decoder does not stop this worker (it uses preloaded waveforms).
    tc_ok = st.get("torchcodec_decode_available")
    print(f"  [{' OK ' if tc_ok else 'WARN'}] {'torchcodec':<16} "
          f"{st.get('torchcodec') or 'native decoder unavailable'}")
    if not tc_ok and st.get("torchcodec_note"):
        print(f"         {st['torchcodec_note'][:200]}")
    if st.get("gpu"):
        print(f"  [{' OK ' if st.get('gpu_compatible') else 'FAIL'}] GPU "
              f"{st['gpu']} ({st.get('capability')})")
    else:
        print("  [INFO] GPU not visible to the worker (CPU mode)")
    for err in st.get("errors", []):
        print(f"  [FAIL] {err}")

    # model cache + token, without ever displaying the token
    cache_state, token_present = _diar_cache_and_token(cfg)
    print(f"  [INFO] model cache      : {cache_state}")
    print(f"  [INFO] HF token         : "
          f"{'available (value not displayed)' if token_present else 'NOT available'}")
    offline_ready = st.get("ok") and cache_state.startswith("present")
    print(f"  [{' OK ' if offline_ready else 'WARN'}] offline readiness: "
          f"{'cached weights present' if offline_ready else 'not proven - see docs'}")

    if enabled and not st.get("ok"):
        print()
        print("  *** DIARIZATION UNAVAILABLE ***")
        print("  Worker environment is invalid; speaker detection will fall "
              "back to")
        print("  single-speaker segmentation. That is NOT real diarization.")
        return True
    return False


def _diar_cache_and_token(cfg: dict) -> tuple[str, bool]:
    dcfg = cfg.get("diarization") or {}
    model_id = str(dcfg.get("model", "pyannote/speaker-diarization-community-1"))
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    folder = cache / ("models--" + model_id.replace("/", "--"))
    if folder.exists():
        snaps = list((folder / "snapshots").glob("*")) if (folder / "snapshots").exists() else []
        rev = snaps[0].name[:12] if snaps else "unknown-revision"
        state = f"present ({model_id} @ {rev})"
    else:
        state = f"absent ({model_id} not downloaded)"
    token = bool(dcfg.get("hf_token") or os.environ.get("HF_TOKEN")
                 or os.environ.get("HUGGINGFACE_TOKEN")
                 or (ROOT / "hf_token.txt").exists())
    return state, token


def run() -> int:
    cfg = _load_cfg()
    print("AEGIS-X PRIME doctor")
    print("=" * 50)
    print("MAIN ENVIRONMENT")
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

    diar_fail = _diarization_section(cfg)

    print()
    print("=" * 50)
    if hard_fail:
        print("RESULT: FAIL — do not run the pipeline")
    elif diar_fail:
        print("RESULT: DEGRADED — main pipeline OK, DIARIZATION UNAVAILABLE")
        print("        Speaker detection will fall back to single-speaker "
              "segmentation,")
        print("        which must never be reported as real diarization.")
    else:
        print("RESULT: OK")
    # A missing/invalid diarization worker is a real limitation, not a pass.
    return 1 if (hard_fail or diar_fail) else 0


if __name__ == "__main__":
    sys.exit(run())
