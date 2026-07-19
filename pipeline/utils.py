"""Shared helpers: device selection, logging, timing, audio I/O."""
from __future__ import annotations

import sys
import time
import contextlib
from pathlib import Path


# Windows consoles often default to a legacy codepage (e.g. cp1256) that can't
# encode our status glyphs. Force UTF-8 on the streams so logging never crashes.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Corporate networks intercept TLS with their own root certificate, which
# Python's bundled CA store rejects (SSL: CERTIFICATE_VERIFY_FAILED on
# huggingface.co). Use the Windows system certificate store instead — it
# trusts the corporate root, and behaves identically on normal networks.
try:
    import truststore as _truststore
    _truststore.inject_into_ssl()
except Exception:
    pass

_sb_patched = False

# ------------------------------------------------------------- cancellation
# A process-wide cancel hook so long-running stages (Whisper decode loops,
# NLLB batches) can abort mid-stage without threading a callback through
# every function signature. Only one job runs at a time (webapp serializes
# them), so a single global is safe.
_cancel_check = None


class JobCancelled(Exception):
    """Raised from check_cancel() when the caller requested cancellation."""


def set_cancel_check(fn) -> None:
    """Install (or clear, with None) the job-wide cancellation probe."""
    global _cancel_check
    _cancel_check = fn


def check_cancel() -> None:
    """Raise JobCancelled if the current job has been asked to stop.
    Broad `except Exception` handlers must re-raise this explicitly."""
    if _cancel_check is not None and _cancel_check():
        raise JobCancelled()


def guard_speechbrain_lazy() -> None:
    """SpeechBrain exposes optional integrations (k2, wordemb, ...) as lazy
    modules whose __getattr__ raises ImportError when the optional dep is
    missing. Transformers/inspect scan modules with hasattr(), which only
    swallows AttributeError - so those ImportErrors crash translation/emotion.

    Patch LazyModule so missing/dunder lookups raise AttributeError instead,
    which makes hasattr() correctly return False. Idempotent; safe if
    SpeechBrain isn't installed.

    Also forces SpeechBrain's model fetching to COPY files instead of
    creating symlinks: on Windows, symlink creation needs admin rights or
    Developer Mode and otherwise fails with WinError 1314."""
    global _sb_patched
    if _sb_patched:
        return
    try:
        from speechbrain.utils import importutils as _iu
        _orig = _iu.LazyModule.__getattr__

        def _safe(self, attr):
            if attr.startswith("__") and attr.endswith("__"):
                raise AttributeError(attr)
            try:
                return _orig(self, attr)
            except ImportError as e:
                raise AttributeError(attr) from e

        _iu.LazyModule.__getattr__ = _safe
        _sb_patched = True
    except Exception:
        pass

    try:
        from speechbrain.utils import fetching as _f
        if (not getattr(_f, "_vi_copy_patch", False)
                and hasattr(_f, "link_with_strategy")
                and hasattr(_f, "LocalStrategy")):
            _orig_link = _f.link_with_strategy

            def _no_symlink(src, dst, local_strategy, *a, **kw):
                if local_strategy == _f.LocalStrategy.SYMLINK:
                    local_strategy = _f.LocalStrategy.COPY
                return _orig_link(src, dst, local_strategy, *a, **kw)

            _f.link_with_strategy = _no_symlink
            _f._vi_copy_patch = True
    except Exception:
        pass


def log(msg: str) -> None:
    """Print a timestamped status line (flushed so it streams live)."""
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    try:
        print(line, flush=True)
    except Exception:
        print(line.encode("ascii", "replace").decode("ascii"), flush=True)


@contextlib.contextmanager
def stage(name: str):
    """Context manager that logs start/end + duration of a pipeline stage."""
    log(f"► {name} ...")
    t0 = time.time()
    try:
        yield
    finally:
        log(f"✓ {name} done ({time.time() - t0:.1f}s)")


def resolve_device(pref: str = "auto") -> str:
    """Return 'cuda' or 'cpu' based on preference and availability.

    Fails LOUDLY when a CUDA device is present but the installed torch build
    cannot execute kernels on it (e.g. cu124 build on an sm_120 RTX 50-series).
    torch.cuda.is_available() still returns True in that state, and previously
    every GPU stage crashed into its fallback chain, presenting degraded output
    as a successful GPU run. Pass --device cpu for a deliberate CPU run.
    """
    if pref == "cpu":
        return "cpu"
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
    except Exception:
        cuda_ok = False
    if cuda_ok:
        try:
            from aegis.gpu import check_torch_cuda
            probe = check_torch_cuda()
        except ImportError:
            log("WARNING: aegis.gpu probe unavailable; skipping GPU-compat check.")
            probe = {"compatible": True, "reason": "probe unavailable"}
        if not probe["compatible"]:
            raise RuntimeError(
                f"GPU/torch mismatch: {probe['reason']}. "
                f"Install a compatible build (see docs/HARDWARE_SETUP.md or run "
                f"'python -m aegis doctor'), or pass --device cpu to run on CPU "
                f"deliberately.")
        return "cuda"
    if pref == "cuda":
        log("WARNING: cuda requested but not available; falling back to cpu.")
    return "cpu"


def free_cuda() -> None:
    """Release cached GPU memory between heavy stages (6GB VRAM is tight)."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def load_wav_mono(path: Path, sample_rate: int):
    """Load an audio file as float32 mono at the given sample rate."""
    import librosa
    y, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    return y


def fmt_ts(seconds: float) -> str:
    """Format seconds as M:SS.s for human-readable timelines."""
    m, s = divmod(max(0.0, float(seconds)), 60)
    return f"{int(m)}:{s:04.1f}"
