"""Worker-environment self-test: imports + GPU compatibility, NO model weights.

Run by setup_diarization.ps1 and by `aegis doctor`. Prints a JSON summary on
stdout and exits non-zero on a hard failure so setup can stop loudly.
"""
from __future__ import annotations

import json
import sys


def main() -> int:
    report: dict = {"ok": True, "python": sys.version.split()[0], "errors": []}

    def need(name: str, key: str) -> object | None:
        try:
            mod = __import__(name)
            report[key] = getattr(mod, "__version__", "unknown")
            return mod
        except Exception as e:  # import failures must be visible, never hidden
            report["ok"] = False
            report["errors"].append(f"{name}: {type(e).__name__}: {e}")
            report[key] = None
            return None

    np = need("numpy", "numpy")
    torch = need("torch", "torch")
    need("torchaudio", "torchaudio")
    need("soundfile", "soundfile")

    # torchcodec: pyannote 4 declares it, and it is the decoder pyannote would
    # use IF it were handed a file path. This worker always hands pyannote a
    # preloaded {'waveform', 'sample_rate'} dict (pyannote's own documented
    # workaround), so a broken torchcodec does not stop diarization.
    # It is reported explicitly and never hidden, but it is NOT a hard failure.
    try:
        import torchcodec
        report["torchcodec"] = getattr(torchcodec, "__version__", "unknown")
        report["torchcodec_decode_available"] = True
        report["torchcodec_note"] = "native decoder loaded"
    except Exception as e:
        first = " ".join(str(e).split())[:200]
        report["torchcodec"] = None
        report["torchcodec_decode_available"] = False
        report["torchcodec_note"] = (
            f"native library unavailable ({type(e).__name__}: {first}); "
            f"NOT required by this worker, which passes pyannote a preloaded "
            f"waveform instead of a file path. On Windows this usually means "
            f"FFmpeg 'full-shared' DLLs are absent.")

    if np is not None and int(np.__version__.split(".")[0]) < 2:
        report["ok"] = False
        report["errors"].append(
            f"worker requires numpy>=2 (pyannote-core 6); found {np.__version__}")

    try:
        import pyannote.audio as pa
        from pyannote.audio import Pipeline  # noqa: F401
        report["pyannote.audio"] = pa.__version__
    except Exception as e:
        report["ok"] = False
        report["pyannote.audio"] = None
        report["errors"].append(f"pyannote.audio: {type(e).__name__}: {e}")

    # GPU compatibility (same check as the main env's aegis.gpu, duplicated here
    # because the worker environment cannot import the main package).
    if torch is not None:
        report["cuda_build"] = torch.version.cuda
        try:
            if torch.cuda.is_available():
                cap = torch.cuda.get_device_capability(0)
                arches = torch.cuda.get_arch_list()
                report["gpu"] = torch.cuda.get_device_name(0)
                report["capability"] = f"sm_{cap[0]}{cap[1]}"
                report["arch_list"] = arches
                compatible = any(
                    a.startswith("sm_") and a[3:].isdigit()
                    and int(a[3:-1] or 0) == cap[0] and int(a[-1]) <= cap[1]
                    for a in arches)
                report["gpu_compatible"] = bool(compatible)
                if not compatible:
                    report["ok"] = False
                    report["errors"].append(
                        f"torch build supports {arches} but GPU is "
                        f"sm_{cap[0]}{cap[1]}")
            else:
                report["gpu"] = None
                report["gpu_compatible"] = None       # CPU-only is allowed
        except Exception as e:
            report["errors"].append(f"gpu probe: {type(e).__name__}: {e}")

    print(json.dumps(report, indent=1))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
