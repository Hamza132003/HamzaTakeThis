"""GPU / PyTorch-build compatibility checks (pure logic, unit-testable).

CUDA binary compatibility: a kernel compiled for sm_XY runs on hardware with the
same major version and minor >= Y. A `compute_XY` (PTX) entry can be JIT-compiled
forward onto any capability >= XY. A build whose arch list satisfies neither for
the detected device cannot execute kernels on it — torch.cuda.is_available() will
still return True, which is exactly the silent-failure trap this module closes.
"""
from __future__ import annotations

import re


def parse_arch(entry: str) -> tuple[str, int, int] | None:
    """'sm_120' -> ('sm', 12, 0);  'compute_90' -> ('compute', 9, 0)."""
    m = re.fullmatch(r"(sm|compute)_(\d+)([a-z]?)", entry.strip())
    if not m:
        return None
    kind, digits = m.group(1), m.group(2)
    # Capability digits: last digit is minor, the rest are major (sm_120 = 12.0).
    major, minor = int(digits[:-1]), int(digits[-1])
    return (kind, major, minor)


def is_compatible(capability: tuple[int, int], arch_list: list[str]) -> tuple[bool, str]:
    """Return (compatible, human-readable reason)."""
    cmaj, cmin = capability
    for entry in arch_list:
        parsed = parse_arch(entry)
        if parsed is None:
            continue
        kind, maj, minor = parsed
        if kind == "sm" and maj == cmaj and minor <= cmin:
            return True, f"binary kernels present ({entry} runs on sm_{cmaj}{cmin})"
        if kind == "compute" and (maj, minor) <= (cmaj, cmin):
            return True, f"PTX forward-compatible ({entry} JITs onto sm_{cmaj}{cmin})"
    return False, (
        f"installed torch build supports {arch_list} but the GPU is "
        f"sm_{cmaj}{cmin}; kernels cannot execute on this device"
    )


def check_torch_cuda() -> dict:
    """Probe the installed torch against the detected GPU.

    Returns a dict: {available, compatible, reason, device_name, capability,
    arch_list, torch_version, cuda_version}. Never raises.
    """
    out: dict = {"available": False, "compatible": True, "reason": "no CUDA device",
                 "device_name": None, "capability": None, "arch_list": [],
                 "torch_version": None, "cuda_version": None}
    try:
        import torch
        out["torch_version"] = torch.__version__
        out["cuda_version"] = torch.version.cuda
        if not torch.cuda.is_available():
            return out
        out["available"] = True
        cap = torch.cuda.get_device_capability(0)
        out["device_name"] = torch.cuda.get_device_name(0)
        out["capability"] = cap
        out["arch_list"] = torch.cuda.get_arch_list()
        ok, reason = is_compatible(cap, out["arch_list"])
        out["compatible"], out["reason"] = ok, reason
    except Exception as e:  # torch missing or probe failure: report, don't crash
        out["compatible"] = True
        out["reason"] = f"probe skipped ({e})"
    return out
