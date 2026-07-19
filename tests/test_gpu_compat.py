"""Unit tests for aegis.gpu — the silent-GPU-failure guard (pure logic)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aegis.gpu import is_compatible, parse_arch


def test_parse_arch():
    assert parse_arch("sm_120") == ("sm", 12, 0)
    assert parse_arch("sm_86") == ("sm", 8, 6)
    assert parse_arch("compute_90") == ("compute", 9, 0)
    assert parse_arch("sm_90a") == ("sm", 9, 0)
    assert parse_arch("garbage") is None


def test_blackwell_rejected_by_cu124_build():
    # The exact real-world failure observed on this machine: RTX 5060 (sm_120)
    # with a cu124 build that tops out at sm_90 → must be INCOMPATIBLE.
    cu124 = ["sm_50", "sm_60", "sm_61", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"]
    ok, reason = is_compatible((12, 0), cu124)
    assert not ok
    assert "sm_120" in reason


def test_blackwell_accepted_by_cu128_build():
    cu128 = ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"]
    ok, _ = is_compatible((12, 0), cu128)
    assert ok


def test_same_major_lower_minor_binary_compat():
    # sm_86 kernels run on sm_89 hardware (same major, minor >= build minor).
    ok, _ = is_compatible((8, 9), ["sm_86"])
    assert ok
    # ...but not the other way around.
    ok, _ = is_compatible((8, 0), ["sm_86"])
    assert not ok


def test_ptx_forward_compat():
    ok, reason = is_compatible((12, 0), ["compute_90"])
    assert ok
    assert "PTX" in reason
