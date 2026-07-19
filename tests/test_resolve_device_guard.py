"""resolve_device must fail LOUDLY on a GPU/torch-build mismatch, never silently
degrade (binding constraint 5). Heavy: imports torch via pipeline.utils."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytestmark = pytest.mark.heavy

torch = pytest.importorskip("torch")

import aegis.gpu  # noqa: E402
from pipeline.utils import resolve_device  # noqa: E402


def test_explicit_cpu_always_allowed(monkeypatch):
    monkeypatch.setattr(aegis.gpu, "check_torch_cuda",
                        lambda: {"compatible": False, "reason": "forced mismatch"})
    assert resolve_device("cpu") == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_mismatch_raises_instead_of_silent_fallback(monkeypatch):
    monkeypatch.setattr(aegis.gpu, "check_torch_cuda",
                        lambda: {"compatible": False, "reason": "forced mismatch"})
    with pytest.raises(RuntimeError, match="GPU/torch mismatch"):
        resolve_device("auto")
    with pytest.raises(RuntimeError, match="GPU/torch mismatch"):
        resolve_device("cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_compatible_gpu_selected(monkeypatch):
    monkeypatch.setattr(aegis.gpu, "check_torch_cuda",
                        lambda: {"compatible": True, "reason": "ok"})
    assert resolve_device("auto") == "cuda"
