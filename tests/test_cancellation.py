"""Cooperative-cancellation contract in pipeline.utils (no heavy imports)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.utils import JobCancelled, check_cancel, set_cancel_check


@pytest.fixture(autouse=True)
def _clean_hook():
    yield
    set_cancel_check(None)


def test_no_hook_never_raises():
    set_cancel_check(None)
    check_cancel()


def test_hook_false_does_not_raise():
    set_cancel_check(lambda: False)
    check_cancel()


def test_hook_true_raises_jobcancelled():
    set_cancel_check(lambda: True)
    with pytest.raises(JobCancelled):
        check_cancel()


def test_jobcancelled_is_exception_subclass():
    # Broad `except Exception` handlers CAN swallow it — which is why every
    # stage must re-raise explicitly. This documents the hazard.
    assert issubclass(JobCancelled, Exception)
