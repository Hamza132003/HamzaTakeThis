"""Webapp-level unit tests. Marked `heavy`: importing webapp.app pulls the full
runtime stack (flask, torch, librosa, matplotlib). Runs locally; skipped in
light CI (`-m "not heavy"`)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytestmark = pytest.mark.heavy

flask = pytest.importorskip("flask")

from webapp.app import MEDIA_EXT, _truthy  # noqa: E402


def test_truthy_handles_formdata_strings():
    # The multipart path stringifies booleans; bool("false") is True in Python —
    # the exact bug this helper fixed for the mic-recording toggles.
    assert _truthy("false") is False
    assert _truthy("False") is False
    assert _truthy("0") is False
    assert _truthy("") is False
    assert _truthy("no") is False
    assert _truthy("true") is True
    assert _truthy("1") is True
    assert _truthy(True) is True
    assert _truthy(False) is False


def test_media_extensions_cover_observed_formats():
    # .mpeg was a real user file rejected before the fix; keep it covered.
    for ext in (".mp3", ".wav", ".m4a", ".mp4", ".mpeg", ".mpg", ".flac", ".webm"):
        assert ext in MEDIA_EXT


def test_pipeline_imports_without_model_downloads():
    # Importing the full pipeline package must not touch the network or load
    # weights (lazy loading is the repo's contract).
    import pipeline.runner  # noqa: F401
    import pipeline.transcription  # noqa: F401
    import pipeline.translation  # noqa: F401
