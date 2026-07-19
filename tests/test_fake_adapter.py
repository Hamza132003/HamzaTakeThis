"""Fake ASR adapter: determinism and fail-closed language verification."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.asr.adapters.base import UnsupportedLanguageError
from pipeline.asr.adapters.fake import FakeASRAdapter


def _audio():
    rng = np.random.default_rng(42)
    return rng.standard_normal(16000).astype(np.float32)


def test_deterministic_across_calls():
    a, b = FakeASRAdapter(seed=7), FakeASRAdapter(seed=7)
    h1 = a.transcribe(_audio(), "he")
    h2 = b.transcribe(_audio(), "he")
    assert [w.word for w in h1.words] == [w.word for w in h2.words]
    assert [w.raw_model_score for w in h1.words] == [w.raw_model_score for w in h2.words]


def test_seed_and_language_change_output():
    audio = _audio()
    h_he = FakeASRAdapter(seed=0).transcribe(audio, "he")
    h_fa = FakeASRAdapter(seed=0).transcribe(audio, "fa")
    assert h_he.language == "he" and h_fa.language == "fa"
    assert [w.word for w in h_he.words] != [w.word for w in h_fa.words]


def test_fail_closed_on_unverified_language():
    adapter = FakeASRAdapter(languages=frozenset({"he", "fa"}))
    with pytest.raises(UnsupportedLanguageError):
        adapter.transcribe(_audio(), "xx")


def test_scores_are_raw_not_confidence():
    h = FakeASRAdapter().transcribe(_audio(), "en")
    for w in h.words:
        assert hasattr(w, "raw_model_score")
        assert not hasattr(w, "confidence")


def test_words_have_monotonic_timestamps():
    h = FakeASRAdapter().transcribe(_audio(), "fa")
    for prev, cur in zip(h.words, h.words[1:]):
        assert cur.start >= prev.start
        assert cur.end >= cur.start
