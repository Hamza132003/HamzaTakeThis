"""Deterministic fake ASR adapter for tests and CI (no models, no network).

Output is a pure function of (audio bytes/shape, language, branch_id, seed), so
tests and CI produce identical hypotheses on every run and platform.
"""
from __future__ import annotations

import hashlib

import numpy as np

from .base import AdapterCapabilities, ASRAdapter, ASRHypothesisDraft, ASRWord

_WORDBANK = {
    "he": ["שלום", "מבחן", "קול", "רעש", "דיבור", "מערכת"],
    "fa": ["سلام", "آزمایش", "صدا", "نویز", "گفتار", "سامانه"],
    "en": ["hello", "test", "voice", "noise", "speech", "system"],
    "ar": ["مرحبا", "اختبار", "صوت", "ضجيج", "كلام", "نظام"],
}


class FakeASRAdapter(ASRAdapter):
    def __init__(self, languages: frozenset[str] = frozenset(_WORDBANK),
                 seed: int = 0, family: str = "fake"):
        self._languages = languages
        self._seed = seed
        self._family = family

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            family=self._family, model_id="fake/deterministic-asr",
            revision="0000000000000000000000000000000000000000",
            languages=self._languages, needs_gpu=False, license_id="none",
            notes="deterministic test double; produces no real recognition")

    def transcribe(self, audio, language: str, branch_id: str = "original_mono",
                   **decode_opts) -> ASRHypothesisDraft:
        self.verify_language(language)
        arr = np.asarray(audio, dtype=np.float32)
        digest = hashlib.sha256(
            arr.tobytes() + f"|{language}|{branch_id}|{self._seed}".encode()
        ).digest()
        n_words = 1 + digest[0] % 5
        bank = _WORDBANK.get(language, _WORDBANK["en"])
        dur = max(len(arr) / 16000.0, 0.5)
        words = []
        for i in range(n_words):
            b = digest[1 + i]
            start = round(dur * i / n_words, 2)
            end = round(dur * (i + 1) / n_words, 2)
            words.append(ASRWord(word=bank[b % len(bank)], start=start, end=end,
                                 raw_model_score=round(0.35 + (b % 60) / 100, 2)))
        caps = self.capabilities()
        return ASRHypothesisDraft(
            family=caps.family, model_id=caps.model_id, revision=caps.revision,
            branch_id=branch_id, language=language, words=words,
            decoder_metadata={"seed": self._seed, "fake": True})
