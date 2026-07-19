"""ASR adapter interface (Phase 0 skeleton).

Every recognizer family plugs in through this interface. Phase 0 ships the
interface plus a deterministic fake for tests; real adapters (Whisper wrap,
Omnilingual CTC) arrive in Phase 4 behind license/hardware gates. Hypothesis
dicts are placeholders until the typed contracts land in Phase 1.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


class UnsupportedLanguageError(Exception):
    """Raised (fail-closed) when an adapter is asked for a language it cannot
    verifiably support. Never downgrade this to a warning: unsupported coverage
    must stop the request, not produce pretend output (spec §10.2)."""


@dataclass(frozen=True)
class AdapterCapabilities:
    family: str                    # e.g. "whisper", "ctc", "fake"
    model_id: str
    revision: str                  # exact pin; "UNPINNED" is not valid for real adapters
    languages: frozenset[str]      # ISO codes the adapter has VERIFIED it supports
    needs_gpu: bool = False
    license_id: str = ""           # key into MODEL_LICENSES.yaml
    notes: str = ""


@dataclass
class ASRWord:
    word: str
    start: float
    end: float
    raw_model_score: float         # NEVER called "confidence" before calibration


@dataclass
class ASRHypothesisDraft:
    """Placeholder for the Phase-1 ASRHypothesis contract."""
    family: str
    model_id: str
    revision: str
    branch_id: str
    language: str
    words: list[ASRWord] = field(default_factory=list)
    decoder_metadata: dict = field(default_factory=dict)
    failure_reason: str | None = None


class ASRAdapter(ABC):
    @abstractmethod
    def capabilities(self) -> AdapterCapabilities: ...

    def verify_language(self, language: str) -> None:
        """Fail closed on unverified language support."""
        caps = self.capabilities()
        if language not in caps.languages:
            raise UnsupportedLanguageError(
                f"{caps.family}:{caps.model_id} has not verified support for "
                f"'{language}' (verified: {sorted(caps.languages)})")

    @abstractmethod
    def transcribe(self, audio, language: str, branch_id: str = "original_mono",
                   **decode_opts) -> ASRHypothesisDraft:
        """Return a hypothesis draft. Must call verify_language first and record
        failures in failure_reason rather than raising mid-batch (except the
        fail-closed language error, which must raise)."""
