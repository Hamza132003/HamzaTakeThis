"""Typed, versioned request/response contract for the diarization worker.

IMPORTANT: this module is imported by BOTH environments (main venv and the
isolated worker venv), so it must depend on nothing but the standard library.
No pydantic, no numpy, no torch — validation is hand-written and total.

Transport: JSON files (request path and response path are passed as argv), so
nothing sensitive appears in a process listing and no shell is involved.

SECURITY: the Hugging Face token is NEVER carried in this contract. The worker
reads it from the `HF_TOKEN` environment variable that the parent sets on the
child process only. It is never an argv item, never in JSON, never logged.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

PROTOCOL_VERSION = 1

# Every state the adapter must be able to distinguish.
FAILURE_STAGES = (
    "worker_env_missing",       # configured interpreter does not exist
    "import_failure",           # worker env cannot import its packages
    "model_not_cached",         # weights absent and offline/no download allowed
    "token_absent",             # gated model, no token available
    "access_denied",            # token present but agreement/access refused
    "model_load_failure",       # pipeline construction failed
    "cuda_incompatible",        # torch build cannot run on this GPU
    "gpu_out_of_memory",
    "timeout",
    "process_crash",            # non-zero exit / no parseable response
    "malformed_response",
    "no_speakers_detected",
    "cancelled",
    "audio_unreadable",
    "unknown",
)

STATES = ("ok", "degraded", "failed", "cancelled")


class ProtocolError(ValueError):
    """Raised on schema violations (unsupported version, bad types, …)."""


@dataclass
class DiarizationRequest:
    job_id: str
    audio_path: str
    audio_sha256: str
    model_id: str
    schema_version: int = PROTOCOL_VERSION
    model_revision: str | None = None
    device: str = "auto"                 # auto | cuda | cpu
    min_speakers: int | None = None
    max_speakers: int | None = None
    offline: bool = True                 # never reach the network by default
    timeout_sec: float = 900.0
    want_exclusive: bool = True          # also request exclusive diarization

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=1)

    @staticmethod
    def from_json(text: str) -> DiarizationRequest:
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as e:
            raise ProtocolError(f"request is not valid JSON: {e}") from e
        if not isinstance(raw, dict):
            raise ProtocolError("request must be a JSON object")
        _require_version(raw)
        for key in ("job_id", "audio_path", "audio_sha256", "model_id"):
            if not isinstance(raw.get(key), str) or not raw[key]:
                raise ProtocolError(f"request field '{key}' must be a non-empty string")
        for key in ("min_speakers", "max_speakers"):
            if raw.get(key) is not None and not isinstance(raw[key], int):
                raise ProtocolError(f"request field '{key}' must be an int or null")
        if raw.get("device") not in (None, "auto", "cuda", "cpu"):
            raise ProtocolError("request field 'device' must be auto|cuda|cpu")
        known = set(DiarizationRequest.__dataclass_fields__)
        return DiarizationRequest(**{k: v for k, v in raw.items() if k in known})


@dataclass
class DiarizationTurn:
    start: float
    end: float
    speaker: str

    def as_dict(self) -> dict[str, Any]:
        return {"start": round(float(self.start), 3),
                "end": round(float(self.end), 3), "speaker": str(self.speaker)}


@dataclass
class DiarizationFailure:
    stage: str
    exception_type: str
    message: str                          # sanitized, single line, bounded
    occurred_utc: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DiarizationResponse:
    """`genuine_pyannote` is the single field a report may rely on to claim
    real diarization. A fallback response must never set it True."""
    job_id: str
    state: str                              # ok | degraded | failed | cancelled
    schema_version: int = PROTOCOL_VERSION
    genuine_pyannote: bool = False
    fallback_used: bool = False
    turns: list[dict] = field(default_factory=list)            # regular
    exclusive_turns: list[dict] = field(default_factory=list)  # exclusive
    speech_regions: list[dict] = field(default_factory=list)
    num_speakers: int = 0
    speakers: list[str] = field(default_factory=list)
    overlap_regions: list[dict] = field(default_factory=list)
    model_id: str | None = None
    model_revision: str | None = None
    pyannote_version: str | None = None
    torch_version: str | None = None
    cuda_version: str | None = None
    device_used: str | None = None
    processing_sec: float | None = None
    peak_vram_mb: float | None = None
    warnings: list[str] = field(default_factory=list)
    failures: list[dict] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=1)

    @staticmethod
    def from_json(text: str) -> DiarizationResponse:
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as e:
            raise ProtocolError(f"response is not valid JSON: {e}") from e
        if not isinstance(raw, dict):
            raise ProtocolError("response must be a JSON object")
        _require_version(raw)
        if raw.get("state") not in STATES:
            raise ProtocolError(f"response 'state' must be one of {STATES}")
        if not isinstance(raw.get("job_id"), str) or not raw["job_id"]:
            raise ProtocolError("response 'job_id' must be a non-empty string")
        for key in ("turns", "exclusive_turns", "speech_regions",
                    "overlap_regions", "warnings", "failures", "speakers"):
            if key in raw and not isinstance(raw[key], list):
                raise ProtocolError(f"response '{key}' must be a list")
        for t in raw.get("turns", []) + raw.get("exclusive_turns", []):
            if not isinstance(t, dict) or not {"start", "end", "speaker"} <= set(t):
                raise ProtocolError("turn objects need start, end and speaker")
        if raw.get("genuine_pyannote") and raw.get("fallback_used"):
            raise ProtocolError("a response cannot be both genuine and fallback")
        known = set(DiarizationResponse.__dataclass_fields__)
        return DiarizationResponse(**{k: v for k, v in raw.items() if k in known})


def _require_version(raw: dict) -> None:
    v = raw.get("schema_version")
    if not isinstance(v, int):
        raise ProtocolError("schema_version must be an integer")
    if v != PROTOCOL_VERSION:
        raise ProtocolError(
            f"unsupported schema_version {v}; this build speaks {PROTOCOL_VERSION}")


def sanitize_message(exc: BaseException, limit: int = 300) -> str:
    """Single-line, bounded, secret-free message."""
    text = " ".join(str(exc).split())
    for marker in ("hf_", "Bearer "):          # never echo credential material
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx] + "[redacted]"
            break
    return text[:limit]
