"""Diarization worker protocol: schema round-trip, rejection of invalid /
unsupported / malformed payloads, and token redaction.

Stdlib-only contract, so these run in light CI with no torch and no models.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from workers.diarization_worker.protocol import (  # noqa: E402
    PROTOCOL_VERSION,
    DiarizationRequest,
    DiarizationResponse,
    ProtocolError,
    sanitize_message,
)


def _req(**kw) -> DiarizationRequest:
    base = dict(job_id="job1", audio_path=r"C:\tmp\a.wav", audio_sha256="a" * 64,
                model_id="pyannote/speaker-diarization-community-1")
    base.update(kw)
    return DiarizationRequest(**base)


def test_request_round_trip():
    r = _req(min_speakers=1, max_speakers=4, device="cuda")
    back = DiarizationRequest.from_json(r.to_json())
    assert back == r
    assert back.schema_version == PROTOCOL_VERSION


def test_response_round_trip():
    r = DiarizationResponse(
        job_id="job1", state="ok", genuine_pyannote=True,
        turns=[{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}],
        speakers=["SPEAKER_00"], num_speakers=1)
    back = DiarizationResponse.from_json(r.to_json())
    assert back == r


@pytest.mark.parametrize("bad", [
    "{not json",
    "[]",
    json.dumps({"schema_version": PROTOCOL_VERSION}),                # missing fields
    json.dumps({"schema_version": PROTOCOL_VERSION, "job_id": "",
                "audio_path": "a", "audio_sha256": "b", "model_id": "m"}),
    json.dumps({"schema_version": PROTOCOL_VERSION, "job_id": "j",
                "audio_path": "a", "audio_sha256": "b", "model_id": "m",
                "device": "tpu"}),
    json.dumps({"schema_version": PROTOCOL_VERSION, "job_id": "j",
                "audio_path": "a", "audio_sha256": "b", "model_id": "m",
                "min_speakers": "two"}),
])
def test_invalid_requests_rejected(bad):
    with pytest.raises(ProtocolError):
        DiarizationRequest.from_json(bad)


def test_unsupported_version_rejected():
    payload = json.dumps({"schema_version": PROTOCOL_VERSION + 1, "job_id": "j",
                          "audio_path": "a", "audio_sha256": "b", "model_id": "m"})
    with pytest.raises(ProtocolError, match="unsupported schema_version"):
        DiarizationRequest.from_json(payload)
    with pytest.raises(ProtocolError):
        DiarizationResponse.from_json(json.dumps(
            {"schema_version": 99, "job_id": "j", "state": "ok"}))


def test_missing_version_rejected():
    with pytest.raises(ProtocolError, match="schema_version"):
        DiarizationRequest.from_json(json.dumps({"job_id": "j"}))


@pytest.mark.parametrize("bad", [
    json.dumps({"schema_version": PROTOCOL_VERSION, "job_id": "j", "state": "great"}),
    json.dumps({"schema_version": PROTOCOL_VERSION, "job_id": "", "state": "ok"}),
    json.dumps({"schema_version": PROTOCOL_VERSION, "job_id": "j", "state": "ok",
                "turns": "not-a-list"}),
    json.dumps({"schema_version": PROTOCOL_VERSION, "job_id": "j", "state": "ok",
                "turns": [{"start": 0.0, "end": 1.0}]}),          # no speaker
])
def test_malformed_responses_rejected(bad):
    with pytest.raises(ProtocolError):
        DiarizationResponse.from_json(bad)


def test_response_cannot_be_genuine_and_fallback():
    payload = json.dumps({"schema_version": PROTOCOL_VERSION, "job_id": "j",
                          "state": "ok", "genuine_pyannote": True,
                          "fallback_used": True})
    with pytest.raises(ProtocolError, match="cannot be both"):
        DiarizationResponse.from_json(payload)


def test_request_contract_carries_no_token_field():
    fields = set(DiarizationRequest.__dataclass_fields__)
    assert not any("token" in f.lower() for f in fields)
    assert "hf_token" not in _req().to_json().lower()


def test_sanitize_message_redacts_credentials_and_bounds_length():
    msg = sanitize_message(RuntimeError("auth failed for hf_" + "A" * 34))
    assert "hf_A" not in msg and "[redacted]" in msg
    assert sanitize_message(RuntimeError("Bearer " + "x" * 50)).find("Bearer x") == -1
    long = sanitize_message(RuntimeError("line1\nline2\t" + "y" * 500))
    assert "\n" not in long and "\t" not in long and len(long) <= 300
