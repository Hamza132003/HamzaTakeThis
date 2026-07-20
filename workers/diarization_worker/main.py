"""Diarization worker entry point — runs ONLY in the isolated worker env.

    python -m workers.diarization_worker.main <request.json> <response.json>

Reads a typed request, runs pyannote.audio 4 `speaker-diarization-community-1`
on a PRELOADED waveform, writes a typed response, and always exits 0 when it
managed to write a response (the state field carries success/failure). The
parent distinguishes crash-vs-structured-failure by whether a parseable
response exists.

Token handling: read from the HF_TOKEN environment variable only. Never an
argv item, never echoed, never written into the response.
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from workers.diarization_worker.protocol import (  # noqa: E402
    DiarizationFailure,
    DiarizationRequest,
    DiarizationResponse,
    sanitize_message,
)


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fail(resp: DiarizationResponse, stage: str, exc: BaseException,
          state: str = "failed") -> DiarizationResponse:
    resp.state = state
    resp.genuine_pyannote = False
    resp.failures.append(DiarizationFailure(
        stage=stage, exception_type=type(exc).__name__,
        message=sanitize_message(exc), occurred_utc=_utc()).as_dict())
    return resp


def _classify(exc: BaseException) -> str:
    """Map a raised exception to a specific failure stage — never one generic
    bucket (the adapter contract requires these to be distinguishable)."""
    name = type(exc).__name__
    text = str(exc).lower()
    if "out of memory" in text or name == "OutOfMemoryError":
        return "gpu_out_of_memory"
    if "no kernel image" in text or "cuda capability" in text:
        return "cuda_incompatible"
    if "401" in text or "unauthorized" in text or "authentication" in text:
        return "token_absent"
    if "403" in text or "gated" in text or "awaiting a review" in text \
            or "accept the" in text or "user agreement" in text:
        return "access_denied"
    if "404" in text or "not found" in text or "couldn't connect" in text \
            or "offline" in text or "cannot find" in text or "local_files_only" in text:
        return "model_not_cached"
    return "model_load_failure"


def run(req: DiarizationRequest) -> DiarizationResponse:
    resp = DiarizationResponse(job_id=req.job_id, state="failed",
                               model_id=req.model_id,
                               model_revision=req.model_revision)
    t0 = time.perf_counter()

    # ---- environment facts -------------------------------------------------
    try:
        import pyannote.audio as pa
        import torch
        resp.pyannote_version = pa.__version__
        resp.torch_version = torch.__version__
        resp.cuda_version = torch.version.cuda
    except Exception as e:
        return _fail(resp, "import_failure", e)

    device = req.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    resp.device_used = device

    # ---- audio (preloaded waveform; torchcodec is NOT the file decoder) ----
    try:
        import soundfile as sf
        data, sr = sf.read(req.audio_path, dtype="float32", always_2d=True)
        if data.shape[1] > 1:                       # documented downmix rule
            data = data.mean(axis=1, keepdims=True)
            resp.warnings.append(
                "input had >1 channel; equal-weight mean downmix applied for "
                "diarization only (canonical asset untouched)")
        waveform = torch.from_numpy(data.T.copy())  # (1, samples), read-only src
        duration = data.shape[0] / float(sr)
    except Exception as e:
        return _fail(resp, "audio_unreadable", e)

    # ---- pipeline ----------------------------------------------------------
    try:
        from pyannote.audio import Pipeline
        token = os.environ.get("HF_TOKEN") or None      # env only, never argv
        kwargs = {"token": token} if token else {}      # pyannote 4 API
        if req.model_revision:
            # Pin the exact commit. Without this pyannote resolves the default
            # 'main' ref, which a revision-pinned snapshot_download never
            # populates (no refs/ entry) — so an otherwise fully cached model
            # fails offline with LocalEntryNotFoundError.
            kwargs["revision"] = req.model_revision
        pipeline = Pipeline.from_pretrained(req.model_id, **kwargs)
        if pipeline is None:
            raise RuntimeError(
                "Pipeline.from_pretrained returned None (model not accessible: "
                "gated repo, missing agreement, or not cached)")
        if device == "cuda":
            pipeline.to(torch.device("cuda"))
    except Exception as e:
        return _fail(resp, _classify(e), e)

    # ---- inference ---------------------------------------------------------
    try:
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        call_kwargs: dict = {}
        if req.min_speakers is not None:
            call_kwargs["min_speakers"] = int(req.min_speakers)
        if req.max_speakers is not None:
            call_kwargs["max_speakers"] = int(req.max_speakers)
        output = pipeline({"waveform": waveform, "sample_rate": int(sr)},
                          **call_kwargs)
        if device == "cuda":
            resp.peak_vram_mb = round(
                torch.cuda.max_memory_allocated() / (1024 ** 2), 1)
    except Exception as e:
        return _fail(resp, _classify(e), e)

    # ---- outputs: regular (overlap evidence) + exclusive --------------------
    try:
        regular = getattr(output, "speaker_diarization", output)
        resp.turns = _turns(regular, duration)
        exclusive = getattr(output, "exclusive_speaker_diarization", None)
        if exclusive is not None and req.want_exclusive:
            resp.exclusive_turns = _turns(exclusive, duration)
        elif req.want_exclusive:
            resp.warnings.append(
                "pipeline exposed no exclusive_speaker_diarization output")

        resp.speakers = sorted({t["speaker"] for t in resp.turns})
        resp.num_speakers = len(resp.speakers)
        resp.overlap_regions = _overlaps(resp.turns)
        resp.speech_regions = _speech_regions(resp.turns)
    except Exception as e:
        return _fail(resp, "malformed_response", e)

    resp.processing_sec = round(time.perf_counter() - t0, 3)
    if not resp.turns:
        resp.state = "degraded"
        resp.genuine_pyannote = True       # real model ran; it found nothing
        resp.failures.append(DiarizationFailure(
            stage="no_speakers_detected", exception_type="EmptyResult",
            message="pipeline returned no speaker turns",
            occurred_utc=_utc()).as_dict())
        return resp

    resp.state = "ok"
    resp.genuine_pyannote = True
    resp.fallback_used = False
    return resp


def _turns(annotation, duration: float) -> list[dict]:
    out = []
    for segment, _, label in annotation.itertracks(yield_label=True):
        start = max(0.0, float(segment.start))
        end = min(float(duration), float(segment.end))
        if end > start:
            out.append({"start": round(start, 3), "end": round(end, 3),
                        "speaker": str(label)})
    out.sort(key=lambda t: (t["start"], t["end"], t["speaker"]))
    return out


def _overlaps(turns: list[dict]) -> list[dict]:
    """Regions where 2+ distinct speakers are simultaneously active."""
    events = []
    for t in turns:
        events.append((t["start"], 1, t["speaker"]))
        events.append((t["end"], -1, t["speaker"]))
    events.sort(key=lambda e: (e[0], -e[1]))
    active: set[str] = set()
    out: list[dict] = []
    open_at = None
    for time_, delta, spk in events:
        if delta == 1:
            active.add(spk)
            if len(active) == 2 and open_at is None:
                open_at = time_
        else:
            if len(active) >= 2 and open_at is not None and time_ > open_at:
                out.append({"start": round(open_at, 3), "end": round(time_, 3),
                            "speakers": sorted(active)})
                open_at = None
            active.discard(spk)
            if len(active) >= 2 and open_at is None:
                open_at = time_
    return out


def _speech_regions(turns: list[dict], pad: float = 0.3) -> list[dict]:
    if not turns:
        return []
    ivals = sorted((max(0.0, t["start"] - pad), t["end"] + pad) for t in turns)
    merged = [list(ivals[0])]
    for a, b in ivals[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [{"start": round(a, 3), "end": round(b, 3)} for a, b in merged]


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: main.py <request.json> <response.json>", file=sys.stderr)
        return 2
    req_path, resp_path = Path(argv[1]), Path(argv[2])
    try:
        req = DiarizationRequest.from_json(req_path.read_text(encoding="utf-8"))
    except Exception as e:
        # Cannot trust job_id; still emit a parseable response.
        resp = DiarizationResponse(job_id="unknown", state="failed")
        _fail(resp, "malformed_response", e)
        resp_path.write_text(resp.to_json(), encoding="utf-8")
        return 0
    if req.offline:
        # Hard offline: no network calls may be attempted from the worker.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        resp = run(req)
    except BaseException as e:                       # never crash silently
        resp = _fail(DiarizationResponse(job_id=req.job_id, state="failed"),
                     "unknown", e)
    resp_path.write_text(resp.to_json(), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
