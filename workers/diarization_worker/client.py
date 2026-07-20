"""Main-process client for the isolated diarization worker.

Launches the worker with ITS OWN interpreter via an argument list (never
`shell=True`, never a token on the command line), exchanges typed JSON through
temp files, and maps every outcome to a specific failure stage.

Runs in the main environment: imports only the standard library and
`protocol.py`. pyannote is never imported here.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from workers.diarization_worker.protocol import (
    DiarizationFailure,
    DiarizationRequest,
    DiarizationResponse,
    ProtocolError,
    sanitize_message,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def default_worker_python() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / ".local" / "share")
    return Path(base) / "AegisXPrime" / "envs" / "diarization" / "Scripts" / "python.exe"


def resolve_worker_python(cfg: dict | None) -> Path:
    configured = ((cfg or {}).get("diarization") or {}).get("worker_python")
    return Path(configured) if configured else default_worker_python()


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _failed(job_id: str, stage: str, exc_type: str, message: str,
            state: str = "failed") -> DiarizationResponse:
    r = DiarizationResponse(job_id=job_id, state=state, genuine_pyannote=False,
                            fallback_used=False)
    r.failures.append(DiarizationFailure(stage=stage, exception_type=exc_type,
                                         message=message,
                                         occurred_utc=_utc()).as_dict())
    return r


def run_diarization(audio_path: Path, audio_sha256: str, cfg: dict | None,
                    token: str | None = None,
                    should_cancel=None) -> DiarizationResponse:
    """Invoke the worker. Never raises; always returns a typed response whose
    `genuine_pyannote` flag is the only basis for claiming real diarization."""
    dcfg = (cfg or {}).get("diarization") or {}
    job_id = uuid.uuid4().hex[:16]

    worker_py = resolve_worker_python(cfg)
    if not worker_py.exists():
        return _failed(job_id, "worker_env_missing", "FileNotFoundError",
                       f"diarization worker interpreter not found at "
                       f"{worker_py.name}; run ./setup_diarization.ps1")

    audio_path = Path(audio_path)
    if not audio_path.is_file():
        return _failed(job_id, "audio_unreadable", "FileNotFoundError",
                       "diarization input audio does not exist")

    req = DiarizationRequest(
        job_id=job_id, audio_path=str(audio_path), audio_sha256=audio_sha256,
        model_id=str(dcfg.get("model", "pyannote/speaker-diarization-community-1")),
        model_revision=dcfg.get("model_revision"),
        device=str(dcfg.get("device", "auto")),
        min_speakers=dcfg.get("min_speakers"),
        max_speakers=dcfg.get("max_speakers"),
        offline=bool(dcfg.get("offline", True)),
        timeout_sec=float(dcfg.get("timeout_sec", 900.0)),
        want_exclusive=bool(dcfg.get("want_exclusive", True)))

    tmpdir = Path(tempfile.mkdtemp(prefix="aegis_diar_"))
    req_path, resp_path = tmpdir / "request.json", tmpdir / "response.json"
    proc = None
    try:
        req_path.write_text(req.to_json(), encoding="utf-8")

        # Token travels ONLY in the child's environment: not argv (visible in
        # process listings), not JSON, never logged.
        env = dict(os.environ)
        env.pop("HF_TOKEN", None)
        env.pop("HUGGINGFACE_TOKEN", None)
        if token:
            env["HF_TOKEN"] = token
        env["PYTHONIOENCODING"] = "utf-8"

        cmd = [str(worker_py), "-m", "workers.diarization_worker.main",
               str(req_path), str(resp_path)]        # argument list, no shell
        proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, encoding="utf-8", errors="replace",
                                shell=False)

        # Cooperative cancellation + timeout without blocking forever.
        deadline = req.timeout_sec
        waited = 0.0
        step = 0.5
        while True:
            try:
                out, err = proc.communicate(timeout=step)
                break
            except subprocess.TimeoutExpired:
                waited += step
                if should_cancel is not None and should_cancel():
                    proc.kill()
                    proc.communicate()
                    return _failed(job_id, "cancelled", "JobCancelled",
                                   "diarization cancelled by operator",
                                   state="cancelled")
                if waited >= deadline:
                    proc.kill()
                    proc.communicate()
                    return _failed(job_id, "timeout", "TimeoutExpired",
                                   f"worker exceeded {deadline:.0f}s")

        if not resp_path.exists():
            tail = " ".join((err or out or "").split())[-300:]
            return _failed(job_id, "process_crash", "WorkerCrash",
                           f"worker exit {proc.returncode} without a response: "
                           f"{tail}" if tail else
                           f"worker exit {proc.returncode} without a response")
        try:
            resp = DiarizationResponse.from_json(
                resp_path.read_text(encoding="utf-8"))
        except ProtocolError as e:
            return _failed(job_id, "malformed_response", "ProtocolError",
                           sanitize_message(e))
        if proc.returncode != 0 and resp.state == "ok":
            # Contradictory: trust the exit code, not the payload.
            return _failed(job_id, "process_crash", "WorkerCrash",
                           f"worker reported ok but exited {proc.returncode}")
        return resp
    except Exception as e:
        return _failed(job_id, "unknown", type(e).__name__, sanitize_message(e))
    finally:
        if proc is not None and proc.poll() is None:      # no orphans
            try:
                proc.kill()
                proc.communicate(timeout=10)
            except Exception:
                pass
        for p in (req_path, resp_path):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            tmpdir.rmdir()
        except OSError:
            pass


def worker_selftest(cfg: dict | None, timeout: float = 120.0) -> dict:
    """Run the worker environment's self-test (imports + GPU, no weights)."""
    import json
    worker_py = resolve_worker_python(cfg)
    if not worker_py.exists():
        return {"ok": False, "present": False,
                "errors": [f"worker interpreter not found: {worker_py}"]}
    script = REPO_ROOT / "workers" / "diarization_worker" / "selftest.py"
    try:
        r = subprocess.run([str(worker_py), str(script)], cwd=str(REPO_ROOT),
                           capture_output=True, text=True, timeout=timeout,
                           shell=False, encoding="utf-8", errors="replace")
        out = (r.stdout or "").strip()
        # The self-test prints pretty-printed (multi-line) JSON, so parse from
        # the first '{' rather than assuming a single line.
        start = out.find("{")
        if start == -1:
            return {"ok": False, "present": True,
                    "errors": [f"worker self-test produced no JSON "
                               f"(exit {r.returncode}): "
                               f"{' '.join((r.stderr or out).split())[-200:]}"]}
        data = json.loads(out[start:])
        data["present"] = True
        data.setdefault("ok", r.returncode == 0)
        return data
    except Exception as e:
        return {"ok": False, "present": True,
                "errors": [f"{type(e).__name__}: {sanitize_message(e)}"]}


if __name__ == "__main__":   # tiny manual probe
    import json
    print(json.dumps(worker_selftest(None), indent=1))
    sys.exit(0)
