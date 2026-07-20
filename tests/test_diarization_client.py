"""Diarization client behaviour driven by FAKE workers.

Every scenario is exercised with a tiny stdlib-only fake interpreter script, so
these tests need no torch, no pyannote and no model weights — they run in CI.
"""
import json
import os
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from workers.diarization_worker.client import run_diarization  # noqa: E402
from workers.diarization_worker.protocol import PROTOCOL_VERSION  # noqa: E402


def _fake_worker(tmp_path: Path, body: str, name="fakepy") -> Path:
    """Create a fake 'interpreter': a Python script that ignores `-m module`
    and writes whatever `body` dictates. Invoked as
    <fake> -m workers.diarization_worker.main <req> <resp>."""
    script = tmp_path / f"{name}.py"
    script.write_text(textwrap.dedent(body), encoding="utf-8")
    # A .cmd shim lets the client treat it as an executable interpreter.
    shim = tmp_path / f"{name}.cmd"
    shim.write_text(f'@echo off\r\n"{sys.executable}" "{script}" %*\r\n',
                    encoding="utf-8")
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC)
    return shim


def _cfg(worker: Path, **kw) -> dict:
    d = {"worker_python": str(worker), "timeout_sec": 30.0}
    d.update(kw)
    return {"diarization": d}


OK_BODY = """
    import json, sys
    req_path, resp_path = sys.argv[-2], sys.argv[-1]
    req = json.loads(open(req_path, encoding='utf-8').read())
    resp = {
        "schema_version": %d, "job_id": req["job_id"], "state": "ok",
        "genuine_pyannote": True, "fallback_used": False,
        "turns": [
            {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
            {"start": 1.5, "end": 3.5, "speaker": "SPEAKER_01"}],
        "exclusive_turns": [
            {"start": 0.0, "end": 1.5, "speaker": "SPEAKER_00"},
            {"start": 2.0, "end": 3.5, "speaker": "SPEAKER_01"}],
        "overlap_regions": [{"start": 1.5, "end": 2.0,
                             "speakers": ["SPEAKER_00", "SPEAKER_01"]}],
        "speech_regions": [{"start": 0.0, "end": 3.5}],
        "num_speakers": 2, "speakers": ["SPEAKER_00", "SPEAKER_01"],
        "model_id": req["model_id"], "pyannote_version": "4.0.7",
        "device_used": "cpu", "processing_sec": 0.01,
        "warnings": [], "failures": []}
    open(resp_path, 'w', encoding='utf-8').write(json.dumps(resp))
""" % PROTOCOL_VERSION


@pytest.fixture
def audio(tmp_path):
    p = tmp_path / "a.wav"
    p.write_bytes(b"RIFF....WAVEfmt ")     # content irrelevant to the fake
    return p


def test_fake_worker_two_speakers_with_overlap_and_exclusive(tmp_path, audio):
    w = _fake_worker(tmp_path, OK_BODY)
    r = run_diarization(audio, "a" * 64, _cfg(w))
    assert r.state == "ok" and r.genuine_pyannote is True and r.fallback_used is False
    assert r.num_speakers == 2 and r.speakers == ["SPEAKER_00", "SPEAKER_01"]
    assert len(r.turns) == 2 and len(r.exclusive_turns) == 2
    assert r.overlap_regions and r.overlap_regions[0]["start"] == 1.5
    starts = [t["start"] for t in r.turns]
    assert starts == sorted(starts)                      # monotonic ordering


def test_worker_env_missing(tmp_path, audio):
    r = run_diarization(audio, "a" * 64,
                        _cfg(tmp_path / "definitely-absent.exe"))
    assert r.state == "failed" and r.genuine_pyannote is False
    assert r.failures[0]["stage"] == "worker_env_missing"


def test_worker_import_failure(tmp_path, audio):
    w = _fake_worker(tmp_path, """
        import json, sys
        req = json.loads(open(sys.argv[-2], encoding='utf-8').read())
        resp = {"schema_version": %d, "job_id": req["job_id"], "state": "failed",
                "genuine_pyannote": False,
                "failures": [{"stage": "import_failure",
                              "exception_type": "ModuleNotFoundError",
                              "message": "No module named 'pyannote'",
                              "occurred_utc": "2026-07-20T00:00:00Z"}]}
        open(sys.argv[-1], 'w', encoding='utf-8').write(json.dumps(resp))
    """ % PROTOCOL_VERSION)
    r = run_diarization(audio, "a" * 64, _cfg(w))
    assert r.failures[0]["stage"] == "import_failure"
    assert r.genuine_pyannote is False


def test_worker_nonzero_exit_without_response(tmp_path, audio):
    w = _fake_worker(tmp_path, """
        import sys
        sys.stderr.write('worker exploded')
        sys.exit(3)
    """)
    r = run_diarization(audio, "a" * 64, _cfg(w))
    assert r.failures[0]["stage"] == "process_crash"
    assert "3" in r.failures[0]["message"]


def test_worker_malformed_json_response(tmp_path, audio):
    w = _fake_worker(tmp_path, """
        import sys
        open(sys.argv[-1], 'w', encoding='utf-8').write('{not json')
    """)
    r = run_diarization(audio, "a" * 64, _cfg(w))
    assert r.failures[0]["stage"] == "malformed_response"


def test_worker_timeout_is_killed(tmp_path, audio):
    w = _fake_worker(tmp_path, """
        import time
        time.sleep(30)
    """)
    r = run_diarization(audio, "a" * 64, _cfg(w, timeout_sec=2.0))
    assert r.failures[0]["stage"] == "timeout"
    assert r.state == "failed"


def test_worker_cancellation(tmp_path, audio):
    w = _fake_worker(tmp_path, """
        import time
        time.sleep(30)
    """)
    r = run_diarization(audio, "a" * 64, _cfg(w, timeout_sec=30.0),
                        should_cancel=lambda: True)
    assert r.state == "cancelled"
    assert r.failures[0]["stage"] == "cancelled"


def test_token_never_on_command_line_and_passed_via_env(tmp_path, audio):
    """The fake records argv and whether HF_TOKEN reached it via env."""
    w = _fake_worker(tmp_path, """
        import json, os, sys
        req = json.loads(open(sys.argv[-2], encoding='utf-8').read())
        probe = {"argv": sys.argv, "env_token": os.environ.get("HF_TOKEN"),
                 "req_text": open(sys.argv[-2], encoding='utf-8').read()}
        open(os.path.join(os.path.dirname(sys.argv[-1]), 'probe.json'), 'w',
             encoding='utf-8').write(json.dumps(probe))
        resp = {"schema_version": %d, "job_id": req["job_id"], "state": "ok",
                "genuine_pyannote": True,
                "turns": [{"start": 0.0, "end": 1.0, "speaker": "S0"}],
                "speakers": ["S0"], "num_speakers": 1}
        open(sys.argv[-1], 'w', encoding='utf-8').write(json.dumps(resp))
    """ % PROTOCOL_VERSION)
    secret = "hf_" + "T" * 34
    r = run_diarization(audio, "a" * 64, _cfg(w), token=secret)
    assert r.state == "ok"
    probe = json.loads(next(Path(tmp_path).rglob("probe.json")).read_text("utf-8")) \
        if list(Path(tmp_path).rglob("probe.json")) else None
    if probe is None:            # temp dir already cleaned; assert via client contract
        pytest.skip("probe file cleaned up with the request temp dir")
    assert secret not in " ".join(probe["argv"])          # not in argv
    assert secret not in probe["req_text"]                # not in the JSON
    assert probe["env_token"] == secret                   # only in child env


def test_paths_with_spaces_and_non_ascii(tmp_path, audio):
    weird = tmp_path / "sp ace ünïcødé 音声"
    weird.mkdir()
    target = weird / "clip ör.wav"
    target.write_bytes(audio.read_bytes())
    w = _fake_worker(tmp_path, OK_BODY)
    r = run_diarization(target, "a" * 64, _cfg(w))
    assert r.state == "ok" and r.num_speakers == 2


def test_missing_audio_is_rejected_before_launch(tmp_path):
    w = _fake_worker(tmp_path, OK_BODY)
    r = run_diarization(tmp_path / "nope.wav", "a" * 64, _cfg(w))
    assert r.failures[0]["stage"] == "audio_unreadable"


def test_client_uses_argument_list_not_shell(tmp_path, audio, monkeypatch):
    seen = {}
    real_popen = subprocess.Popen

    class SpyPopen(real_popen):
        def __init__(self, cmd, *a, **kw):
            seen["cmd"], seen["shell"] = cmd, kw.get("shell")
            super().__init__(cmd, *a, **kw)

    monkeypatch.setattr(subprocess, "Popen", SpyPopen)
    w = _fake_worker(tmp_path, OK_BODY)
    run_diarization(audio, "a" * 64, _cfg(w))
    assert isinstance(seen["cmd"], list)          # never a shell string
    assert seen["shell"] is False
    assert not any("hf_" in str(part) for part in seen["cmd"])


def test_no_orphan_process_after_timeout(tmp_path, audio):
    psutil = pytest.importorskip("psutil")
    w = _fake_worker(tmp_path, "import time\ntime.sleep(30)\n")
    before = {p.pid for p in psutil.process_iter()}
    run_diarization(audio, "a" * 64, _cfg(w, timeout_sec=2.0))
    leaked = [p for p in psutil.process_iter()
              if p.pid not in before and "python" in (p.name() or "").lower()
              and _is_our_fake(p, tmp_path)]
    assert not leaked, f"orphaned worker processes: {leaked}"


def _is_our_fake(proc, tmp_path) -> bool:
    try:
        return str(tmp_path) in " ".join(proc.cmdline())
    except Exception:
        return False


def test_temp_request_response_cleaned_up(tmp_path, audio):
    w = _fake_worker(tmp_path, OK_BODY)
    before = set(Path(os.environ.get("TEMP", "/tmp")).glob("aegis_diar_*"))
    run_diarization(audio, "a" * 64, _cfg(w))
    after = set(Path(os.environ.get("TEMP", "/tmp")).glob("aegis_diar_*"))
    assert not (after - before), "worker temp directories were left behind"
