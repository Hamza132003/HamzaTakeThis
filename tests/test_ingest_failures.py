"""Structured ingest failure reporting.

Every stage failure must produce a machine-readable IngestFailure (stage,
exception type, sanitized message, timestamp, recoverability, legacy-continued
flag) — never just a log line and `ingest_manifest=None`. The legacy pipeline
must keep working in all of these cases.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")
pytest.importorskip("pydantic")
pytest.importorskip("imageio_ffmpeg")

from pipeline.ingest import ingest as ingest_mod  # noqa: E402
from pipeline.ingest.ingest import ingest_file  # noqa: E402

pytestmark = pytest.mark.heavy


def _fixture(tmp_path, seconds=0.4):
    sr = 16000
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    data = np.stack([0.4 * np.sin(2 * np.pi * 440 * t),
                     0.3 * np.sin(2 * np.pi * 660 * t)], axis=1).astype(np.float32)
    p = tmp_path / "src.wav"
    sf.write(str(p), data, sr, subtype="FLOAT")
    return p


def _run(tmp_path):
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    return ingest_file(_fixture(tmp_path), out,
                       {"storage": {"artifact_dir": str(tmp_path / "store")}})


def _stages(manifest):
    return {f.stage for f in manifest.failures}


def test_hash_failure_is_structured_and_non_recoverable(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest_mod, "stream_hashes",
                        lambda p: (_ for _ in ()).throw(OSError("disk gone")))
    m = _run(tmp_path)
    assert m.failures and _stages(m) == {"hash"}
    f = m.failures[0]
    assert f.exception_type == "OSError"
    assert "disk gone" in f.message
    assert f.recoverable is False and f.legacy_continued is True
    assert f.occurred_utc and f.occurred_utc.endswith("Z")
    assert m.failure_reason and not m.ok


def test_probe_failure_is_recorded_but_ingest_continues(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest_mod, "probe",
                        lambda p: (_ for _ in ()).throw(RuntimeError("probe boom")))
    m = _run(tmp_path)
    assert "probe" in _stages(m)
    assert m.source.media is None
    assert m.derived, "channel extraction should still have run"
    assert "channels" in m.stages_completed


def test_store_failure_recorded_and_no_artifacts_written(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest_mod, "resolve_store_dir",
                        lambda cfg: (_ for _ in ()).throw(PermissionError("no store")))
    m = _run(tmp_path)
    assert "store" in _stages(m)
    assert m.derived == []
    assert m.store_relative_dir is None
    assert not m.ok


def test_channel_extraction_failure_is_structured(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest_mod, "build_channel_assets",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("ffmpeg died")))
    m = _run(tmp_path)
    assert "channels" in _stages(m)
    assert m.condition_vector is None          # diagnostics need the mono asset
    assert m.source.sha256                     # hashing still succeeded
    assert not m.ok


def test_diagnostics_failure_keeps_channel_evidence(tmp_path, monkeypatch):
    import pipeline.diagnostics as diag
    monkeypatch.setattr(diag, "analyse_path",
                        lambda p: (_ for _ in ()).throw(ValueError("bad analysis")))
    m = _run(tmp_path)
    assert "diagnostics" in _stages(m)
    assert m.derived, "channel assets must survive a diagnostics failure"
    assert m.condition_vector is None


def test_manifest_write_failure_is_structured(tmp_path, monkeypatch):
    real = ingest_mod.atomic_write_bytes

    def fail_manifest(dest, data):
        if Path(dest).name == ingest_mod.MANIFEST_NAME:
            raise OSError("read-only store")
        return real(dest, data)

    monkeypatch.setattr(ingest_mod, "atomic_write_bytes", fail_manifest)
    m = _run(tmp_path)
    assert "manifest_write" in _stages(m)
    assert m.derived                            # work completed before the write


def test_failure_messages_are_sanitized_single_line_and_bounded(tmp_path, monkeypatch):
    secretish = "line1\nline2\t" + "x" * 500
    monkeypatch.setattr(ingest_mod, "probe",
                        lambda p: (_ for _ in ()).throw(RuntimeError(secretish)))
    m = _run(tmp_path)
    msg = next(f for f in m.failures if f.stage == "probe").message
    assert "\n" not in msg and "\t" not in msg
    assert len(msg) <= 300


def test_facade_always_returns_structured_status_even_on_failure(tmp_path, monkeypatch):
    from pipeline.audio import extract_audio
    monkeypatch.setattr(ingest_mod, "resolve_store_dir",
                        lambda cfg: (_ for _ in ()).throw(PermissionError("no store")))
    src = _fixture(tmp_path)
    out = extract_audio(src, tmp_path / "o", 16000,
                        cfg={"storage": {"artifact_dir": str(tmp_path / "s")}})
    # Legacy artifacts unaffected...
    assert out["work_wav"].exists() and out["hq_wav"].exists()
    # ...and the failure is machine-readable, not just a log line.
    status = out["ingest_status"]
    assert status["state"] == "degraded" and status["ok"] is False
    assert any(f["stage"] == "store" for f in status["failures"])
    assert status["failure_reason"]


def test_disabled_ingest_reports_disabled_state(tmp_path):
    from pipeline.audio import extract_audio
    src = _fixture(tmp_path)
    out = extract_audio(src, tmp_path / "o2", 16000, cfg={"ingest": {"enabled": False}})
    assert out["ingest_manifest"] is None
    assert out["ingest_status"]["state"] == "disabled"
