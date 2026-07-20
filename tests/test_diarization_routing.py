"""Diarization input-branch routing.

Pure-logic tests (no models) always run. The empirical two-speaker checks are
marked `heavy` and additionally require the cached community-1 weights, a
token and the worker environment — they self-skip when any is absent, so CI
and clean clones stay green.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.diarization import (  # noqa: E402
    DEFAULT_INPUT_BRANCH,
    SUPPORTED_INPUT_BRANCHES,
    resolve_input_branch,
)

FIXTURE = Path(r"C:\aegis-phase0-verification\e2e\two_speaker_fixture.wav")


# ------------------------------------------------------------- routing logic
def test_default_branch_is_raw():
    assert DEFAULT_INPUT_BRANCH == "raw_16k"
    assert set(SUPPORTED_INPUT_BRANCHES) == {"raw_16k", "enhanced_16k"}


def test_resolves_raw_by_default(tmp_path):
    raw, enh = tmp_path / "raw.wav", tmp_path / "enh.wav"
    raw.write_bytes(b"x")
    enh.write_bytes(b"y")
    name, path, err = resolve_input_branch(
        {"raw_16k": raw, "enhanced_16k": enh}, {})
    assert name == "raw_16k" and path == raw and err is None


def test_explicit_enhanced_is_honoured(tmp_path):
    raw, enh = tmp_path / "raw.wav", tmp_path / "enh.wav"
    raw.write_bytes(b"x")
    enh.write_bytes(b"y")
    name, path, err = resolve_input_branch(
        {"raw_16k": raw, "enhanced_16k": enh}, {"input_branch": "enhanced_16k"})
    assert name == "enhanced_16k" and path == enh and err is None


def test_missing_configured_branch_does_not_fall_back(tmp_path):
    """The whole point: an unavailable branch is an error, never a silent
    switch to the other branch."""
    enh = tmp_path / "enh.wav"
    enh.write_bytes(b"y")
    name, path, err = resolve_input_branch(
        {"raw_16k": tmp_path / "absent.wav", "enhanced_16k": enh}, {})
    assert name == "raw_16k"
    assert path is None                      # did NOT substitute enhanced
    assert err and "missing" in err


def test_unsupported_branch_rejected(tmp_path):
    name, path, err = resolve_input_branch(
        {"raw_16k": tmp_path}, {"input_branch": "magic_branch"})
    assert path is None and err and "unsupported" in err


def test_routing_failure_reports_structured_stage(tmp_path):
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    pytest.importorskip("librosa")
    from pipeline import diarization
    real = tmp_path / "enh.wav"                       # real audio, so the
    sf.write(str(real), np.zeros(16000, dtype="float32"), 16000)  # fallback works
    out = diarization.diarize(
        {"raw_16k": tmp_path / "gone.wav", "enhanced_16k": real},
        {"enabled": True}, "cpu", models=None)
    assert out["method"] == "fallback_single_speaker"
    assert out["genuine_pyannote"] is False
    assert out["failure_stage"] == "routing_unavailable"
    assert out["input_branch"] == "raw_16k"
    assert "NOT real diarization" in out["warning"]


def test_fallback_survives_unreadable_audio(tmp_path):
    """A corrupt fallback input must degrade, not raise out of the stage."""
    pytest.importorskip("librosa")
    from pipeline.diarization import _fallback
    bad = tmp_path / "corrupt.wav"
    bad.write_bytes(b"not audio at all")
    out = _fallback(bad, reason="process_crash", failure_stage="process_crash")
    assert out["genuine_pyannote"] is False
    assert out["segments"] == []
    assert "fallback_load_error" in out


def test_routing_stage_is_registered():
    from pipeline.diarization import _STAGE_HELP
    from workers.diarization_worker.protocol import FAILURE_STAGES
    assert "routing_unavailable" in FAILURE_STAGES
    assert "routing_unavailable" in _STAGE_HELP


def test_config_default_is_raw_16k():
    import yaml
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["diarization"]["input_branch"] == "raw_16k"


# ------------------------------------------------- empirical branch behaviour
def _worker_ready() -> bool:
    from workers.diarization_worker.client import resolve_worker_python
    if not resolve_worker_python(None).exists():
        return False
    if not (ROOT / "hf_token.txt").exists():
        return False
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    return (cache / "models--pyannote--speaker-diarization-community-1").exists()


needs_model = pytest.mark.skipif(
    not FIXTURE.exists() or not _worker_ready(),
    reason="needs the two-speaker fixture, worker env, token and cached weights")


@pytest.mark.heavy
@needs_model
def test_raw_branch_finds_two_speakers_enhanced_finds_one(tmp_path):
    """The measurement that motivated the routing change. Uses the real model,
    offline, on both branches produced by one pipeline run."""
    from workers.diarization_worker.client import run_diarization
    token = (ROOT / "hf_token.txt").read_text(encoding="utf-8").strip()
    outdir = tmp_path / "run"
    r = subprocess.run(
        [sys.executable, "main.py", str(FIXTURE), "--outdir", str(outdir)],
        cwd=str(ROOT), capture_output=True, text=True, timeout=3600)
    assert r.returncode == 0, r.stdout[-800:]
    stem = FIXTURE.stem
    raw = outdir / stem / "audio_16k_mono.wav"
    enh = outdir / stem / "voice_16k.wav"
    assert raw.exists() and enh.exists()

    cfg = {"diarization": {
        "model": "pyannote/speaker-diarization-community-1",
        "model_revision": "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee",
        "offline": True, "device": "auto", "timeout_sec": 1800,
        "want_exclusive": True}}
    raw_r = run_diarization(raw, "a" * 64, cfg, token=token)
    enh_r = run_diarization(enh, "a" * 64, cfg, token=token)

    assert raw_r.genuine_pyannote and enh_r.genuine_pyannote
    assert raw_r.num_speakers >= 2, "raw branch must resolve both speakers"
    assert raw_r.overlap_regions, "raw branch must preserve overlap evidence"
    # Documents the regression that motivated the default; if enhancement ever
    # stops destroying separability this will fail and should be revisited.
    assert enh_r.num_speakers < raw_r.num_speakers, (
        "enhanced branch no longer degrades speaker separability - re-evaluate "
        "the routing default and update docs/DIARIZATION_RUNTIME_ARCHITECTURE.md")


@pytest.mark.heavy
@needs_model
def test_default_pipeline_routes_to_raw_and_reports_it(tmp_path):
    outdir = tmp_path / "run2"
    r = subprocess.run(
        [sys.executable, "main.py", str(FIXTURE), "--outdir", str(outdir)],
        cwd=str(ROOT), capture_output=True, text=True, timeout=3600)
    assert r.returncode == 0, r.stdout[-800:]
    rep = json.loads((outdir / FIXTURE.stem / "report.json").read_text(encoding="utf-8"))
    ds = rep["diarization_status"]
    assert ds["input_branch"] == "raw_16k"
    assert ds["input_file"] == "audio_16k_mono.wav"
    assert ds["input_sha256"] and len(ds["input_sha256"]) == 64
    assert ds["genuine_pyannote"] is True
    assert rep["diarization_method"] == "pyannote4_isolated"
    assert rep["num_speakers"] >= 2
    assert ds["overlap_count"] >= 1
    assert ds["model_revision"] == "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee"
    # the markdown report must state the branch too
    md = (outdir / FIXTURE.stem / "report.md").read_text(encoding="utf-8")
    assert "raw_16k" in md and "audio_16k_mono.wav" in md
