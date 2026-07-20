"""Environment isolation invariants and fallback honesty.

These encode the reason the worker exists: the two environments have
contradictory numpy requirements, and a fallback must never be presentable as
real diarization.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MANIFEST = ROOT / "workers" / "diarization_worker" / "environment_manifest.json"


# ----------------------------------------------------- dependency-conflict facts
def test_main_requirements_do_not_install_pyannote():
    """A pyannote entry in the main requirements would look like a working
    backend while never being used (and would drag numpy 2 in)."""
    text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    active = [ln.strip() for ln in text.splitlines()
              if ln.strip() and not ln.strip().startswith("#")]
    assert not any("pyannote" in ln.lower() for ln in active), \
        "pyannote must not be an installable requirement of the main environment"
    assert "pyannote" in text.lower(), "the exclusion should be documented in-place"


def test_worker_requirements_demand_numpy2_and_pyannote4():
    text = (ROOT / "requirements-diarization.txt").read_text(encoding="utf-8")
    active = [ln.split("#")[0].strip() for ln in text.splitlines()
              if ln.strip() and not ln.strip().startswith("#")]
    assert "pyannote.audio==4.0.7" in active
    numpy_pins = [ln for ln in active if ln.startswith("numpy==")]
    assert numpy_pins, "worker must pin numpy explicitly"
    major = int(numpy_pins[0].split("==")[1].split(".")[0])
    assert major >= 2, "worker requires numpy>=2 (pyannote-core 6)"
    # exact pins only - a floating range would let the conflict drift back
    assert all("==" in ln for ln in active), f"non-exact pins found: {active}"


def test_conflict_is_documented_not_silently_resolved():
    doc = (ROOT / "docs" / "DIARIZATION_RUNTIME_ARCHITECTURE.md").read_text(encoding="utf-8")
    for needle in ("numpy>=1.24.3,<2.0", "numpy>=2", "clearvoice", "--no-deps",
                   "monkeypatch"):
        assert needle.lower() in doc.lower(), f"conflict doc missing '{needle}'"


@pytest.mark.heavy
def test_main_environment_keeps_numpy1_and_clearvoice():
    import numpy
    assert int(numpy.__version__.split(".")[0]) < 2, \
        "main environment must stay on numpy<2 for ClearVoice"
    import importlib.util
    assert importlib.util.find_spec("clearvoice") is not None


@pytest.mark.heavy
def test_worker_environment_manifest_requires_numpy2():
    if not MANIFEST.exists():
        pytest.skip("worker environment manifest not generated on this machine")
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert m["constraints"]["numpy"].startswith(">=2")
    assert m["constraints"]["main_env_numpy"].startswith("<2")
    assert int(m["packages"]["numpy"].split(".")[0]) >= 2
    assert m["packages"]["pyannote.audio"].startswith("4.")
    assert m["model"]["weights_downloaded"] is False, \
        "manifest must not claim weights are present until they are"


# ------------------------------------------------------------- fallback honesty
def test_fallback_never_claims_pyannote_success(tmp_path):
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    from pipeline.diarization import _fallback
    wav = tmp_path / "a.wav"
    sf.write(str(wav), np.zeros(16000, dtype="float32"), 16000)
    out = _fallback(wav, reason="worker_env_missing",
                    failure_stage="worker_env_missing", warning="degraded")
    assert out["method"] == "fallback_single_speaker"
    assert out["genuine_pyannote"] is False
    assert out["fallback_used"] is True
    assert out["failure_stage"] == "worker_env_missing"
    assert "pyannote" not in out["method"]
    assert out["num_speakers"] == 1


def test_every_failure_stage_has_distinct_help_text():
    from pipeline.diarization import _STAGE_HELP
    from workers.diarization_worker.protocol import FAILURE_STAGES
    for stage in FAILURE_STAGES:
        assert stage in _STAGE_HELP, f"no operator-facing explanation for '{stage}'"
    # distinct messages: failures must not collapse into one generic string
    assert len(set(_STAGE_HELP.values())) == len(_STAGE_HELP)


def test_diarize_reports_worker_env_missing_without_importing_pyannote(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    from pipeline import diarization
    wav = tmp_path / "a.wav"
    sf.write(str(wav), np.zeros(16000, dtype="float32"), 16000)
    cfg = {"enabled": True, "worker_python": str(tmp_path / "absent.exe")}
    out = diarization.diarize(wav, cfg, "cpu", models=None)
    assert out["method"] == "fallback_single_speaker"
    assert out["genuine_pyannote"] is False
    assert out["failure_stage"] == "worker_env_missing"
    assert "warning" in out and "NOT real diarization" in out["warning"]
    # the main process must never have imported pyannote
    assert "pyannote.audio" not in sys.modules


def test_pipeline_module_does_not_import_pyannote():
    src = (ROOT / "pipeline" / "diarization.py").read_text(encoding="utf-8")
    assert "import pyannote" not in src
    assert "from pyannote" not in src


# ------------------------------------------------------------------- doctor
@pytest.mark.heavy
def test_doctor_reports_both_environments_separately():
    r = subprocess.run([sys.executable, "-m", "aegis", "doctor"], cwd=str(ROOT),
                       capture_output=True, text=True, timeout=600)
    out = r.stdout
    assert "MAIN ENVIRONMENT" in out
    assert "DIARIZATION WORKER" in out
    assert out.index("MAIN ENVIRONMENT") < out.index("DIARIZATION WORKER")
    # the old excuse must be gone
    assert "false-MISS" not in out and "known false" not in out
    # token state is reported without ever printing a token value
    assert "HF token" in out
    assert "hf_" not in out.replace("hf_token.txt", "")


def test_readme_no_longer_instructs_stale_model_agreements():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "speaker-diarization-community-1" in readme
    # the stale 3.1 / segmentation-3.0 instructions must not be presented as
    # required steps any more
    for stale in ("hf.co/pyannote/speaker-diarization-3.1",
                  "hf.co/pyannote/segmentation-3.0"):
        assert f"- https://{stale}" not in readme
