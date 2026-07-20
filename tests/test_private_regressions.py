"""Private regression cases (definitions in tests/regression_cases/*.json).

The recordings themselves are NEVER in Git. Each case self-skips when its
audio is absent on this machine, so CI and clean clones stay green.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CASES = sorted((ROOT / "tests" / "regression_cases").glob("*.json"))

pytestmark = pytest.mark.heavy


def _load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.mark.parametrize("case_file", CASES, ids=lambda p: p.stem)
def test_case_definition_is_well_formed(case_file):
    """Always runs: the registry itself must stay valid and audio-free."""
    c = _load(case_file)
    for key in ("case_id", "audio_path", "audio_sha256", "expected",
                "known_hallucinated_phrases", "acceptance"):
        assert key in c, f"{case_file.name} missing '{key}'"
    assert len(c["audio_sha256"]) == 64
    # the registry must not embed audio
    assert case_file.stat().st_size < 20_000
    # and the audio must not live inside the repository
    assert ROOT not in Path(c["audio_path"]).parents, \
        "private audio must not be stored inside the repo"


def _case_ready(c: dict) -> bool:
    p = Path(c["audio_path"])
    if not p.is_file():
        return False
    from workers.diarization_worker.client import resolve_worker_python
    if not resolve_worker_python(None).exists():
        return False
    if not (ROOT / "hf_token.txt").exists():
        return False
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    return (cache / "models--pyannote--speaker-diarization-community-1").exists()


@pytest.mark.parametrize("case_file", CASES, ids=lambda p: p.stem)
def test_private_case_acceptance(case_file, tmp_path):
    c = _load(case_file)
    if not _case_ready(c):
        pytest.skip("private audio / worker env / token / weights unavailable here")

    audio = Path(c["audio_path"])
    actual = hashlib.sha256(audio.read_bytes()).hexdigest()
    assert actual == c["audio_sha256"], (
        "registered recording changed on disk; re-register the case rather "
        "than silently accepting different audio")

    outdir = tmp_path / "run"
    r = subprocess.run([sys.executable, "main.py", str(audio),
                        "--outdir", str(outdir)],
                       cwd=str(ROOT), capture_output=True, text=True, timeout=7200)
    assert r.returncode == 0, r.stdout[-1500:]
    rep = json.loads((outdir / audio.stem / "report.json").read_text(encoding="utf-8"))
    exp, acc = c["expected"], c["acceptance"]
    ds = rep["diarization_status"]
    failures = []

    if acc.get("genuine_pyannote") and not ds.get("genuine_pyannote"):
        failures.append(f"genuine_pyannote={ds.get('genuine_pyannote')}")
    if acc.get("fallback_used") is False and ds.get("fallback_used"):
        failures.append("fallback_used=True")

    if acc.get("both_speaker_labels_in_transcript"):
        labels = {s.get("speaker") for s in rep.get("speech", [])}
        missing = set(exp["speaker_labels_present"]) - labels
        if missing:
            failures.append(f"missing speaker labels in transcript: {sorted(missing)}")

    # Hallucinated outros must never appear as accepted (non-withheld) text.
    if acc.get("no_untriggered_outro_phrases"):
        for s in rep.get("speech", []):
            if (s.get("quality") or {}).get("unreliable"):
                continue                       # withheld: correctly handled
            blob = f"{s.get('text','')} {s.get('english','')}".lower()
            for phrase in c["known_hallucinated_phrases"]:
                if phrase.lower() in blob:
                    failures.append(f"outro phrase accepted as real: '{phrase}'")

    if acc.get("hallucinated_segments_not_translated"):
        for s in rep.get("speech", []):
            if (s.get("quality") or {}).get("unreliable"):
                if s.get("english") or s.get("arabic"):
                    failures.append("unreliable segment was translated")

    if acc.get("all_speech_accounted_for"):
        cov = rep.get("coverage")
        if not cov:
            failures.append("report has no coverage accounting")

    assert not failures, (
        f"{c['case_id']} acceptance failures:\n  - " + "\n  - ".join(failures))
