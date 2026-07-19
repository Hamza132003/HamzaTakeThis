"""Release gate blocks while the license is deferred; secret scanner catches tokens."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def test_release_gate_currently_blocked():
    # Binding decision 2: license deferred → the gate MUST fail right now.
    r = subprocess.run([PY, str(ROOT / "tools" / "release_gate.py")],
                       capture_output=True, text=True)
    assert r.returncode == 1
    assert "RELEASE BLOCKED" in r.stdout
    assert "LICENSE" in r.stdout


def test_release_gate_warns_about_nllb_nc():
    r = subprocess.run([PY, str(ROOT / "tools" / "release_gate.py")],
                       capture_output=True, text=True)
    assert "CC-BY-NC" in r.stdout


def test_secret_scan_clean_on_repo():
    r = subprocess.run([PY, str(ROOT / "tools" / "secret_scan.py")],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout


def test_secret_pattern_catches_realistic_token():
    sys.path.insert(0, str(ROOT))
    from tools.secret_scan import PATTERNS
    fake = "hf_" + "A" * 34
    assert any(p.search(f"token = '{fake}'") for p in PATTERNS.values())
    # Documentation placeholders must NOT trigger.
    assert not any(p.search("paste hf_xxx into the field") for p in PATTERNS.values())


def test_gitignore_excludes_sensitive_paths():
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for required in ("hf_token.txt", "outputs/", "uploads/", "*.wav", "server.log"):
        assert required in gitignore, f"missing {required} in .gitignore"
