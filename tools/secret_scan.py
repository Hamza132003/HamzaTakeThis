"""Secret scanner over git-tracked files. Exit 1 on findings. No dependencies.

Patterns require realistic lengths so documentation placeholders (hf_xxx, hf_…)
do not trigger. Run:  python tools/secret_scan.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PATTERNS = {
    "huggingface token": re.compile(r"hf_[A-Za-z0-9]{30,}"),
    "openai-style key": re.compile(r"sk-[A-Za-z0-9]{32,}"),
    "aws access key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "github token": re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}"),
    "private key block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "slack token": re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
}

SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".ico", ".woff", ".woff2", ".lock"}


def tracked_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                         text=True, check=True)
    return [ROOT / line for line in out.stdout.splitlines() if line.strip()]


def scan() -> int:
    findings: list[str] = []
    for path in tracked_files():
        if path.suffix.lower() in SKIP_SUFFIXES or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for name, pat in PATTERNS.items():
            for m in pat.finditer(text):
                line_no = text.count("\n", 0, m.start()) + 1
                findings.append(f"{path.relative_to(ROOT)}:{line_no}: {name}")
    if findings:
        print("SECRET SCAN: FINDINGS (do not commit):")
        for f in findings:
            print(" ", f)
        return 1
    print(f"SECRET SCAN: clean ({len(tracked_files())} tracked files).")
    return 0


if __name__ == "__main__":
    sys.exit(scan())
