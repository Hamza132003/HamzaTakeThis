"""Release gate. Exit 0 only when releasing/packaging is permitted.

Currently expected to BLOCK: the project license decision is deferred
(docs/LICENSE_DECISION.md) and NLLB weights are CC-BY-NC-4.0.
Run:  python tools/release_gate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run() -> int:
    blockers: list[str] = []
    warnings: list[str] = []

    if not any((ROOT / n).exists() for n in ("LICENSE", "LICENSE.md", "LICENSE.txt")):
        blockers.append("no project LICENSE file — owner decision deferred "
                        "(docs/LICENSE_DECISION.md)")
    if (ROOT / "RELEASE_BLOCKED.md").exists():
        blockers.append("RELEASE_BLOCKED.md present — remove only together with a "
                        "recorded owner license decision")

    try:
        licenses = (ROOT / "MODEL_LICENSES.yaml").read_text(encoding="utf-8")
        if "CC-BY-NC" in licenses:
            warnings.append("non-commercial model weights in use (NLLB CC-BY-NC-4.0): "
                            "commercial distribution profile unavailable")
    except OSError:
        blockers.append("MODEL_LICENSES.yaml missing")

    for w in warnings:
        print(f"RELEASE GATE WARNING: {w}")
    if blockers:
        print("RELEASE BLOCKED:")
        for b in blockers:
            print(f"  - {b}")
        return 1
    print("RELEASE GATE: open.")
    return 0


if __name__ == "__main__":
    sys.exit(run())
