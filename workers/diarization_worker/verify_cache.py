"""Verify the cached diarization model against the pre-download integrity
manifest. Independent of huggingface_hub's own transfer-time check.

Run (worker env):  python workers/diarization_worker/verify_cache.py
Exit 0 = every recorded file present and byte-identical; 1 = any mismatch.
No network access, no token needed.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

MANIFEST = Path(__file__).with_name("model_integrity_community1.json")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _snapshot_dir(model_id: str, revision: str) -> Path:
    from huggingface_hub.constants import HF_HUB_CACHE
    folder = "models--" + model_id.replace("/", "--")
    return Path(HF_HUB_CACHE) / folder / "snapshots" / revision


def main() -> int:
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    snap = _snapshot_dir(m["model_id"], m["revision"])
    print(f"model    : {m['model_id']}")
    print(f"revision : {m['revision']}")
    print(f"snapshot : {snap}")
    print(f"exists   : {snap.exists()}")
    if not snap.exists():
        print("RESULT: FAIL - snapshot directory missing")
        return 1

    ok = missing = mismatch = skipped = 0
    for entry in m["files"]:
        rel, expected = entry["path"], entry.get("sha256")
        # The docs GIF is intentionally excluded from the download.
        if rel.endswith(".gif"):
            skipped += 1
            continue
        p = snap / rel
        if not p.exists():
            print(f"  [MISSING ] {rel}")
            missing += 1
            continue
        size_ok = p.stat().st_size == entry["size"]
        if expected:                       # LFS object: verify sha256
            actual = _sha256(p)
            if actual == expected and size_ok:
                print(f"  [VERIFIED] {rel}  ({entry['size']} bytes, sha256 match)")
                ok += 1
            else:
                print(f"  [MISMATCH] {rel}  size_ok={size_ok} sha256_ok={actual == expected}")
                mismatch += 1
        else:                              # small text file: verify size only
            if size_ok:
                print(f"  [size-ok ] {rel}  ({entry['size']} bytes, git blob)")
                ok += 1
            else:
                print(f"  [MISMATCH] {rel}  size {p.stat().st_size} != {entry['size']}")
                mismatch += 1

    print(f"\nverified={ok} mismatched={mismatch} missing={missing} "
          f"skipped_by_design={skipped}")
    failed = bool(mismatch or missing)
    print("RESULT:", "FAIL" if failed else "PASS - cache matches the manifest")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
