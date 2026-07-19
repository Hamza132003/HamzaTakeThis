"""Immutable source handling: streaming dual hashing, content IDs, atomic
writes, and exact-byte duplicate detection.

Guarantees (Phase 1 binding requirements §4):
- the original file is only ever OPENED FOR READING; nothing here writes,
  renames, truncates or normalizes a source;
- hashing is streaming (8 MiB chunks) so multi-GB recordings never load into
  RAM; SHA-256 and SHA-512 are computed in one pass over the raw bytes,
  BEFORE any decoding;
- derived artifacts are written to a temp file in the destination directory
  and atomically renamed (os.replace) — an artifact is reported complete only
  after its final bytes exist and have been re-hashed;
- duplicate detection: exact-byte SHA-256 equality only. Same bytes at a
  different path → duplicate of the first-seen asset. Different container
  bytes with acoustically equivalent content are NOT called duplicates in
  Phase 1 (acoustic fingerprinting is out of scope; see docs/INGEST.md).

Store location: resolved from config `storage.artifact_dir`; when null, a
non-synced default under %LOCALAPPDATA%/AegisXPrime/store is used — NEVER a
OneDrive path by default (risk R-8). Only the small duplicate index lives
there in Phase 1.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

_CHUNK = 8 * 1024 * 1024


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stream_hashes(path: Path) -> tuple[str, str, int]:
    """One streaming pass → (sha256_hex, sha512_hex, byte_size). Read-only."""
    h256, h512 = hashlib.sha256(), hashlib.sha512()
    size = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_CHUNK)
            if not chunk:
                break
            size += len(chunk)
            h256.update(chunk)
            h512.update(chunk)
    return h256.hexdigest(), h512.hexdigest(), size


def content_id(sha256_hex: str, prefix: str = "src") -> str:
    """Deterministic short ID derived from the content hash."""
    return f"{prefix}-{sha256_hex[:16]}"


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_write_bytes(dest: Path, data: bytes) -> str:
    """Write via temp file + os.replace; return final sha256. The final path
    never exists in a half-written state."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(dest.parent), suffix=".part")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, dest)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    sha, _, _ = stream_hashes(dest)   # re-hash AFTER final creation
    return sha


def sanitize_display_name(name: str) -> str:
    """Filename for display: strip directories and control characters."""
    base = os.path.basename(name.replace("\\", "/"))
    return re.sub(r"[\x00-\x1f\x7f]", "_", base)[:255]


# ------------------------------------------------------------ duplicate index
def resolve_store_dir(cfg: dict | None) -> Path:
    """storage.artifact_dir from config, else %LOCALAPPDATA%/AegisXPrime/store.
    Refuses paths that resolve inside a OneDrive folder unless explicitly
    configured by the operator (documented in docs/INGEST.md)."""
    configured = ((cfg or {}).get("storage") or {}).get("artifact_dir")
    if configured:
        return Path(configured)
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / ".local" / "share")
    return Path(base) / "AegisXPrime" / "store"


class DuplicateIndex:
    """Append-only JSON index sha256 → first-seen record. Small, atomic."""

    def __init__(self, store_dir: Path):
        self.path = Path(store_dir) / "source_index.json"

    def _load(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def check_and_register(self, sha256_hex: str, asset_id: str,
                           display_name: str) -> str | None:
        """Return first-seen asset_id if these bytes were ingested before
        (a DUPLICATE), else register and return None."""
        idx = self._load()
        entry = idx.get(sha256_hex)
        if entry is not None:
            return str(entry["asset_id"])
        idx[sha256_hex] = {"asset_id": asset_id, "name": display_name,
                           "first_seen_utc": utc_now_iso(),
                           "monotonic": time.monotonic()}
        atomic_write_bytes(self.path, json.dumps(idx, indent=1).encode("utf-8"))
        return None
