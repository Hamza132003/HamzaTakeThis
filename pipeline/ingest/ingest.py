"""Ingest orchestrator (Phase 1): hash → probe → channel assets → diagnostics
→ manifest. Additive only — never touches the source or the legacy outputs.
"""
from __future__ import annotations

import json
from pathlib import Path

from pipeline.contracts import AudioAsset, IngestManifest
from pipeline.ingest.channel_factory import build_channel_assets
from pipeline.ingest.immutable import (
    DuplicateIndex,
    atomic_write_bytes,
    content_id,
    resolve_store_dir,
    sanitize_display_name,
    stream_hashes,
    utc_now_iso,
)
from pipeline.ingest.media_probe import probe
from pipeline.utils import log

MANIFEST_NAME = "ingest_manifest.json"


def _tool_versions() -> dict[str, str]:
    out = {}
    for mod in ("numpy", "scipy", "av", "soundfile", "pydantic"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = "unavailable"
    try:
        import imageio_ffmpeg
        out["ffmpeg"] = imageio_ffmpeg.get_ffmpeg_version()
    except Exception:
        out["ffmpeg"] = "unavailable"
    return out


def ingest_file(src: Path, out_dir: Path, cfg: dict | None = None) -> IngestManifest:
    """Full Phase 1 ingest of one source file. The source is only ever read."""
    src = Path(src)
    out_dir = Path(out_dir)
    sha256, sha512, size = stream_hashes(src)       # hashed BEFORE decoding
    asset_id = content_id(sha256)
    display = sanitize_display_name(src.name)

    dup_of = None
    try:
        store = resolve_store_dir(cfg)
        dup_of = DuplicateIndex(store).check_and_register(sha256, asset_id, display)
    except OSError as e:
        log(f"  (duplicate index unavailable: {e})")

    source = AudioAsset(
        asset_id=asset_id, source_filename=display,
        source_path_local=str(src),                 # NON-PORTABLE by contract
        sha256=sha256, sha512=sha512, byte_size=size,
        created_utc=utc_now_iso(), media=probe(src), duplicate_of=dup_of)
    if dup_of:
        source.warnings.append(
            f"exact-byte duplicate of previously ingested asset {dup_of}; "
            f"not an independent recording")

    manifest = IngestManifest(
        manifest_id=f"man-{sha256[:16]}", source=source,
        tool_versions=_tool_versions(), created_utc=utc_now_iso())

    try:
        derived, audio, sr = build_channel_assets(src, out_dir, asset_id, sha256)
        manifest.derived = derived
        from pipeline.diagnostics import analyse
        manifest.condition_vector = analyse(audio, sr)
    except Exception as e:
        manifest.failure_reason = f"channel/diagnostic stage failed: {e}"
        log(f"WARNING: ingest degraded ({e}); legacy pipeline unaffected.")

    atomic_write_bytes(out_dir / MANIFEST_NAME,
                       manifest.model_dump_json(indent=1).encode("utf-8"))
    return manifest


def load_manifest(out_dir: Path) -> IngestManifest | None:
    p = Path(out_dir) / MANIFEST_NAME
    if not p.exists():
        return None
    return IngestManifest.model_validate_json(p.read_text(encoding="utf-8"))
