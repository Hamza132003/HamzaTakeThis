"""Ingest orchestrator (Phase 1, repaired): hash → store → probe → channel
assets → diagnostics → canonical manifest → output-dir pointer.

Canonical artifacts live in the NON-SYNCED artifact store, content-addressed by
source SHA-256:

    <store>/sources/<sha[:2]>/<sha>/ingest_manifest.json
    <store>/sources/<sha[:2]>/<sha>/assets/{channel_N,mono_mix,mid,side}.wav

The recording's output directory receives only a small portable pointer
(`ingest_pointer.json`) that carries store-RELATIVE paths — never an absolute
user-profile path — so nothing large and nothing personal is written into the
(possibly OneDrive-synced, possibly Git-tracked) working tree. Legacy 16 kHz /
48 kHz mono outputs are untouched and stay exactly where the runner and
dashboard expect them.

Failures never raise: every stage is wrapped and recorded as a structured
`IngestFailure` (stage, exception type, sanitized message, timestamp,
recoverability, whether the legacy path continued). The function always returns
an `IngestManifest`, so `ingest_manifest=None` is never the only machine-
readable signal.
"""
from __future__ import annotations

from pathlib import Path

from pipeline.contracts import (
    AudioAsset,
    IngestFailure,
    IngestManifest,
    IngestPointer,
    MediaMetadata,
)
from pipeline.ingest.channel_factory import BLOCK_FRAMES, build_channel_assets
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
POINTER_NAME = "ingest_pointer.json"
_MAX_MSG = 300


def _sanitize(exc: BaseException) -> str:
    """Short, single-line, secret-free message. Never dumps environment data."""
    return " ".join(str(exc).split())[:_MAX_MSG]


def _fail(stage, exc, content_id_=None, recoverable=True) -> IngestFailure:
    return IngestFailure(
        stage=stage, exception_type=type(exc).__name__, message=_sanitize(exc),
        occurred_utc=utc_now_iso(), source_content_id=content_id_,
        recoverable=recoverable, legacy_continued=True)


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


def canonical_dirs(store_dir: Path, sha256: str) -> tuple[Path, str]:
    """(absolute canonical source dir, store-relative posix path).

    Content-addressed and validated: the resolved directory must stay inside
    the store root, so a crafted hash-like value cannot escape it.
    """
    if not (len(sha256) == 64 and all(c in "0123456789abcdef" for c in sha256)):
        raise ValueError("source content hash is not a 64-char lowercase hex digest")
    rel = f"sources/{sha256[:2]}/{sha256}"
    root = Path(store_dir).resolve()
    target = (root / rel).resolve()
    if root != target and root not in target.parents:
        raise ValueError("canonical path escapes the configured artifact store")
    return target, rel


def ingest_file(src: Path, out_dir: Path, cfg: dict | None = None) -> IngestManifest:
    """Full Phase 1 ingest of one source file. The source is only ever read.

    Never raises: stage failures are recorded structurally and the manifest is
    still returned (and persisted where persistence is still possible).
    """
    src, out_dir = Path(src), Path(out_dir)
    failures: list[IngestFailure] = []
    stages: list[str] = []
    display = sanitize_display_name(src.name)

    # ---- stage: hash (before any decoding) --------------------------------
    try:
        sha256, sha512, size = stream_hashes(src)
        asset_id = content_id(sha256)
        stages.append("hash")
    except Exception as e:
        f = _fail("hash", e, recoverable=False)
        log(f"WARNING: evidence-grade ingest did NOT complete — hashing failed "
            f"({f.exception_type}: {f.message}); legacy pipeline continues.")
        return IngestManifest(
            manifest_id="man-unhashed",
            source=AudioAsset(asset_id="src-unknown", source_filename=display,
                              source_path_local=str(src), sha256="", sha512="",
                              byte_size=0, created_utc=utc_now_iso(),
                              failure_reason=f.message),
            created_utc=utc_now_iso(), failures=[f], stages_completed=stages,
            failure_reason=f"hash stage failed: {f.message}")

    # ---- stage: store resolution + duplicate index ------------------------
    canon_dir: Path | None = None
    store_rel: str | None = None
    dup_of = None
    try:
        store = resolve_store_dir(cfg)
        canon_dir, store_rel = canonical_dirs(store, sha256)
        canon_dir.mkdir(parents=True, exist_ok=True)
        stages.append("store")
        try:
            dup_of = DuplicateIndex(store).check_and_register(
                sha256, asset_id, display)
        except Exception as e:
            failures.append(_fail("store", e, asset_id))
    except Exception as e:
        failures.append(_fail("store", e, asset_id, recoverable=False))
        log(f"WARNING: artifact store unavailable ({type(e).__name__}); "
            f"canonical ingest artifacts will NOT be written.")

    # ---- stage: probe -----------------------------------------------------
    try:
        media: MediaMetadata | None = probe(src)
        stages.append("probe")
    except Exception as e:
        media = None
        failures.append(_fail("probe", e, asset_id))

    source = AudioAsset(
        asset_id=asset_id, source_filename=display,
        source_path_local=str(src),                 # NON-PORTABLE by contract
        sha256=sha256, sha512=sha512, byte_size=size,
        created_utc=utc_now_iso(), media=media, duplicate_of=dup_of)
    if dup_of:
        source.warnings.append(
            f"exact-byte duplicate of previously ingested asset {dup_of}; "
            f"not an independent recording")

    manifest = IngestManifest(
        manifest_id=f"man-{sha256[:16]}", source=source,
        tool_versions=_tool_versions(), created_utc=utc_now_iso(),
        store_relative_dir=store_rel, analysis_block_frames=BLOCK_FRAMES,
        analysis_policy=(
            f"channel extraction streams {BLOCK_FRAMES}-frame blocks (no "
            f"whole-file load); diagnostics read the full file only below the "
            f"documented duration threshold, otherwise 3 deterministic "
            f"seek-read windows"),
        failures=failures, stages_completed=stages)

    # ---- stage: channel assets (bounded memory) ---------------------------
    if canon_dir is not None:
        try:
            manifest.derived = build_channel_assets(
                src, canon_dir / "assets", asset_id, sha256)
            stages.append("channels")
        except Exception as e:
            failures.append(_fail("channels", e, asset_id))
            log(f"WARNING: channel preservation failed ({type(e).__name__}); "
                f"evidence-grade ingest incomplete.")

    # ---- stage: diagnostics (bounded memory) ------------------------------
    mono = next((d for d in manifest.derived if d.role == "mono_mix"), None)
    if mono is not None and canon_dir is not None:
        try:
            from pipeline.diagnostics import analyse_path
            manifest.condition_vector = analyse_path(canon_dir / mono.rel_path)
            stages.append("diagnostics")
        except Exception as e:
            failures.append(_fail("diagnostics", e, asset_id))

    manifest.stages_completed = stages
    manifest.failures = failures
    if failures:
        manifest.failure_reason = "; ".join(
            f"{f.stage}: {f.exception_type}" for f in failures)

    # ---- stage: canonical manifest write ----------------------------------
    manifest_rel = None
    if canon_dir is not None:
        try:
            atomic_write_bytes(canon_dir / MANIFEST_NAME,
                               manifest.model_dump_json(indent=1).encode("utf-8"))
            manifest_rel = f"{store_rel}/{MANIFEST_NAME}"
            stages.append("manifest_write")
        except Exception as e:
            failures.append(_fail("manifest_write", e, asset_id))

    # ---- stage: portable pointer in the recording output dir --------------
    try:
        pointer = IngestPointer(
            source_sha256=sha256, source_asset_id=asset_id,
            store_relative_dir=store_rel or "",
            manifest_relative_path=manifest_rel or "",
            created_utc=utc_now_iso(), ok=not failures,
            failure_reason=manifest.failure_reason)
        atomic_write_bytes(out_dir / POINTER_NAME,
                           pointer.model_dump_json(indent=1).encode("utf-8"))
    except Exception as e:
        failures.append(_fail("pointer_write", e, asset_id))

    manifest.stages_completed = stages
    manifest.failures = failures
    if failures and manifest.failure_reason is None:
        manifest.failure_reason = "; ".join(
            f"{f.stage}: {f.exception_type}" for f in failures)
    if failures:
        log("WARNING: evidence-grade ingest did NOT complete cleanly "
            f"({manifest.failure_reason}); legacy pipeline continues.")
    return manifest


def load_manifest(canonical_dir: Path) -> IngestManifest | None:
    p = Path(canonical_dir) / MANIFEST_NAME
    if not p.exists():
        return None
    return IngestManifest.model_validate_json(p.read_text(encoding="utf-8"))


def load_pointer(out_dir: Path) -> IngestPointer | None:
    p = Path(out_dir) / POINTER_NAME
    if not p.exists():
        return None
    return IngestPointer.model_validate_json(p.read_text(encoding="utf-8"))
