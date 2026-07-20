"""Ingest contracts (schema version 2).

Rules enforced here (per approved spec §5 and Phase 1 binding requirements):
- every persisted contract carries `schema_version`;
- `extra="forbid"`: unknown keys are REJECTED at validation time — this is the
  documented unknown-key policy; accepting a new key requires a schema-version
  decision and a migration note in docs/INGEST.md;
- no calibrated-confidence field exists in any ingest contract, by design;
- raw model scores never appear here (ingest is model-free);
- portable manifests reference artifacts by RELATIVE path + content hash; the
  absolute source path is retained only in `source_path_local`, explicitly
  marked non-portable (privacy: it may contain a user-profile path).

SCHEMA VERSION DECISION — v1 → v2 (Phase 1 repair pass)
Reason: canonical artifacts moved out of the (OneDrive-synced) recording output
tree into the non-synced artifact store, and ingest failures became structured.
Changes:
- `DerivedAudio.rel_path` is now relative to the CANONICAL SOURCE DIRECTORY in
  the artifact store (was: relative to the recording output directory);
- `IngestManifest.store_relative_dir` added (store-root-relative canonical dir);
- `IngestManifest.failures` / `IngestFailure` added;
- `IngestManifest.analysis_block_frames` / `analysis_policy` added;
- `IngestPointer` added (small portable file left in the output directory).
Migration: v1 manifests remain readable by pinning `schema_version=1` at the
call site; no automatic rewrite is performed, and no v1 manifest is deleted.
See docs/INGEST.md "Schema history".
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

INGEST_SCHEMA_VERSION = 2


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: int = INGEST_SCHEMA_VERSION


class DiagnosticMeasurement(_Contract):
    """One numeric diagnostic. `kind` distinguishes measured facts from
    estimates/proxies — an SNR proxy is NEVER a measurement of true SNR."""
    name: str
    value: Optional[float] = None          # None when invalid/unavailable
    units: str
    method: str                            # short formula/algorithm reference
    kind: Literal["measured", "estimated", "proxy"]
    valid: bool = True
    reason: Optional[str] = None           # required when valid is False
    window: Optional[str] = None           # e.g. "full-file" | "win=2048 hop=512"


class ChannelMetadata(_Contract):
    index: int
    rms: Optional[float] = None
    peak: Optional[float] = None
    clipping_ratio: Optional[float] = None
    dropout_runs: Optional[int] = None
    warnings: list[str] = Field(default_factory=list)


class MediaMetadata(_Contract):
    """Structured probe output. Unavailable fields are None with a reason in
    `unavailable_reasons[field]` — never guessed."""
    container_format: Optional[str] = None
    audio_codec: Optional[str] = None
    codec_profile: Optional[str] = None
    n_audio_streams: Optional[int] = None
    selected_stream_index: Optional[int] = None
    stream_selection_rule: str = "first audio stream (index order)"
    duration_sec: Optional[float] = None
    bit_rate: Optional[int] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    channel_layout: Optional[str] = None
    sample_format: Optional[str] = None
    start_time_sec: Optional[float] = None
    time_base: Optional[str] = None
    disposition: Optional[str] = None
    tags: dict[str, str] = Field(default_factory=dict)
    has_video: Optional[bool] = None
    probe_tool: str = "pyav"
    probe_tool_version: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)
    unavailable_reasons: dict[str, str] = Field(default_factory=dict)


class AudioAsset(_Contract):
    """The immutable original. Identified by content, not by path."""
    asset_id: str                          # deterministic: "src-" + sha256[:16]
    source_filename: str                   # basename only (display-sanitized)
    source_path_local: Optional[str] = None  # NON-PORTABLE; may embed user paths
    sha256: str
    sha512: str
    byte_size: int
    created_utc: str                       # ISO-8601 UTC
    media: Optional[MediaMetadata] = None
    duplicate_of: Optional[str] = None     # asset_id of first-seen identical bytes
    warnings: list[str] = Field(default_factory=list)
    failure_reason: Optional[str] = None


class DerivedAudio(_Contract):
    """A derived artifact (per-channel PCM, mono mix, mid, side, legacy mono)."""
    asset_id: str                          # "drv-" + sha256[:16]
    parent_asset_id: str
    parent_sha256: str
    role: str                              # e.g. "channel_0" | "mono_mix" | "mid"
    transform: str                         # exact transform + scaling convention
    tool: str                              # e.g. "ffmpeg 7.x (imageio-ffmpeg)" | "numpy"
    rel_path: str                          # v2: RELATIVE to the canonical source
                                           # dir in the artifact store
    sha256: str
    byte_size: int
    created_utc: str
    sample_rate: int
    channels: int
    sample_format: str                     # e.g. "pcm_f32le" | "pcm_s16le"
    duration_sec: Optional[float] = None
    gain_adjusted: bool = False            # True only if any gain/normalization applied
    warnings: list[str] = Field(default_factory=list)
    failure_reason: Optional[str] = None


class ConditionVector(_Contract):
    """Deterministic, model-free signal diagnostics + categorical conditions.
    Every categorical condition retains its supporting measurements, the
    thresholds version used, a severity in [0,1], a reason, and a status that
    may be 'indeterminate'. No LLM assigns any condition."""
    thresholds_version: str
    analysis_mode: Literal["full-file", "windowed-sample"]
    measurements: list[DiagnosticMeasurement] = Field(default_factory=list)
    channel_metadata: list[ChannelMetadata] = Field(default_factory=list)
    conditions: list[CategoricalCondition] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class CategoricalCondition(_Contract):
    condition: str                         # e.g. "probable_narrowband_radio"
    status: Literal["present", "absent", "indeterminate"]
    severity: float = 0.0                  # 0..1; meaningful only when present
    reason: str
    supporting_measurements: list[str] = Field(default_factory=list)
    thresholds_version: str


class IngestFailure(_Contract):
    """Structured record of one ingest-stage failure. Never a bare log line:
    `ingest_manifest=None` must not be the only machine-readable signal."""
    stage: Literal["hash", "store", "probe", "channels", "diagnostics",
                   "manifest_write", "pointer_write", "unknown"]
    exception_type: str
    message: str                           # sanitized; never secrets/env dumps
    occurred_utc: str
    source_content_id: Optional[str] = None
    recoverable: bool = True               # False → evidence-grade ingest void
    legacy_continued: bool = True          # legacy transcript path still ran
    warnings: list[str] = Field(default_factory=list)


class IngestManifest(_Contract):
    """Top-level record tying source → derived assets → diagnostics."""
    manifest_id: str
    source: AudioAsset
    derived: list[DerivedAudio] = Field(default_factory=list)
    condition_vector: Optional[ConditionVector] = None
    tool_versions: dict[str, str] = Field(default_factory=dict)
    created_utc: str
    # v2 additions
    store_relative_dir: Optional[str] = None   # e.g. "sources/ab/abcd…"
    stages_completed: list[str] = Field(default_factory=list)
    failures: list[IngestFailure] = Field(default_factory=list)
    analysis_block_frames: Optional[int] = None   # bounded-memory block size
    analysis_policy: Optional[str] = None         # human-readable policy note
    warnings: list[str] = Field(default_factory=list)
    failure_reason: Optional[str] = None          # summary of `failures`

    @property
    def ok(self) -> bool:
        return not self.failures and self.failure_reason is None


class IngestPointer(_Contract):
    """Small portable file left in the recording output directory pointing at
    the canonical manifest in the artifact store. Deliberately contains NO
    absolute path: the store root is resolved from configuration at read time,
    so the pointer stays portable and leaks no user-profile path."""
    source_sha256: str
    source_asset_id: str
    store_relative_dir: str                # resolve against storage.artifact_dir
    manifest_relative_path: str            # e.g. "sources/ab/abcd…/ingest_manifest.json"
    created_utc: str
    ok: bool = True
    failure_reason: Optional[str] = None


ConditionVector.model_rebuild()
