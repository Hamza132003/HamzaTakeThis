"""Ingest contracts (schema version 1).

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
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

INGEST_SCHEMA_VERSION = 1


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
    rel_path: str                          # RELATIVE to the manifest directory
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


class IngestManifest(_Contract):
    """Top-level record tying source → derived assets → diagnostics."""
    manifest_id: str
    source: AudioAsset
    derived: list[DerivedAudio] = Field(default_factory=list)
    condition_vector: Optional[ConditionVector] = None
    tool_versions: dict[str, str] = Field(default_factory=dict)
    created_utc: str
    warnings: list[str] = Field(default_factory=list)
    failure_reason: Optional[str] = None


ConditionVector.model_rebuild()
