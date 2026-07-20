"""Typed, versioned contracts for AEGIS-X PRIME (Phase 1: ingest)."""
from .ingest import (  # noqa: F401
    INGEST_SCHEMA_VERSION,
    AudioAsset,
    CategoricalCondition,
    ChannelMetadata,
    ConditionVector,
    DerivedAudio,
    DiagnosticMeasurement,
    IngestFailure,
    IngestManifest,
    IngestPointer,
    MediaMetadata,
)
