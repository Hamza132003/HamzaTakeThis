"""Ingest-contract tests: round-trip, JSON-schema export, unknown-key policy,
schema versioning, and the no-calibrated-confidence rule."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pydantic = pytest.importorskip("pydantic")

from pipeline.contracts import (  # noqa: E402
    INGEST_SCHEMA_VERSION,
    AudioAsset,
    CategoricalCondition,
    ConditionVector,
    DerivedAudio,
    DiagnosticMeasurement,
    IngestManifest,
    MediaMetadata,
)

ALL_MODELS = [AudioAsset, CategoricalCondition, ConditionVector, DerivedAudio,
              DiagnosticMeasurement, IngestManifest, MediaMetadata]


def _asset(**kw) -> AudioAsset:
    base = dict(asset_id="src-0011223344556677", source_filename="clip.wav",
                sha256="a" * 64, sha512="b" * 128, byte_size=1234,
                created_utc="2026-07-19T00:00:00Z")
    base.update(kw)
    return AudioAsset(**base)


def test_round_trip_manifest():
    m = IngestManifest(
        manifest_id="man-1", source=_asset(),
        derived=[DerivedAudio(
            asset_id="drv-0011223344556677", parent_asset_id="src-0011223344556677",
            parent_sha256="a" * 64, role="mono_mix",
            transform="mean(ch0..chN) / 1.0 (no gain)", tool="numpy",
            rel_path="ingest/mono_mix.wav", sha256="c" * 64, byte_size=10,
            created_utc="2026-07-19T00:00:00Z", sample_rate=48000, channels=1,
            sample_format="pcm_f32le")],
        condition_vector=ConditionVector(
            thresholds_version="diag-v1", analysis_mode="full-file",
            measurements=[DiagnosticMeasurement(
                name="rms", value=0.1, units="linear", method="sqrt(mean(x^2))",
                kind="measured")],
            conditions=[CategoricalCondition(
                condition="probable_clipping", status="absent", reason="ratio<thr",
                supporting_measurements=["clipping_ratio"],
                thresholds_version="diag-v1")]),
        created_utc="2026-07-19T00:00:00Z")
    dumped = m.model_dump_json()
    restored = IngestManifest.model_validate_json(dumped)
    assert restored == m


def test_every_model_exports_json_schema():
    for model in ALL_MODELS:
        schema = model.model_json_schema()
        assert "properties" in schema
        assert "schema_version" in schema["properties"], model.__name__


def test_unknown_keys_rejected():
    # Documented policy: extra="forbid" — accepting a new key requires a
    # schema-version decision.
    with pytest.raises(pydantic.ValidationError):
        AudioAsset.model_validate({**_asset().model_dump(), "surprise": 1})


def test_schema_version_present_and_current():
    assert _asset().schema_version == INGEST_SCHEMA_VERSION == 1


def test_no_calibrated_confidence_fields():
    for model in ALL_MODELS:
        for field_name in model.model_fields:
            assert "confidence" not in field_name.lower(), (
                f"{model.__name__}.{field_name}: calibrated-confidence fields "
                f"are banned in ingest contracts")


def test_measurement_kind_is_constrained():
    with pytest.raises(pydantic.ValidationError):
        DiagnosticMeasurement(name="snr", value=1.0, units="dB", method="q90-q10",
                              kind="ground_truth")  # not an allowed kind


def test_portable_manifest_paths_are_relative():
    d = DerivedAudio(
        asset_id="drv-1", parent_asset_id="src-1", parent_sha256="a" * 64,
        role="mid", transform="(L+R)/2", tool="numpy",
        rel_path="ingest/mid.wav", sha256="d" * 64, byte_size=1,
        created_utc="2026-07-19T00:00:00Z", sample_rate=48000, channels=1,
        sample_format="pcm_f32le")
    assert not Path(d.rel_path).is_absolute()
