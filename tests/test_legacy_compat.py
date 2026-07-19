"""Legacy byte-compatibility regression (Phase 1 acceptance §7).

Compares:
1. the legacy extraction as it existed at the Phase 0 checkpoint — the
   REFERENCE below is a verbatim copy of `extract_audio` from commit
   3255a532c487ae1cc2197cb31e7217590108d81c (pipeline/audio.py); and
2. the new compatibility façade `pipeline.audio.extract_audio`.

Acceptance: byte-identical `audio_16k_mono.wav` and `audio_48k_mono.wav` for a
deterministic fixture (same ffmpeg binary, same commands → identical bytes).
Heavy: requires the bundled ffmpeg.
"""
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytestmark = pytest.mark.heavy

np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")
pytest.importorskip("pydantic")
imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")

from pipeline.audio import extract_audio  # noqa: E402


def _reference_phase0_extract(input_path, out_dir, sample_rate,
                              hq_sample_rate=48000):
    """Verbatim logic of the Phase 0 checkpoint extract_audio."""
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    out_dir.mkdir(parents=True, exist_ok=True)
    work_wav = out_dir / "audio_16k_mono.wav"
    hq_wav = out_dir / "audio_48k_mono.wav"
    for target, args in (
        (work_wav, ["-ac", "1", "-ar", str(sample_rate)]),
        (hq_wav, ["-ac", "1", "-ar", str(hq_sample_rate)]),
    ):
        cmd = [ffmpeg, "-y", "-i", str(input_path), "-vn",
               "-acodec", "pcm_s16le", *args, str(target)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode == 0 and target.exists()
    return {"work_wav": work_wav, "hq_wav": hq_wav}


def _fixture(tmp_path):
    sr = 44100
    t = np.arange(int(sr * 1.5)) / sr
    left = 0.4 * np.sin(2 * np.pi * 440 * t)
    right = 0.3 * np.sin(2 * np.pi * 660 * t)
    p = tmp_path / "src.wav"
    sf.write(str(p), np.stack([left, right], axis=1).astype(np.float32), sr,
             subtype="FLOAT")
    return p


def test_facade_legacy_files_byte_identical_to_phase0(tmp_path):
    src = _fixture(tmp_path)
    ref = _reference_phase0_extract(src, tmp_path / "ref", 16000)
    new = extract_audio(src, tmp_path / "new", 16000, cfg={"ingest": {"enabled": True}})
    for key in ("work_wav", "hq_wav"):
        b_ref = Path(ref[key]).read_bytes()
        b_new = Path(new[key]).read_bytes()
        assert b_ref == b_new, f"{key} not byte-identical to Phase 0 output"


def test_facade_returns_legacy_keys_plus_manifest(tmp_path):
    src = _fixture(tmp_path)
    out = extract_audio(src, tmp_path / "o", 16000, cfg={"ingest": {"enabled": True},
                                                         "storage": {"artifact_dir":
                                                                     str(tmp_path / "store")}})
    assert set(out) == {"work_wav", "hq_wav", "ingest_manifest"}
    assert out["work_wav"].name == "audio_16k_mono.wav"
    assert out["hq_wav"].name == "audio_48k_mono.wav"
    m = out["ingest_manifest"]
    assert m is not None and m.source.sha256 and m.source.sha512
    roles = {d.role for d in m.derived}
    assert {"channel_0", "channel_1", "mono_mix", "mid", "side"} <= roles
    assert m.condition_vector is not None
    assert (tmp_path / "o" / "ingest_manifest.json").exists()


def test_ingest_disabled_matches_legacy_only(tmp_path):
    src = _fixture(tmp_path)
    out = extract_audio(src, tmp_path / "o", 16000, cfg={"ingest": {"enabled": False}})
    assert out["ingest_manifest"] is None
    assert not (tmp_path / "o" / "ingest").exists()
    assert not (tmp_path / "o" / "ingest_manifest.json").exists()
    assert (tmp_path / "o" / "audio_16k_mono.wav").exists()


def test_source_untouched_by_facade(tmp_path):
    src = _fixture(tmp_path)
    before = src.read_bytes()
    extract_audio(src, tmp_path / "o", 16000,
                  cfg={"storage": {"artifact_dir": str(tmp_path / "store")}})
    assert src.read_bytes() == before


def test_duplicate_detection_through_facade(tmp_path):
    src = _fixture(tmp_path)
    dup = tmp_path / "copy.wav"
    dup.write_bytes(src.read_bytes())
    cfg = {"storage": {"artifact_dir": str(tmp_path / "store")}}
    m1 = extract_audio(src, tmp_path / "o1", 16000, cfg=cfg)["ingest_manifest"]
    m2 = extract_audio(dup, tmp_path / "o2", 16000, cfg=cfg)["ingest_manifest"]
    assert m1.source.duplicate_of is None
    assert m2.source.duplicate_of == m1.source.asset_id
    assert m2.source.warnings


def test_runner_and_webapp_still_import():
    import pipeline.runner  # noqa: F401
    import webapp.app  # noqa: F401
