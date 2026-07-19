"""Channel factory: per-channel preservation, mid/side math, no fake assets,
no clipping from matrixing, source untouched. Heavy: needs bundled ffmpeg."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytestmark = pytest.mark.heavy

np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")
pytest.importorskip("pydantic")
pytest.importorskip("imageio_ffmpeg")

from pipeline.ingest.channel_factory import build_channel_assets  # noqa: E402

SR = 16000


def _write_src(tmp_path, data, name="src.wav"):
    # Float WAV: keeps fixture samples exact. (soundfile's WAV default is
    # 16-bit PCM, which quantizes at write time and would make "exact
    # preservation" assertions compare against pre-quantization values.)
    p = tmp_path / name
    sf.write(str(p), data, SR, subtype="FLOAT")
    return p


def _tone(freq, seconds=0.5, amp=0.5):
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _load(out_dir, rel):
    y, sr = sf.read(str(out_dir / rel), dtype="float32")
    return y, sr


def test_stereo_hard_panned_channels_preserved(tmp_path):
    left, right = _tone(440), _tone(880)
    src = _write_src(tmp_path, np.stack([left, right], axis=1))
    out = tmp_path / "out"
    contracts, audio, sr = build_channel_assets(src, out, "src-x", "a" * 64)
    roles = {c.role for c in contracts}
    assert roles == {"channel_0", "channel_1", "mono_mix", "mid", "side"}
    ch0, _ = _load(out, "ingest/channel_0.wav")
    ch1, _ = _load(out, "ingest/channel_1.wav")
    # float32 WAV round-trip through ffmpeg decode is exact for float sources
    assert np.array_equal(ch0, left) and np.array_equal(ch1, right)


def test_mid_side_exact_math_and_no_matrix_clipping(tmp_path):
    left = _tone(440, amp=0.9)
    right = -left                                  # worst case for side
    src = _write_src(tmp_path, np.stack([left, right], axis=1))
    out = tmp_path / "out"
    build_channel_assets(src, out, "src-x", "a" * 64)
    mid, _ = _load(out, "ingest/mid.wav")
    side, _ = _load(out, "ingest/side.wav")
    assert np.allclose(mid, (left + right) / 2, atol=1e-7)
    assert np.allclose(side, (left - right) / 2, atol=1e-7)
    # /2 convention bounds output below source peak: no clipping introduced
    assert np.max(np.abs(side)) <= np.max(np.abs(left)) + 1e-7


def test_mono_source_gets_no_fake_mid_side(tmp_path):
    src = _write_src(tmp_path, _tone(300))
    out = tmp_path / "out"
    contracts, audio, sr = build_channel_assets(src, out, "src-x", "a" * 64)
    assert {c.role for c in contracts} == {"mono_mix"}
    assert not (out / "ingest" / "mid.wav").exists()
    assert not (out / "ingest" / "side.wav").exists()


def test_four_channel_source_preserves_every_channel(tmp_path):
    chans = [_tone(200 * (i + 1), amp=0.2) for i in range(4)]
    src = _write_src(tmp_path, np.stack(chans, axis=1))
    out = tmp_path / "out"
    contracts, audio, sr = build_channel_assets(src, out, "src-x", "a" * 64)
    roles = {c.role for c in contracts}
    assert {"channel_0", "channel_1", "channel_2", "channel_3",
            "mono_mix"} == roles                    # downmix without discarding
    mix, _ = _load(out, "ingest/mono_mix.wav")
    assert np.allclose(mix, np.mean(np.stack(chans), axis=0), atol=1e-7)


def test_identical_stereo_channels_side_is_silence(tmp_path):
    ch = _tone(500)
    src = _write_src(tmp_path, np.stack([ch, ch], axis=1))
    out = tmp_path / "out"
    build_channel_assets(src, out, "src-x", "a" * 64)
    side, _ = _load(out, "ingest/side.wav")
    assert np.max(np.abs(side)) < 1e-7


def test_source_bytes_untouched_and_no_gain_flag(tmp_path):
    src = _write_src(tmp_path, np.stack([_tone(440), _tone(880)], axis=1))
    before = src.read_bytes()
    contracts, _, _ = build_channel_assets(src, tmp_path / "o", "src-x", "a" * 64)
    assert src.read_bytes() == before
    assert all(c.gain_adjusted is False for c in contracts)
    assert all(not Path(c.rel_path).is_absolute() for c in contracts)


def test_derived_hashes_reproducible(tmp_path):
    src = _write_src(tmp_path, np.stack([_tone(440), _tone(880)], axis=1))
    c1, _, _ = build_channel_assets(src, tmp_path / "o1", "src-x", "a" * 64)
    c2, _, _ = build_channel_assets(src, tmp_path / "o2", "src-x", "a" * 64)
    h1 = {c.role: c.sha256 for c in c1}
    h2 = {c.role: c.sha256 for c in c2}
    assert h1 == h2                                # deterministic artifacts
