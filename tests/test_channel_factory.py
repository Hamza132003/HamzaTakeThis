"""Channel factory: per-channel preservation, mid/side math, no fake assets,
no clipping from matrixing, source untouched, and BOUNDED MEMORY (no
whole-file audio load). Heavy: needs bundled ffmpeg."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytestmark = pytest.mark.heavy

np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")
pytest.importorskip("pydantic")
pytest.importorskip("imageio_ffmpeg")

from pipeline.ingest import channel_factory  # noqa: E402
from pipeline.ingest.channel_factory import BLOCK_FRAMES, build_channel_assets  # noqa: E402

SR = 16000


def _write_src(tmp_path, data, name="src.wav"):
    # Float WAV keeps fixture samples exact (soundfile's WAV default is 16-bit
    # PCM, which would quantize before the comparison).
    p = tmp_path / name
    sf.write(str(p), data, SR, subtype="FLOAT")
    return p


def _tone(freq, seconds=0.5, amp=0.5, sr=SR):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _load(assets_dir, name):
    y, sr = sf.read(str(assets_dir / name), dtype="float32")
    return y, sr


def test_stereo_hard_panned_channels_preserved(tmp_path):
    left, right = _tone(440), _tone(880)
    src = _write_src(tmp_path, np.stack([left, right], axis=1))
    assets = tmp_path / "assets"
    contracts = build_channel_assets(src, assets, "src-x", "a" * 64)
    assert {c.role for c in contracts} == {"channel_0", "channel_1", "mono_mix",
                                           "mid", "side"}
    ch0, _ = _load(assets, "channel_0.wav")
    ch1, _ = _load(assets, "channel_1.wav")
    assert np.array_equal(ch0, left) and np.array_equal(ch1, right)


def test_mid_side_exact_math_and_no_matrix_clipping(tmp_path):
    left = _tone(440, amp=0.9)
    right = -left                                  # worst case for side
    src = _write_src(tmp_path, np.stack([left, right], axis=1))
    assets = tmp_path / "assets"
    build_channel_assets(src, assets, "src-x", "a" * 64)
    mid, _ = _load(assets, "mid.wav")
    side, _ = _load(assets, "side.wav")
    assert np.allclose(mid, (left + right) / 2, atol=1e-7)
    assert np.allclose(side, (left - right) / 2, atol=1e-7)
    assert np.max(np.abs(side)) <= np.max(np.abs(left)) + 1e-7


def test_mono_source_gets_no_fake_mid_side(tmp_path):
    src = _write_src(tmp_path, _tone(300))
    assets = tmp_path / "assets"
    contracts = build_channel_assets(src, assets, "src-x", "a" * 64)
    assert {c.role for c in contracts} == {"mono_mix"}
    assert not (assets / "mid.wav").exists()
    assert not (assets / "side.wav").exists()


def test_four_channel_source_preserves_every_channel(tmp_path):
    chans = [_tone(200 * (i + 1), amp=0.2) for i in range(4)]
    src = _write_src(tmp_path, np.stack(chans, axis=1))
    assets = tmp_path / "assets"
    contracts = build_channel_assets(src, assets, "src-x", "a" * 64)
    assert {c.role for c in contracts} == {"channel_0", "channel_1", "channel_2",
                                           "channel_3", "mono_mix"}
    mix, _ = _load(assets, "mono_mix.wav")
    assert np.allclose(mix, np.mean(np.stack(chans), axis=0), atol=1e-7)


def test_identical_stereo_channels_side_is_silence(tmp_path):
    ch = _tone(500)
    src = _write_src(tmp_path, np.stack([ch, ch], axis=1))
    assets = tmp_path / "assets"
    build_channel_assets(src, assets, "src-x", "a" * 64)
    side, _ = _load(assets, "side.wav")
    assert np.max(np.abs(side)) < 1e-7


def test_source_bytes_untouched_and_no_gain_flag(tmp_path):
    src = _write_src(tmp_path, np.stack([_tone(440), _tone(880)], axis=1))
    before = src.read_bytes()
    contracts = build_channel_assets(src, tmp_path / "assets", "src-x", "a" * 64)
    assert src.read_bytes() == before
    assert all(c.gain_adjusted is False for c in contracts)
    assert all(not Path(c.rel_path).is_absolute() for c in contracts)
    assert all(c.rel_path.startswith("assets/") for c in contracts)


def test_derived_hashes_reproducible(tmp_path):
    src = _write_src(tmp_path, np.stack([_tone(440), _tone(880)], axis=1))
    c1 = build_channel_assets(src, tmp_path / "a1", "src-x", "a" * 64)
    c2 = build_channel_assets(src, tmp_path / "a2", "src-x", "a" * 64)
    assert {c.role: c.sha256 for c in c1} == {c.role: c.sha256 for c in c2}


def test_block_size_does_not_change_output_bytes(tmp_path):
    """Streaming must be transparent: a different block size may not alter the
    artifact bytes, otherwise hashes would depend on tuning."""
    src = _write_src(tmp_path, np.stack([_tone(440, seconds=2.0),
                                         _tone(880, seconds=2.0)], axis=1))
    big = build_channel_assets(src, tmp_path / "big", "src-x", "a" * 64,
                               block_frames=1 << 20)
    small = build_channel_assets(src, tmp_path / "small", "src-x", "a" * 64,
                                 block_frames=1024)
    assert {c.role: c.sha256 for c in big} == {c.role: c.sha256 for c in small}


def test_no_whole_file_audio_load(tmp_path, monkeypatch):
    """Regression guard for the bounded-memory requirement: a whole-file
    `soundfile.read` on the decoded recording must never happen. Only
    `soundfile.blocks` (bounded) may touch the audio."""
    src = _write_src(tmp_path, np.stack([_tone(440, seconds=3.0),
                                         _tone(880, seconds=3.0)], axis=1))

    def forbidden_read(*a, **kw):
        raise AssertionError(
            "soundfile.read called: the channel factory must stream blocks, "
            "not materialize the whole recording")

    seen_blocksizes = []
    real_blocks = sf.blocks

    def spy_blocks(*a, **kw):
        seen_blocksizes.append(kw.get("blocksize"))
        return real_blocks(*a, **kw)

    monkeypatch.setattr(sf, "read", forbidden_read)
    monkeypatch.setattr(sf, "blocks", spy_blocks)

    contracts = build_channel_assets(src, tmp_path / "assets", "src-x", "a" * 64)
    assert contracts                       # succeeded without any sf.read
    assert seen_blocksizes and all(b == BLOCK_FRAMES for b in seen_blocksizes)


def test_block_frames_bound_is_modest():
    # One second at 48 kHz: the memory bound must stay small and documented.
    assert BLOCK_FRAMES == 48_000
    assert channel_factory.BLOCK_FRAMES * 4 * 8 < 2_000_000   # < ~2 MB for 8ch
