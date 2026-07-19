"""Streaming hashing, atomic writes, duplicate detection, and media probing."""
import hashlib
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

np = pytest.importorskip("numpy")
pytest.importorskip("pydantic")

from pipeline.ingest.immutable import (  # noqa: E402
    DuplicateIndex,
    atomic_write_bytes,
    content_id,
    resolve_store_dir,
    sanitize_display_name,
    stream_hashes,
)


def test_stream_hashes_match_reference_on_multi_chunk_file(tmp_path):
    # >8 MiB forces multiple streaming chunks; reference = one-shot hashlib.
    data = os.urandom(9 * 1024 * 1024)
    p = tmp_path / "big.bin"
    p.write_bytes(data)
    s256, s512, size = stream_hashes(p)
    assert s256 == hashlib.sha256(data).hexdigest()
    assert s512 == hashlib.sha512(data).hexdigest()
    assert size == len(data)


def test_source_never_modified_by_hashing(tmp_path):
    p = tmp_path / "src.bin"
    p.write_bytes(b"evidence")
    before = (p.stat().st_mtime_ns, p.read_bytes())
    stream_hashes(p)
    assert (p.stat().st_mtime_ns, p.read_bytes()) == before


def test_content_id_deterministic():
    sha = "ab" * 32
    assert content_id(sha) == content_id(sha) == f"src-{sha[:16]}"


def test_atomic_write_and_posthash(tmp_path):
    dest = tmp_path / "out" / "x.bin"
    sha = atomic_write_bytes(dest, b"hello")
    assert dest.read_bytes() == b"hello"
    assert sha == hashlib.sha256(b"hello").hexdigest()
    assert not list(dest.parent.glob("*.part"))       # no temp litter


def test_atomic_write_interruption_leaves_no_final_file(tmp_path, monkeypatch):
    dest = tmp_path / "y.bin"

    def boom(src, dst):
        raise OSError("simulated crash before rename")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        atomic_write_bytes(dest, b"data")
    assert not dest.exists()                          # never half-written


def test_duplicate_same_bytes_different_path(tmp_path):
    idx = DuplicateIndex(tmp_path / "store")
    a = tmp_path / "a.wav"
    b = tmp_path / "sub" / "b.wav"
    b.parent.mkdir()
    a.write_bytes(b"identical-bytes")
    b.write_bytes(b"identical-bytes")
    sha_a, _, _ = stream_hashes(a)
    sha_b, _, _ = stream_hashes(b)
    assert sha_a == sha_b
    assert idx.check_and_register(sha_a, "src-1", "a.wav") is None
    assert idx.check_and_register(sha_b, "src-2", "b.wav") == "src-1"


def test_different_bytes_not_duplicates(tmp_path):
    idx = DuplicateIndex(tmp_path / "store")
    assert idx.check_and_register("f" * 64, "src-1", "x") is None
    assert idx.check_and_register("e" * 64, "src-2", "y") is None


def test_store_dir_default_outside_repo_and_config_override(tmp_path):
    default = resolve_store_dir(None)
    assert "onedrive" not in str(default).lower()
    override = resolve_store_dir({"storage": {"artifact_dir": str(tmp_path)}})
    assert override == tmp_path


def test_sanitize_display_name():
    assert sanitize_display_name(r"C:\Users\someone\secret\clip.wav") == "clip.wav"
    assert sanitize_display_name("a/b/c\x00evil.wav") == "c_evil.wav"


# ---------------------------------------------------------------- media probe
sf = pytest.importorskip("soundfile")
pytest.importorskip("av")

from pipeline.ingest.media_probe import probe  # noqa: E402


def _wav(tmp_path, name, channels, sr=16000, seconds=0.5):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    x = 0.2 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    data = np.stack([x] * channels, axis=1) if channels > 1 else x
    p = tmp_path / name
    sf.write(str(p), data, sr)
    return p


def test_probe_mono_wav(tmp_path):
    m = probe(_wav(tmp_path, "mono.wav", 1))
    assert m.container_format and "wav" in m.container_format
    assert m.channels == 1 and m.sample_rate == 16000
    assert m.n_audio_streams == 1 and m.has_video is False
    assert m.duration_sec == pytest.approx(0.5, abs=0.05)


def test_probe_stereo_and_multichannel(tmp_path):
    assert probe(_wav(tmp_path, "st.wav", 2)).channels == 2
    assert probe(_wav(tmp_path, "quad.wav", 4)).channels == 4


def test_probe_malformed_file_reports_reason_not_guess(tmp_path):
    p = tmp_path / "junk.wav"
    p.write_bytes(os.urandom(512))
    m = probe(p)
    assert m.warnings, "malformed input must produce a warning"
    assert m.sample_rate is None            # never invented
    assert "container_format" in m.unavailable_reasons


def test_probe_never_modifies_source(tmp_path):
    p = _wav(tmp_path, "keep.wav", 2)
    before = p.read_bytes()
    probe(p)
    assert p.read_bytes() == before
