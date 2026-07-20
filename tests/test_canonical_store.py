"""Canonical artifact store: non-synced default, operator override, path-
traversal rejection, content-addressed reuse, and the rule that large Phase 1
artifacts never default into the repository output tree."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

np = pytest.importorskip("numpy")
pytest.importorskip("pydantic")

from pipeline.ingest.immutable import resolve_store_dir  # noqa: E402
from pipeline.ingest.ingest import canonical_dirs  # noqa: E402

SHA = "ab" * 32


def test_default_store_is_outside_a_onedrive_repository(monkeypatch, tmp_path):
    # Simulate the real situation: repo inside OneDrive, no configured store.
    fake_local = tmp_path / "AppData" / "Local"
    fake_local.mkdir(parents=True)
    monkeypatch.setenv("LOCALAPPDATA", str(fake_local))
    store = resolve_store_dir(None)
    assert "onedrive" not in str(store).lower()
    assert str(store).startswith(str(fake_local))


def test_operator_configured_store_is_honoured(tmp_path):
    assert resolve_store_dir({"storage": {"artifact_dir": str(tmp_path)}}) == tmp_path


def test_canonical_dir_is_content_addressed(tmp_path):
    absolute, rel = canonical_dirs(tmp_path, SHA)
    assert rel == f"sources/{SHA[:2]}/{SHA}"
    assert absolute == (tmp_path / rel).resolve()


def test_identical_content_resolves_to_same_canonical_dir(tmp_path):
    a, rel_a = canonical_dirs(tmp_path, SHA)
    b, rel_b = canonical_dirs(tmp_path, SHA)
    assert (a, rel_a) == (b, rel_b)


@pytest.mark.parametrize("evil", [
    "../../etc/passwd",
    "..\\..\\windows\\system32",
    "not-a-hash",
    "",
    "AB" * 32,          # uppercase is not our canonical lowercase hex
    "zz" * 32,
    "a" * 63,
])
def test_path_traversal_and_malformed_ids_rejected(tmp_path, evil):
    with pytest.raises(ValueError):
        canonical_dirs(tmp_path, evil)


# --------------------------------------------------------------- integration
sf = pytest.importorskip("soundfile")
pytest.importorskip("imageio_ffmpeg")

from pipeline.ingest.ingest import POINTER_NAME, ingest_file, load_pointer  # noqa: E402


def _fixture(tmp_path, name="src.wav", seconds=0.4):
    sr = 16000
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    data = np.stack([0.4 * np.sin(2 * np.pi * 440 * t),
                     0.3 * np.sin(2 * np.pi * 660 * t)], axis=1).astype(np.float32)
    p = tmp_path / name
    sf.write(str(p), data, sr, subtype="FLOAT")
    return p


@pytest.mark.heavy
def test_artifacts_go_to_store_not_output_tree(tmp_path):
    src = _fixture(tmp_path)
    out_dir = tmp_path / "outputs" / "src"
    out_dir.mkdir(parents=True)
    store = tmp_path / "store"
    m = ingest_file(src, out_dir, {"storage": {"artifact_dir": str(store)}})

    # Canonical artifacts live in the store...
    canon = store / m.store_relative_dir
    assert (canon / "ingest_manifest.json").exists()
    for d in m.derived:
        assert (canon / d.rel_path).exists()
    # ...and the output tree holds ONLY the small pointer (no wav assets).
    assert (out_dir / POINTER_NAME).exists()
    assert not (out_dir / "ingest").exists()
    assert not list(out_dir.glob("*.wav"))
    assert (out_dir / POINTER_NAME).stat().st_size < 2000


@pytest.mark.heavy
def test_pointer_is_portable_and_leaks_no_absolute_path(tmp_path):
    src = _fixture(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    store = tmp_path / "store"
    ingest_file(src, out_dir, {"storage": {"artifact_dir": str(store)}})
    raw = (out_dir / POINTER_NAME).read_text(encoding="utf-8")
    assert str(store) not in raw
    assert str(tmp_path) not in raw
    assert ":\\" not in raw and not raw.count("/Users/")
    ptr = load_pointer(out_dir)
    assert ptr is not None and ptr.store_relative_dir.startswith("sources/")
    assert not Path(ptr.manifest_relative_path).is_absolute()


@pytest.mark.heavy
def test_duplicate_content_reuses_canonical_dir(tmp_path):
    src = _fixture(tmp_path, "a.wav")
    dup = tmp_path / "b.wav"
    dup.write_bytes(src.read_bytes())
    store = tmp_path / "store"
    cfg = {"storage": {"artifact_dir": str(store)}}
    o1, o2 = tmp_path / "o1", tmp_path / "o2"
    o1.mkdir()
    o2.mkdir()
    m1 = ingest_file(src, o1, cfg)
    m2 = ingest_file(dup, o2, cfg)
    assert m1.store_relative_dir == m2.store_relative_dir     # same content dir
    assert m2.source.duplicate_of == m1.source.asset_id
    assert len(list((store / "sources").rglob("ingest_manifest.json"))) == 1
