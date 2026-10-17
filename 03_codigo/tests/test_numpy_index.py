import numpy as np

from video_search.adapters.storage.numpy_index import NumpyIndex
from video_search.domain.entities import FrameRecord


def test_save_load_search(tmp_path):
    index = NumpyIndex()
    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(5, 8)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    for i, vec in enumerate(vectors):
        index.add(vec, FrameRecord(index=i, timestamp_sec=i * 2.0, video="test.mp4"))

    index.save(tmp_path / "index")
    loaded = NumpyIndex.load(tmp_path / "index")

    assert len(loaded) == 5
    score, rec, emb = loaded.search_with_embeddings(vectors[2], top_k=1)[0]
    assert rec.index == 2
    assert score > 0.99
    np.testing.assert_allclose(emb, vectors[2], atol=1e-6)


def test_empty_index_roundtrip(tmp_path):
    NumpyIndex().save(tmp_path / "empty")
    loaded = NumpyIndex.load(tmp_path / "empty")

    assert len(loaded) == 0
    assert loaded.search_with_embeddings(np.ones(2, dtype=np.float32), top_k=3) == []
