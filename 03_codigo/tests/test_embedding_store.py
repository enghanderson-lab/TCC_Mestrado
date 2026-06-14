import numpy as np

from video_search.embedding_store import EmbeddingStore, FrameRecord


def test_save_load_search(tmp_path):
    store = EmbeddingStore()
    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(5, 8)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    for i, vec in enumerate(vectors):
        store.add(vec, FrameRecord(index=i, timestamp_sec=i * 2.0, video="test.mp4"))

    index_path = tmp_path / "index"
    store.save(index_path)
    loaded = EmbeddingStore.load(index_path)

    assert len(loaded) == 5

    results = loaded.search(vectors[2], top_k=1)
    assert results[0][1].index == 2
    assert results[0][0] > 0.99
