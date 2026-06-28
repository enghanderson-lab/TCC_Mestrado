import numpy as np

from video_search.config import EmbeddingFilterConfig
from video_search.embedding_filter import EmbeddingSimilarityFilter


def test_first_embedding_always_accepted():
    embedding_filter = EmbeddingSimilarityFilter(EmbeddingFilterConfig())
    assert embedding_filter.should_accept("cam1", np.array([1.0, 0.0, 0.0])) is True


def test_identical_embedding_rejected():
    embedding_filter = EmbeddingSimilarityFilter(EmbeddingFilterConfig(similarity_threshold=0.97))
    embedding = np.array([1.0, 0.0, 0.0])
    assert embedding_filter.should_accept("cam1", embedding) is True
    assert embedding_filter.should_accept("cam1", embedding) is False


def test_orthogonal_embedding_accepted():
    embedding_filter = EmbeddingSimilarityFilter(EmbeddingFilterConfig(similarity_threshold=0.97))
    assert embedding_filter.should_accept("cam1", np.array([1.0, 0.0, 0.0])) is True
    assert embedding_filter.should_accept("cam1", np.array([0.0, 1.0, 0.0])) is True


def test_per_camera_state_is_independent():
    embedding_filter = EmbeddingSimilarityFilter(EmbeddingFilterConfig(similarity_threshold=0.97))
    embedding = np.array([1.0, 0.0, 0.0])
    assert embedding_filter.should_accept("cam1", embedding) is True
    assert embedding_filter.should_accept("cam2", embedding) is True


def test_disabled_always_accepts():
    embedding_filter = EmbeddingSimilarityFilter(EmbeddingFilterConfig(enabled=False))
    embedding = np.array([1.0, 0.0, 0.0])
    assert embedding_filter.should_accept("cam1", embedding) is True
    assert embedding_filter.should_accept("cam1", embedding) is True
