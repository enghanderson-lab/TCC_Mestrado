from video_search.config import MultiIndexConfig


def test_load_none_returns_defaults():
    config = MultiIndexConfig.load(None)
    assert config.motion_detection.enabled is True
    assert config.batch.batch_size == 16
    assert config.embedding_filter.similarity_threshold == 0.97


def test_load_yaml_overrides_defaults(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        """
motion_detection:
  enabled: false
  motion_ratio: 0.05
batch:
  batch_size: 4
  timeout_ms: 50
embedding_filter:
  similarity_threshold: 0.9
""",
        encoding="utf-8",
    )

    config = MultiIndexConfig.load(yaml_path)

    assert config.motion_detection.enabled is False
    assert config.motion_detection.motion_ratio == 0.05
    assert config.batch.batch_size == 4
    assert config.batch.timeout_ms == 50
    assert config.embedding_filter.similarity_threshold == 0.9
    # Campos nao sobrescritos mantem o default.
    assert config.motion_detection.resize_width == 320


def test_load_yaml_missing_sections_uses_defaults(tmp_path):
    yaml_path = tmp_path / "partial.yaml"
    yaml_path.write_text("batch:\n  batch_size: 8\n", encoding="utf-8")

    config = MultiIndexConfig.load(yaml_path)

    assert config.batch.batch_size == 8
    assert config.motion_detection.enabled is True
    assert config.embedding_filter.similarity_threshold == 0.97
