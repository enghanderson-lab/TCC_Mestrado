"""Configuracao do pipeline multi-camera (motion detection, batching entre
cameras, filtro de similaridade de embeddings). Unico arquivo de tunables
para esses 3 estagios novos (ver `multi_index.py`)."""

from pathlib import Path
from typing import Optional, Union

import yaml
from pydantic import BaseModel


class MotionDetectionConfig(BaseModel):
    enabled: bool = True
    resize_width: int = 320
    resize_height: int = 180
    pixel_threshold: int = 25
    motion_ratio: float = 0.02


class BatchConfig(BaseModel):
    enabled: bool = True
    batch_size: int = 16
    timeout_ms: int = 200


class EmbeddingFilterConfig(BaseModel):
    enabled: bool = True
    similarity_threshold: float = 0.97


class MultiIndexConfig(BaseModel):
    motion_detection: MotionDetectionConfig = MotionDetectionConfig()
    batch: BatchConfig = BatchConfig()
    embedding_filter: EmbeddingFilterConfig = EmbeddingFilterConfig()

    @classmethod
    def load(cls, path: Optional[Union[str, Path]]) -> "MultiIndexConfig":
        """Carrega de um YAML; se `path` for None, retorna os defaults."""
        if path is None:
            return cls()
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)
