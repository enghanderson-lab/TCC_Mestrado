"""Configuracao da aplicacao.

- MultiIndexConfig: tunables do pipeline multi-camera (motion detection,
  batching entre cameras, filtro de similaridade), carregados de YAML.
- Settings: caminhos e chaves da API, lidos de variaveis de ambiente.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

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


def _env_list(name: str) -> List[str]:
    return [item.strip() for item in os.getenv(name, "").split(",") if item.strip()]


@dataclass
class Settings:
    """Configuracao de execucao da API (ver API.md)."""
    api_key: str = "dev-key"
    db_path: Path = Path("cameras.db")
    index_root: Path = Path("index")
    upload_dir: Path = Path("uploads")
    cors_origins: List[str] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            api_key=os.getenv("VS_API_KEY", "dev-key"),
            db_path=Path(os.getenv("VS_DB_PATH", "cameras.db")),
            index_root=Path(os.getenv("VS_INDEX_DIR", "index")),
            upload_dir=Path(os.getenv("VS_UPLOAD_DIR", "uploads")),
            cors_origins=_env_list("VS_CORS_ORIGINS"),
        )
