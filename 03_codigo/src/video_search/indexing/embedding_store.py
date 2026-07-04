"""Armazenamento e busca de embeddings de frames.

Dois formatos suportados:
  - Legado (numpy): embeddings.npy + records.json + metadata.json
  - FAISS: faiss.index + records.json + metadata.json (ver faiss_store.py)

Use `load_store(path)` para carregar qualquer formato automaticamente.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np


@dataclass
class FrameRecord:
    index: int
    timestamp_sec: float
    video: str
    video_path: str = ""
    caption: str = ""


class EmbeddingStore:
    """Índice simples de embeddings normalizados com busca por similaridade de
    cosseno (produto interno), via força bruta com numpy.

    Viável para vídeos de até centenas de milhares de frames; para escalas
    maiores, considerar FAISS/Qdrant (ver 01_revisao_bibliografica)."""

    def __init__(self) -> None:
        self._embeddings: List[np.ndarray] = []
        self._records: List[FrameRecord] = []
        self.model_name: str = "siglip"

    def add(self, embedding: np.ndarray, record: FrameRecord) -> None:
        self._embeddings.append(np.asarray(embedding, dtype=np.float32))
        self._records.append(record)

    def __len__(self) -> int:
        return len(self._records)

    def save(self, path: Union[str, Path], model_name: str = "siglip") -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "embeddings.npy", np.stack(self._embeddings))
        with open(path / "records.json", "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in self._records], f, ensure_ascii=False, indent=2)
        with open(path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump({"model_name": model_name}, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "EmbeddingStore":
        path = Path(path)
        store = cls()
        embeddings = np.load(path / "embeddings.npy")
        with open(path / "records.json", encoding="utf-8") as f:
            records = json.load(f)
        store._embeddings = [row for row in embeddings]
        store._records = [FrameRecord(**r) for r in records]
        meta_path = path / "metadata.json"
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                store.model_name = json.load(f).get("model_name", "siglip")
        return store

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[float, FrameRecord]]:
        return [(s, r) for s, r, _ in self.search_with_embeddings(query_embedding, top_k)]

    def search_with_embeddings(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> List[Tuple[float, FrameRecord, np.ndarray]]:
        """Busca Top-K retornando (score, record, embedding) — necessário para MMR."""
        if not self._embeddings:
            return []
        matrix = np.stack(self._embeddings)
        q = np.asarray(query_embedding, dtype=np.float32)
        scores = matrix @ q
        top_idx = np.argsort(-scores)[:top_k]
        return [(float(scores[i]), self._records[i], self._embeddings[i]) for i in top_idx]


def load_store(path: Union[str, Path]) -> "EmbeddingStore":
    """Carrega o índice em `path` detectando o formato automaticamente.

    - Se `faiss.index` existe → usa FaissStore (busca FAISS)
    - Caso contrário → usa EmbeddingStore legado (numpy dot-product)

    Ambos expõem a mesma interface: search() e search_with_embeddings().
    """
    path = Path(path)
    if (path / "faiss.index").exists():
        from .faiss_store import FaissStore
        return FaissStore.load(path)  # type: ignore[return-value]
    return EmbeddingStore.load(path)
