"""Índice legado em numpy: busca por força bruta (produto interno).

Formato de disco: embeddings.npy + records.json + metadata.json.
Usado como fallback quando FAISS não está instalado e para ler índices
gerados antes da adoção do FAISS. Viável até centenas de milhares de
frames; o formato padrão é o FAISS (ver faiss_index.py).
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from ...domain.entities import FrameRecord
from ...domain.ports import Candidate


class NumpyIndex:
    def __init__(self) -> None:
        self._embeddings: List[np.ndarray] = []
        self._records: List[FrameRecord] = []
        self._matrix: Optional[np.ndarray] = None  # cache de np.stack, invalidado em add()
        self.model_name: str = "siglip"

    def add(self, embedding: np.ndarray, record: FrameRecord) -> None:
        self._embeddings.append(np.asarray(embedding, dtype=np.float32))
        self._records.append(record)
        self._matrix = None

    def __len__(self) -> int:
        return len(self._records)

    def save(self, path: Union[str, Path], model_name: str = "siglip") -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        matrix = np.stack(self._embeddings) if self._embeddings else np.empty((0, 0), dtype=np.float32)
        np.save(path / "embeddings.npy", matrix)
        with open(path / "records.json", "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in self._records], f, ensure_ascii=False, indent=2)
        with open(path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump({"model_name": model_name}, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "NumpyIndex":
        path = Path(path)
        index = cls()
        index._matrix = np.load(path / "embeddings.npy").astype(np.float32)
        index._embeddings = list(index._matrix)
        with open(path / "records.json", encoding="utf-8") as f:
            index._records = [FrameRecord(**r) for r in json.load(f)]
        meta_path = path / "metadata.json"
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                index.model_name = json.load(f).get("model_name", "siglip")
        return index

    def search_with_embeddings(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Candidate]:
        if not self._embeddings:
            return []
        if self._matrix is None:
            self._matrix = np.stack(self._embeddings)
        scores = self._matrix @ np.asarray(query_embedding, dtype=np.float32)
        top_idx = np.argsort(-scores)[:top_k]
        return [(float(scores[i]), self._records[i], self._matrix[i]) for i in top_idx]
