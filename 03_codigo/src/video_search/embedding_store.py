"""Armazenamento e busca de embeddings de frames (índice em memória, numpy)."""

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
        self.model_name: str = "clip"

    def add(self, embedding: np.ndarray, record: FrameRecord) -> None:
        self._embeddings.append(np.asarray(embedding, dtype=np.float32))
        self._records.append(record)

    def __len__(self) -> int:
        return len(self._records)

    def save(self, path: Union[str, Path], model_name: str = "clip") -> None:
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
                store.model_name = json.load(f).get("model_name", "clip")
        return store

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[float, FrameRecord]]:
        if not self._embeddings:
            return []
        matrix = np.stack(self._embeddings)
        scores = matrix @ np.asarray(query_embedding, dtype=np.float32)
        top_idx = np.argsort(-scores)[:top_k]
        return [(float(scores[i]), self._records[i]) for i in top_idx]
