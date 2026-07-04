"""Índice vetorial FAISS com persistência, atualização incremental e suporte GPU.

Substitui a busca linear do EmbeddingStore (O(N) numpy dot) por busca FAISS
(O(N) SIMD para IndexFlatIP). Para índices de vídeo típicos (< 500 K frames,
768 dim) a diferença de velocidade é mensurável no nível de milissegundos.

Inicialização lazy: o índice FAISS é criado no primeiro `add()`, usando a
dimensão real do primeiro vetor. Isso evita que o caller precise conhecer a
dimensão de antemão (útil em testes com embedders falsos de dim != 768).

Formato de disco:
    <dir>/faiss.index    — índice FAISS serializado
    <dir>/records.json   — metadados por frame (FrameRecord)
    <dir>/metadata.json  — model_name, store_type, dim
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

try:
    import faiss as _faiss
    FAISS_AVAILABLE = True
except ImportError:
    _faiss = None  # type: ignore[assignment]
    FAISS_AVAILABLE = False

from .embedding_store import FrameRecord


def _require_faiss() -> None:
    if not FAISS_AVAILABLE:
        raise ImportError(
            "FAISS não instalado. Instale com:\n"
            "  pip install faiss-cpu          # CPU\n"
            "  pip install faiss-gpu          # GPU (requer CUDA)"
        )


def _to_gpu(cpu_index, device_id: int = 0):
    """Tenta mover o índice para GPU; retorna o original se não for possível."""
    if not hasattr(_faiss, "StandardGpuResources"):
        return cpu_index
    try:
        import torch
        if not torch.cuda.is_available():
            return cpu_index
        res = _faiss.StandardGpuResources()
        return _faiss.index_cpu_to_gpu(res, device_id, cpu_index)
    except Exception:
        return cpu_index


class FaissStore:
    """Armazena embeddings L2-normalizados num índice FAISS IndexFlatIP.

    IndexFlatIP usa produto interno como métrica — equivalente à similaridade
    de cosseno para vetores normalizados. Suporta adição incremental sem
    re-treinamento.

    O índice FAISS é criado lazily no primeiro `add()`, usando a dimensão do
    primeiro vetor inserido. Isso permite que embedders com dim != 768
    (ex.: fakes de teste) funcionem sem configuração explícita.
    """

    def __init__(self, use_gpu: bool = True) -> None:
        _require_faiss()
        self._use_gpu = use_gpu
        self._dim: Optional[int] = None
        self._index = None          # criado no primeiro add()
        self._on_gpu = False
        self._records: List[FrameRecord] = []
        self.model_name: str = "siglip"

    def _ensure_index(self, dim: int) -> None:
        if self._index is not None:
            return
        self._dim = dim
        cpu_index = _faiss.IndexFlatIP(dim)
        self._index = _to_gpu(cpu_index) if self._use_gpu else cpu_index
        self._on_gpu = self._index is not cpu_index

    # ------------------------------------------------------------------
    # Escrita
    # ------------------------------------------------------------------

    def add(self, embedding: np.ndarray, record: FrameRecord) -> None:
        emb = np.asarray(embedding, dtype=np.float32).ravel()
        self._ensure_index(emb.shape[0])
        self._index.add(emb.reshape(1, self._dim))
        self._records.append(record)

    def save(self, path: Union[str, Path], model_name: str = "siglip") -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self._index is not None:
            cpu_index = _faiss.index_gpu_to_cpu(self._index) if self._on_gpu else self._index
            _faiss.write_index(cpu_index, str(path / "faiss.index"))
        else:
            # índice vazio: cria um placeholder com dim padrão
            _faiss.write_index(_faiss.IndexFlatIP(self._dim or 768), str(path / "faiss.index"))

        with open(path / "records.json", "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in self._records], f, ensure_ascii=False, indent=2)
        with open(path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump({
                "model_name": model_name,
                "store_type": "faiss",
                "dim": self._dim or 768,
            }, f)

    # ------------------------------------------------------------------
    # Leitura
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: Union[str, Path], use_gpu: bool = True) -> "FaissStore":
        _require_faiss()
        path = Path(path)

        with open(path / "metadata.json", encoding="utf-8") as f:
            meta = json.load(f)

        store = cls.__new__(cls)
        store._use_gpu = use_gpu
        store.model_name = meta.get("model_name", "siglip")

        cpu_index = _faiss.read_index(str(path / "faiss.index"))
        store._dim = cpu_index.d
        store._index = _to_gpu(cpu_index) if use_gpu else cpu_index
        store._on_gpu = store._index is not cpu_index

        with open(path / "records.json", encoding="utf-8") as f:
            store._records = [FrameRecord(**r) for r in json.load(f)]

        return store

    # ------------------------------------------------------------------
    # Busca
    # ------------------------------------------------------------------

    def search(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> List[Tuple[float, FrameRecord]]:
        """Busca Top-K por similaridade de cosseno. Compatível com EmbeddingStore."""
        return [(s, r) for s, r, _ in self.search_with_embeddings(query_embedding, top_k)]

    def search_with_embeddings(
        self, query_embedding: np.ndarray, top_k: int = 100
    ) -> List[Tuple[float, FrameRecord, np.ndarray]]:
        """Busca Top-K retornando (score, record, embedding).

        O embedding retornado é reconstruído do IndexFlatIP (sem overhead
        extra — o índice flat já armazena os vetores originais).
        Necessário para o MMR reranker calcular diversidade entre candidatos.
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        k = min(top_k, self._index.ntotal)
        vec = np.asarray(query_embedding, dtype=np.float32).reshape(1, self._dim)
        scores, indices = self._index.search(vec, k)

        cpu_index = _faiss.index_gpu_to_cpu(self._index) if self._on_gpu else self._index
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            emb = np.empty(self._dim, dtype=np.float32)
            cpu_index.reconstruct(int(idx), emb)
            results.append((float(score), self._records[int(idx)], emb))
        return results

    def __len__(self) -> int:
        return 0 if self._index is None else self._index.ntotal
