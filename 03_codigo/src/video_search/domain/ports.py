"""Portas (interfaces) do núcleo da aplicação.

Os serviços em `application/` dependem apenas destes Protocols; as
implementações concretas (SigLIP, Qwen, FAISS, OpenCV, SQLite...) vivem em
`adapters/` e são ligadas em `bootstrap.py`. Trocar um modelo ou um índice
significa escrever um novo adapter que satisfaça o Protocol correspondente,
sem tocar nos serviços — e os testes usam fakes no lugar dos modelos.

`profiler` (Optional[ProfilingContext]) é aceito pelas operações caras para
registrar tempos por etapa; None = sem medição.
"""

from typing import Callable, Iterator, List, Optional, Protocol, Sequence, Tuple, TypeVar

import numpy as np
from PIL.Image import Image

from .entities import Camera, Frame, FrameRecord, IndexJob

T = TypeVar("T")

# (score, record, embedding) — o embedding é necessário para o MMR.
Candidate = Tuple[float, FrameRecord, np.ndarray]


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------

class ImageEmbedder(Protocol):
    def encode_images(self, images: Sequence[Image], profiler=None) -> np.ndarray:
        """(N, dim) L2-normalizado."""


class TextEmbedder(Protocol):
    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        """(N, dim) L2-normalizado, no mesmo espaço do ImageEmbedder."""


class Captioner(Protocol):
    def describe_for_query_batch(
        self, images: Sequence[Image], query: str, profiler=None
    ) -> List[str]:
        """Uma legenda curta por imagem, condicionada à query."""


class ConfidenceScorer(Protocol):
    def score(self, query: str, captions: Sequence[str]) -> List[float]:
        """Confiança de cada legenda corresponder à query."""


# ---------------------------------------------------------------------------
# Execução na GPU
# ---------------------------------------------------------------------------

class GpuRunner(Protocol):
    SEARCH: int
    INDEXING: int

    def run(self, fn: Callable[..., T], *args, priority: int, **kwargs) -> T:
        """Executa `fn` de forma serializada com as demais chamadas de GPU;
        menor `priority` passa na frente."""


# ---------------------------------------------------------------------------
# Armazenamento
# ---------------------------------------------------------------------------

class VectorIndex(Protocol):
    model_name: str

    def add(self, embedding: np.ndarray, record: FrameRecord) -> None: ...

    def save(self, path: str) -> None: ...

    def search_with_embeddings(self, query_embedding: np.ndarray, top_k: int) -> List[Candidate]: ...

    def __len__(self) -> int: ...


class IndexRepository(Protocol):
    def new(self) -> VectorIndex:
        """Índice vazio, pronto para receber `add()`."""

    def load(self, location: str) -> VectorIndex:
        """Índice persistido em `location`; pode reutilizar cópia em memória."""

    def delete(self, location: str) -> None: ...


class CameraRepository(Protocol):
    def upsert(self, camera: Camera) -> None: ...

    def get(self, camera_id: str) -> Optional[Camera]: ...

    def list_all(self) -> List[Camera]: ...

    def delete(self, camera_id: str) -> bool: ...


class JobRepository(Protocol):
    def save(self, job: IndexJob) -> None: ...

    def get(self, job_id: str) -> Optional[IndexJob]: ...

    def mark_interrupted(self) -> int:
        """Marca como erro os jobs que ficaram "running" de uma execução
        anterior do processo; retorna quantos foram marcados."""


# ---------------------------------------------------------------------------
# Mídia
# ---------------------------------------------------------------------------

class FrameSource(Protocol):
    def iter_frames(self, video_path: str, interval_sec: float) -> Iterator[Frame]: ...

    def frames_at(self, video_path: str, timestamps_sec: Sequence[float]) -> List[Image]: ...

    def estimate_samples(self, video_path: str, interval_sec: float) -> int: ...
