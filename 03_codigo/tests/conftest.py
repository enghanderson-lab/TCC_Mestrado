"""Fakes das portas de modelo (domain/ports.py) e utilitários comuns.

Os serviços recebem estes fakes no lugar de SigLIP/Qwen/sentence-transformer,
o que permite testar o fluxo completo (indexação → índice FAISS → busca →
API) sem GPU e sem baixar modelos.
"""

import cv2
import numpy as np
import pytest

from video_search.adapters.gpu_executor import GpuExecutor


def write_video(path, colors, fps=5, size=(64, 64)):
    """Um frame por cor (BGR); retorna o caminho como str."""
    width, height = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for color in colors:
        writer.write(np.full((height, width, 3), color, dtype=np.uint8))
    writer.release()
    return str(path)


def brightness_embedding(mean: float) -> np.ndarray:
    """Embedding 2-D determinístico a partir do brilho médio (0-1): preto e
    branco ficam ortogonais, cinzas no meio — suficiente para o filtro de
    similaridade e para o ranking."""
    vec = np.array([mean, 1.0 - mean], dtype=np.float32)
    return vec / np.linalg.norm(vec)


class FakeImageEmbedder:
    def __init__(self):
        self.batches = []

    def encode_images(self, images, profiler=None):
        self.batches.append(len(images))
        return np.stack([
            brightness_embedding(float(np.asarray(img.convert("L"), dtype=np.float32).mean()) / 255.0)
            for img in images
        ])


class FakeTextEmbedder:
    """Query com "clara" → embedding de branco; qualquer outra → preto."""

    def __init__(self):
        self.calls = 0

    def encode_text(self, texts):
        self.calls += 1
        return np.stack([brightness_embedding(1.0 if "clara" in t else 0.0) for t in texts])


class FakeCaptioner:
    def __init__(self):
        self.calls = []

    def describe_for_query_batch(self, images, query, profiler=None):
        self.calls.append(len(images))
        return [
            "cena clara" if np.asarray(img.convert("L")).mean() > 127 else "cena escura"
            for img in images
        ]


class FakeScorer:
    def score(self, query, captions):
        return [1.0 if query.split()[-1] in c else 0.2 for c in captions]


@pytest.fixture
def gpu():
    executor = GpuExecutor()
    yield executor
    executor.shutdown()
