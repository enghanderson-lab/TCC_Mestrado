"""Logica de indexacao e busca, compartilhada entre a CLI e a API HTTP.

Serializa o uso da GPU com um lock global: indexacao e busca carregam e
liberam o SigLIP/Qwen2.5-VL na VRAM (ver `vlm_describer.py`), e os dois
modelos grandes residentes ao mesmo tempo causam offload parcial para CPU
mesmo numa GPU de 8GB. Permitir indexacao e busca concorrentes duplicaria
esse risco, por isso cada chamada a `run_index`/`run_search` adquire o
mesmo lock antes de tocar a GPU.
"""

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch

from .embedding_store import EmbeddingStore, FrameRecord
from .frame_extractor import extract_frame_at, extract_frames
from .siglip_embedder import SigLIPEmbedder

CONFIDENCE_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

_gpu_lock = threading.Lock()


@dataclass
class IndexSummary:
    output_dir: str
    frame_count: int
    model_name: str = "siglip"


@dataclass
class SearchResult:
    frame_index: int
    timestamp_sec: float
    video: str
    retrieval_score: float
    confidence: Optional[float] = None
    caption: Optional[str] = None


def run_index(
    video_path: str,
    output_dir: str,
    interval: float = 1.0,
    batch_size: int = 16,
    limit: int = 0,
    on_progress: Optional[Callable[[int], None]] = None,
) -> IndexSummary:
    """Extrai frames do video, gera embeddings SigLIP e salva o indice em
    `output_dir`. `on_progress(frames_processados)` e chamado apos cada lote,
    se informado (usado pelo job em background da API para reportar status)."""
    with _gpu_lock:
        embedder = SigLIPEmbedder()
        store = EmbeddingStore()
        video_name = Path(video_path).name
        resolved_video_path = str(Path(video_path).resolve())

        batch_images: List = []
        batch_records: List[FrameRecord] = []

        def flush() -> None:
            if not batch_images:
                return
            embeddings = embedder.encode_images(batch_images)
            for emb, rec in zip(embeddings, batch_records):
                store.add(emb, rec)
            batch_images.clear()
            batch_records.clear()
            if on_progress is not None:
                on_progress(len(store))

        for frame in extract_frames(video_path, interval_sec=interval):
            if limit and frame.index >= limit:
                break
            batch_images.append(frame.image)
            batch_records.append(
                FrameRecord(
                    index=frame.index,
                    timestamp_sec=frame.timestamp_sec,
                    video=video_name,
                    video_path=resolved_video_path,
                )
            )
            if len(batch_images) >= batch_size:
                flush()
        flush()

        store.save(output_dir, model_name="siglip")
        del embedder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return IndexSummary(output_dir=str(output_dir), frame_count=len(store))


def run_search(
    query: str,
    index_dir: str,
    top_k: int = 12,
    with_caption: bool = True,
) -> List[SearchResult]:
    """Busca os top-K frames do indice mais similares a `query`. Quando
    `with_caption` e verdadeiro, gera legenda condicionada a query (Qwen2.5-VL)
    e confianca (similaridade texto-texto) para cada resultado."""
    with _gpu_lock:
        store = EmbeddingStore.load(index_dir)

        embedder = SigLIPEmbedder()
        query_embedding = embedder.encode_text([query])[0]
        raw_results = store.search(query_embedding, top_k=top_k)

        # Libera a VRAM do embedder de retrieval antes de carregar o
        # Qwen2.5-VL (ver docstring do modulo).
        del embedder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        describer = None
        text_model = None
        if with_caption:
            from sentence_transformers import SentenceTransformer
            from .vlm_describer import Qwen2VLDescriber

            describer = Qwen2VLDescriber()
            text_model = SentenceTransformer(CONFIDENCE_MODEL, device="cpu")

        results: List[SearchResult] = []
        for score, rec in raw_results:
            result = SearchResult(
                frame_index=rec.index,
                timestamp_sec=rec.timestamp_sec,
                video=rec.video,
                retrieval_score=score,
            )
            if describer is not None and rec.video_path:
                image = extract_frame_at(rec.video_path, rec.timestamp_sec)
                caption = describer.describe_for_query(image, query)
                text_emb = text_model.encode([query, caption], normalize_embeddings=True)
                result.confidence = float(np.dot(text_emb[0], text_emb[1]))
                result.caption = caption
            results.append(result)

        return results
