"""Logica de indexacao e busca, usada pela CLI.

Serializa o uso da GPU com um lock global: indexacao e busca carregam e
liberam o SigLIP/Qwen2.5-VL na VRAM (ver `vlm_describer.py`), e os dois
modelos grandes residentes ao mesmo tempo causam offload parcial para CPU
mesmo numa GPU de 8GB. Permitir indexacao e busca concorrentes duplicaria
esse risco, por isso cada chamada a `run_index`/`run_search` adquire o
mesmo lock antes de tocar a GPU.
"""

import threading
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from PIL.Image import Image

from .embedding_store import EmbeddingStore, FrameRecord
from .frame_extractor import extract_frames, extract_frames_at
from .siglip_embedder import SigLIPEmbedder

CONFIDENCE_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

_gpu_lock = threading.Lock()


def _make_embedder() -> SigLIPEmbedder:
    return SigLIPEmbedder()


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
    interval: float = 2.0,
    batch_size: int = 16,
    limit: int = 0,
    on_progress: Optional[Callable[[int], None]] = None,
) -> IndexSummary:
    """Extrai frames do video, gera embeddings (SigLIP) e salva o indice em
    `output_dir`. `on_progress(frames_processados)` e chamado apos cada
    lote, se informado."""
    with _gpu_lock:
        embedder = _make_embedder()
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

        store.save(output_dir)
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
    `with_caption` e verdadeiro, gera legenda condicionada a query
    (Qwen2.5-VL) e confianca (similaridade texto-texto) para cada resultado
    -- legenda e confianca sao calculadas em LOTE (uma unica chamada de
    `generate()`/`encode()` para todos os resultados) em vez de uma chamada
    por resultado, para manter a busca rapida com top_k>1 (ver
    `Qwen2VLDescriber.describe_for_query_batch`)."""
    with _gpu_lock:
        store = EmbeddingStore.load(index_dir)

        embedder = _make_embedder()
        query_embedding = embedder.encode_text([query])[0]
        raw_results = store.search(query_embedding, top_k=top_k)

        # Libera a VRAM do embedder de retrieval antes de carregar o
        # Qwen2.5-VL (ver docstring do modulo).
        del embedder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        results = [
            SearchResult(
                frame_index=rec.index,
                timestamp_sec=rec.timestamp_sec,
                video=rec.video,
                retrieval_score=score,
            )
            for score, rec in raw_results
        ]

        captionable = [i for i, (_, rec) in enumerate(raw_results) if rec.video_path]
        if not with_caption or not captionable:
            return results

        # Agrupa por video_path (normalmente um unico video por indice) para
        # extrair todos os frames necessarios com o video aberto uma unica
        # vez (ver `extract_frames_at`).
        by_video: Dict[str, List[int]] = defaultdict(list)
        for i in captionable:
            by_video[raw_results[i][1].video_path].append(i)

        images: Dict[int, Image] = {}
        for video_path, indices in by_video.items():
            timestamps = [raw_results[i][1].timestamp_sec for i in indices]
            for idx, frame in zip(indices, extract_frames_at(video_path, timestamps)):
                images[idx] = frame

        from sentence_transformers import SentenceTransformer

        from .hf_utils import load_offline_first
        from .vlm_describer import Qwen2VLDescriber

        describer = Qwen2VLDescriber()
        captions = describer.describe_for_query_batch([images[i] for i in captionable], query)

        text_model = load_offline_first(SentenceTransformer, CONFIDENCE_MODEL, device="cpu")
        embeddings = text_model.encode([query] + captions, normalize_embeddings=True)
        query_emb, caption_embs = embeddings[0], embeddings[1:]

        for i, caption, caption_emb in zip(captionable, captions, caption_embs):
            results[i].caption = caption
            results[i].confidence = float(np.dot(query_emb, caption_emb))

        del describer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results
