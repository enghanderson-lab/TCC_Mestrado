"""Lógica de indexação e busca semântica em vídeo.

Dois modos de busca:
    FAST     — SigLIP → Top-K imediato (sem VLM, sem reranking)
    DETAILED — SigLIP → Top-N candidatos → MMR Reranker → Top-K → Qwen

O Qwen processa apenas os top_k frames selecionados pelo reranker, reduzindo
drasticamente o número de inferências VLM em relação à busca ingênua.

GPU serializada via `_gpu_lock`: SigLIP e Qwen nunca são chamados
concorrentemente pelo mesmo processo.
"""

import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
from PIL.Image import Image

from .embedding_store import load_store
from ..media.frame_extractor import extract_frames_at

CONFIDENCE_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

_gpu_lock = threading.Lock()


class SearchMode(str, Enum):
    """Modo de busca semântica.

    FAST:     SigLIP → resultado imediato. Sem VLM, latência mínima.
    DETAILED: SigLIP → MMR reranker → Qwen. Máxima qualidade semântica,
              maior latência.
    """
    FAST = "fast"
    DETAILED = "detailed"


def _get_manager(model_manager=None):
    if model_manager is not None:
        return model_manager
    from ..models.model_manager import get_model_manager
    return get_model_manager()


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


# ---------------------------------------------------------------------------
# Indexação
# ---------------------------------------------------------------------------

def run_index(
    video_path: str,
    output_dir: str,
    interval: float = 2.0,
    batch_size: int = 16,
    limit: int = 0,
    on_progress: Optional[Callable[[int], None]] = None,
    model_manager=None,
    embedding_cache=None,
    profiler=None,           # Optional[ProfilingContext]
    profiling_dir: Optional[str] = None,
) -> IndexSummary:
    """Extrai frames, gera embeddings SigLIP e salva índice FAISS.

    `profiler`: contexto de profiling pré-criado (usa-se quando o caller já
    criou um profiler e quer agregar medições). Quando None e `profiling_dir`
    é fornecido, cria um novo contexto internamente e gera o relatório ao final.
    """
    from .async_pipeline import AsyncIndexPipeline, PipelineConfig, PipelineSource
    from ..utils.config import EmbeddingFilterConfig, MotionDetectionConfig
    from ..profiling import ProfilingContext, HardwareMonitor, generate_report

    mm = _get_manager(model_manager)

    own_profiler = profiler is None and profiling_dir is not None
    if profiler is None:
        profiler = ProfilingContext()

    hw_monitor = HardwareMonitor().start() if profiling_dir is not None else None

    progress_cb = None
    if on_progress is not None:
        def progress_cb(_camera_id: str, count: int) -> None:
            on_progress(count)

    source = PipelineSource(
        camera_id="default",
        video_path=video_path,
        output_dir=output_dir,
    )
    pipeline = AsyncIndexPipeline(
        embedder=mm.get_siglip(),
        config=PipelineConfig(batch_size=batch_size),
        motion_config=MotionDetectionConfig(enabled=False),
        embedding_config=EmbeddingFilterConfig(enabled=False),
        gpu_lock=_gpu_lock,
        embedding_cache=embedding_cache,
        profiler=profiler,
    )
    results = pipeline.run([source], interval_sec=interval, limit=limit, on_progress=progress_cb)

    if hw_monitor is not None:
        hw_monitor.stop()

    if own_profiler and profiling_dir is not None:
        hw = hw_monitor.summary() if hw_monitor else None
        generate_report(profiler, label="indexing", hardware=hw, profiling_dir=profiling_dir)

    return IndexSummary(output_dir=str(output_dir), frame_count=results[0].frames_accepted)


# ---------------------------------------------------------------------------
# Busca
# ---------------------------------------------------------------------------

def run_search(
    query: str,
    index_dir: str,
    top_k: int = 5,
    mode: SearchMode = SearchMode.DETAILED,
    retrieval_k: int = 100,
    reranker_lambda: float = 0.7,
    model_manager=None,
    profiler=None,           # Optional[ProfilingContext]
    profiling_dir: Optional[str] = None,
    # Backward compat: with_caption=False → SearchMode.FAST
    with_caption: Optional[bool] = None,
) -> List[SearchResult]:
    """Busca semântica em frames indexados.

    Args:
        query:           descrição em linguagem natural.
        index_dir:       pasta do índice (FAISS ou numpy legado).
        top_k:           nº de resultados retornados ao usuário.
        mode:            FAST (só SigLIP) ou DETAILED (SigLIP + MMR + Qwen).
        retrieval_k:     nº de candidatos FAISS antes do reranker (modo DETAILED).
        reranker_lambda: parâmetro MMR λ (1.0=relevância pura, 0.0=diversidade pura).
        model_manager:   instância de ModelManager; None usa o singleton.
        profiler:        contexto de profiling pré-criado pelo caller.
        profiling_dir:   pasta para salvar relatório (None = só log).
        with_caption:    compat. legada — False → FAST, True → DETAILED.
    """
    from ..profiling import ProfilingContext, HardwareMonitor, generate_report

    # Compat legada
    if with_caption is not None:
        mode = SearchMode.DETAILED if with_caption else SearchMode.FAST

    mm = _get_manager(model_manager)

    own_profiler = profiler is None
    if profiler is None:
        profiler = ProfilingContext()

    hw_monitor = HardwareMonitor().start() if profiling_dir is not None else None
    p = profiler

    with _gpu_lock:
        t0 = time.perf_counter()
        store = load_store(index_dir)
        p.add_time("load_store", time.perf_counter() - t0)

        embedder = mm.get_siglip()
        t0 = time.perf_counter()
        query_embedding = embedder.encode_text([query])[0]
        p.add_time("siglip_text_encode", time.perf_counter() - t0, 1)

        if mode == SearchMode.FAST:
            t0 = time.perf_counter()
            results = _search_fast(query_embedding, store, top_k)
            p.add_time("faiss_search", time.perf_counter() - t0, top_k)
        else:
            results = _search_detailed(
                query, query_embedding, store, top_k, retrieval_k,
                reranker_lambda, mm, p,
            )

    if hw_monitor is not None:
        hw_monitor.stop()

    if own_profiler and profiling_dir is not None:
        p.count("frames_indexed", len(results))
        hw = hw_monitor.summary() if hw_monitor else None
        generate_report(p, label="search", hardware=hw, profiling_dir=profiling_dir)

    return results


def _search_fast(
    query_embedding: np.ndarray,
    store,
    top_k: int,
) -> List[SearchResult]:
    """Modo rápido: Top-K direto do índice vetorial, sem VLM."""
    raw = store.search(query_embedding, top_k=top_k)
    return [
        SearchResult(
            frame_index=rec.index,
            timestamp_sec=rec.timestamp_sec,
            video=rec.video,
            retrieval_score=score,
        )
        for score, rec in raw
    ]


def _search_detailed(
    query: str,
    query_embedding: np.ndarray,
    store,
    top_k: int,
    retrieval_k: int,
    reranker_lambda: float,
    mm,
    profiler,
) -> List[SearchResult]:
    """Modo detalhado: FAISS(retrieval_k) → MMR(top_k) → Qwen."""
    from .reranker import MmrReranker, RerankerConfig
    p = profiler

    # 1. Retrieval
    t0 = time.perf_counter()
    candidates = store.search_with_embeddings(query_embedding, top_k=retrieval_k)
    p.add_time("faiss_search", time.perf_counter() - t0, len(candidates))

    if not candidates:
        return []

    # 2. MMR reranking
    t0 = time.perf_counter()
    reranker = MmrReranker(RerankerConfig(lambda_=reranker_lambda))
    reranked = reranker.rerank(query_embedding, candidates, k=top_k)
    p.add_time("mmr_rerank", time.perf_counter() - t0, len(reranked))

    results = [
        SearchResult(
            frame_index=rec.index,
            timestamp_sec=rec.timestamp_sec,
            video=rec.video,
            retrieval_score=score,
        )
        for score, rec in reranked
    ]

    # 3. Qwen apenas em frames com video_path
    captionable = [
        i for i, (_, rec) in enumerate(reranked) if rec.video_path
    ]
    if not captionable:
        return results

    # Agrupa por vídeo para abrir cada arquivo uma única vez
    by_video: Dict[str, List[int]] = defaultdict(list)
    for i in captionable:
        by_video[reranked[i][1].video_path].append(i)

    t0 = time.perf_counter()
    images: Dict[int, Image] = {}
    for video_path, indices in by_video.items():
        timestamps = [reranked[i][1].timestamp_sec for i in indices]
        for idx, frame in zip(indices, extract_frames_at(video_path, timestamps)):
            images[idx] = frame
    p.add_time("frame_extract_search", time.perf_counter() - t0, len(images))

    # 4. Qwen em lote
    describer = mm.get_qwen()
    captions = describer.describe_for_query_batch(
        [images[i] for i in captionable], query, profiler=p
    )

    # 5. Confiança: similaridade texto-texto query × legenda
    t0 = time.perf_counter()
    text_model = mm.get_sentence_transformer()
    embeddings = text_model.encode([query] + captions, normalize_embeddings=True)
    query_emb, caption_embs = embeddings[0], embeddings[1:]
    p.add_time("confidence_score", time.perf_counter() - t0, len(captions))

    for i, caption, caption_emb in zip(captionable, captions, caption_embs):
        results[i].caption = caption
        results[i].confidence = float(np.dot(query_emb, caption_emb))

    return results
