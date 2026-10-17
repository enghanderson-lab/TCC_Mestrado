"""Caso de uso: busca semântica em uma ou mais câmeras indexadas.

Dois modos:
    FAST     — SigLIP → Top-K direto do índice vetorial (sem VLM)
    DETAILED — SigLIP → Top-N candidatos → MMR → Top-K → legenda (Qwen)
               + confiança (similaridade texto-texto query × legenda)

O VLM processa apenas os top_k frames selecionados pelo MMR, reduzindo
drasticamente o número de inferências em relação à busca ingênua.

Único ponto de busca do sistema: a CLI e a API são apenas entradas
diferentes para `SearchService.search()`.
"""

import logging
import threading
import time
from collections import OrderedDict, defaultdict
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..domain.entities import Camera, FrameRecord, SearchResult
from ..domain.mmr import MmrReranker
from ..domain.ports import (
    Captioner, ConfidenceScorer, FrameSource, GpuRunner, IndexRepository, TextEmbedder,
)

logger = logging.getLogger(__name__)


class SearchMode(str, Enum):
    FAST = "fast"
    DETAILED = "detailed"


# (score, record, embedding, camera_id)
_Hit = Tuple[float, FrameRecord, np.ndarray, str]


class SearchService:
    def __init__(
        self,
        text_embedder: TextEmbedder,
        captioner: Captioner,
        scorer: ConfidenceScorer,
        indexes: IndexRepository,
        frames: FrameSource,
        gpu: GpuRunner,
        query_cache_size: int = 256,
    ) -> None:
        self._text_embedder = text_embedder
        self._captioner = captioner
        self._scorer = scorer
        self._indexes = indexes
        self._frames = frames
        self._gpu = gpu
        # LRU de embeddings de query: evita re-codificar a mesma consulta.
        self._query_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._query_cache_size = query_cache_size
        self._query_cache_lock = threading.Lock()

    def search(
        self,
        query: str,
        cameras: Sequence[Camera],
        mode: SearchMode = SearchMode.FAST,
        top_k: int = 5,
        retrieval_k: Optional[int] = None,
        mmr_lambda: float = 0.7,
        profiler=None,           # Optional[ProfilingContext]
    ) -> List[SearchResult]:
        """Busca `query` em todas as `cameras`, com ranking global.

        Args:
            retrieval_k: candidatos do índice antes do MMR (modo DETAILED);
                         None = max(4·top_k, 20).
            mmr_lambda:  1.0 = relevância pura, 0.0 = diversidade pura.
        """
        mode = SearchMode(mode)
        p = profiler
        if mode == SearchMode.FAST:
            retrieval_k = top_k
        elif retrieval_k is None:
            retrieval_k = max(top_k * 4, 20)

        query_emb = self._encode_query(query, p)
        hits = self._retrieve(query_emb, cameras, retrieval_k, p)

        if mode == SearchMode.DETAILED and hits:
            t0 = time.perf_counter()
            by_record = {id(h[1]): h for h in hits}
            reranked = MmrReranker(mmr_lambda).rerank(
                query_emb, [(s, r, e) for s, r, e, _ in hits], k=top_k
            )
            hits = [by_record[id(rec)] for _, rec in reranked]
            if p:
                p.add_time("mmr_rerank", time.perf_counter() - t0, len(hits))
            scored = self._caption_and_score(query, [h[1] for h in hits], p)
        else:
            hits = hits[:top_k]
            scored = {}

        return [
            SearchResult(
                camera_id=camera_id,
                frame_index=rec.index,
                timestamp_sec=rec.timestamp_sec,
                video=rec.video,
                video_path=rec.video_path,
                score=float(score),
                caption=scored[i][0] if i in scored else None,
                confidence=scored[i][1] if i in scored else None,
            )
            for i, (score, rec, _, camera_id) in enumerate(hits)
        ]

    # ------------------------------------------------------------------

    def _encode_query(self, query: str, p) -> np.ndarray:
        with self._query_cache_lock:
            if query in self._query_cache:
                self._query_cache.move_to_end(query)
                return self._query_cache[query]

        t0 = time.perf_counter()
        emb = self._gpu.run(self._text_embedder.encode_text, [query], priority=self._gpu.SEARCH)[0]
        if p:
            p.add_time("siglip_text_encode", time.perf_counter() - t0, 1)

        with self._query_cache_lock:
            self._query_cache[query] = emb
            if len(self._query_cache) > self._query_cache_size:
                self._query_cache.popitem(last=False)
        return emb

    def _retrieve(self, query_emb: np.ndarray, cameras: Sequence[Camera], k: int, p) -> List[_Hit]:
        """Top-`k` global entre as câmeras; câmeras cujo índice não pode ser
        lido (ex.: removido do disco) são ignoradas."""
        hits: List[_Hit] = []
        for cam in cameras:
            try:
                t0 = time.perf_counter()
                index = self._indexes.load(cam.output_dir)
                if p:
                    p.add_time("load_store", time.perf_counter() - t0)
                t0 = time.perf_counter()
                found = index.search_with_embeddings(query_emb, top_k=k)
                if p:
                    p.add_time("faiss_search", time.perf_counter() - t0, len(found))
            except Exception as exc:
                logger.warning("indice da camera %s indisponivel: %s", cam.camera_id, exc)
                continue
            hits.extend((score, rec, emb, cam.camera_id) for score, rec, emb in found)
        hits.sort(key=lambda h: h[0], reverse=True)
        return hits[:k]

    def _caption_and_score(
        self, query: str, records: Sequence[FrameRecord], p
    ) -> Dict[int, Tuple[str, float]]:
        """{posição em records: (legenda, confiança)} para os frames cujo
        vídeo pode ser lido."""
        # Agrupa por vídeo para abrir cada arquivo uma única vez
        by_video: Dict[str, List[int]] = defaultdict(list)
        for i, rec in enumerate(records):
            if rec.video_path:
                by_video[rec.video_path].append(i)

        t0 = time.perf_counter()
        images = {}
        for video_path, indices in by_video.items():
            try:
                frames = self._frames.frames_at(video_path, [records[i].timestamp_sec for i in indices])
            except (FileNotFoundError, ValueError) as exc:
                logger.warning("frames de %s indisponiveis: %s", video_path, exc)
                continue
            images.update(zip(indices, frames))
        if p:
            p.add_time("frame_extract_search", time.perf_counter() - t0, len(images))
        if not images:
            return {}

        captionable = sorted(images)
        captions = self._gpu.run(
            self._captioner.describe_for_query_batch,
            [images[i] for i in captionable], query,
            profiler=p, priority=self._gpu.SEARCH,
        )

        t0 = time.perf_counter()
        confidences = self._scorer.score(query, captions)
        if p:
            p.add_time("confidence_score", time.perf_counter() - t0, len(captions))

        return {i: (c, conf) for i, c, conf in zip(captionable, captions, confidences)}
