"""Caso de uso: busca semântica em uma ou mais câmeras indexadas.

Dois modos:
    FAST     — SigLIP → Top-K direto do índice vetorial (sem VLM)
    DETAILED — SigLIP → Top-N candidatos → MMR → Top-K → recorte ampliado
               (SigLIP escolhe a região) → legenda (Qwen, frame + recorte)
               → confiança (NLI: a legenda implica a query?)

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
    Captioner, ConfidenceScorer, FrameSource, GpuRunner, ImageEmbedder, IndexRepository, TextEmbedder,
)
from ..domain.tiling import ZoomConfig, tile_boxes

logger = logging.getLogger(__name__)


class SearchMode(str, Enum):
    FAST = "fast"
    DETAILED = "detailed"


def default_retrieval_k(top_k: int) -> int:
    """Candidatos do índice que o modo DETAILED passa ao MMR."""
    return max(top_k * 4, 20)


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
        image_embedder: Optional[ImageEmbedder] = None,
        zoom: Optional[ZoomConfig] = None,
        query_cache_size: int = 256,
    ) -> None:
        """`zoom` + `image_embedder`: liga o recorte ampliado no modo
        DETAILED (ver domain/tiling.py); sem eles o VLM vê só o frame."""
        if zoom is not None and image_embedder is None:
            raise ValueError("zoom exige image_embedder para escolher o recorte")
        self._text_embedder = text_embedder
        self._image_embedder = image_embedder
        self._zoom = zoom
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
                         None = default_retrieval_k(top_k).
            mmr_lambda:  1.0 = relevância pura, 0.0 = diversidade pura.
        """
        mode = SearchMode(mode)
        p = profiler
        if mode == SearchMode.FAST:
            retrieval_k = top_k
        elif retrieval_k is None:
            retrieval_k = default_retrieval_k(top_k)

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
            scored = self._caption_and_score(query, query_emb, [h[1] for h in hits], p)
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
        self, query: str, query_emb: np.ndarray, records: Sequence[FrameRecord], p
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
        frames = [images[i] for i in captionable]
        zooms = self._best_tiles(frames, query_emb, p) if self._zoom else None
        captions = self._gpu.run(
            self._captioner.describe_for_query_batch,
            frames, query, zooms=zooms,
            profiler=p, priority=self._gpu.SEARCH,
        )

        t0 = time.perf_counter()
        confidences = self._scorer.score(query, captions)
        if p:
            p.add_time("confidence_score", time.perf_counter() - t0, len(captions))

        return {i: (c, conf) for i, c, conf in zip(captionable, captions, confidences)}

    def _best_tiles(self, frames, query_emb: np.ndarray, p) -> List:
        """Para cada frame, o recorte da grade com maior similaridade SigLIP
        com a query — onde o alvo provavelmente está."""
        t0 = time.perf_counter()
        boxes = [tile_boxes(f.width, f.height, self._zoom) for f in frames]
        tiles = [f.crop(b) for f, bs in zip(frames, boxes) for b in bs]
        embs = self._gpu.run(self._image_embedder.encode_images, tiles, priority=self._gpu.SEARCH)
        sims = np.asarray(embs) @ np.asarray(query_emb)
        best, offset = [], 0
        for bs in boxes:
            best.append(tiles[offset + int(np.argmax(sims[offset:offset + len(bs)]))])
            offset += len(bs)
        if p:
            p.add_time("zoom_select", time.perf_counter() - t0, len(tiles))
        return best
