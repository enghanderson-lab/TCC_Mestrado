"""Endpoint de busca semântica multi-câmera.

FAST:
    Adquire _gpu_lock uma vez, codifica a query (com cache LRU), pesquisa o
    FAISS de todas as câmeras e retorna os top_k globais ordenados por score.

DETAILED:
    Segue o mesmo fluxo mas, após a busca FAISS, aplica MMR cross-câmera e
    passa os top_k frames ao Qwen para geração de legendas + confidence score.
"""

import threading
import time
from collections import defaultdict
from typing import List

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool

from ..auth import require_api_key
from ..camera_db import CameraDB, CameraRecord
from ..deps import get_camera_db
from ..schemas import FrameResult, SearchRequest, SearchResponse

router = APIRouter(prefix="/search", tags=["search"])

# ── Query-embedding LRU cache ────────────────────────────────────────────────
# Evita re-codificar a mesma query quando o usuário pesquisa a mesma string
# em câmeras diferentes ou refaz a consulta.

_embed_cache: dict = {}
_embed_cache_lock = threading.Lock()
_CACHE_MAX = 256


def _get_cached_embed(query: str, embedder) -> np.ndarray:
    with _embed_cache_lock:
        if query in _embed_cache:
            return _embed_cache[query]

    emb = embedder.encode_text([query])[0]

    with _embed_cache_lock:
        if len(_embed_cache) >= _CACHE_MAX:
            _embed_cache.pop(next(iter(_embed_cache)))
        _embed_cache[query] = emb

    return emb


# ── Search logic (runs in threadpool — acquires GPU lock) ────────────────────

def _sync_search(
    query: str,
    cameras: List[CameraRecord],
    top_k: int,
    mode: str,
    mmr_lambda: float,
) -> List[dict]:
    from video_search.indexing.embedding_store import load_store
    from video_search.indexing.pipeline import _gpu_lock
    from video_search.indexing.reranker import MmrReranker, RerankerConfig
    from video_search.media.frame_extractor import extract_frames_at
    from video_search.models.model_manager import get_model_manager

    mm = get_model_manager()
    retrieval_k = max(top_k * 4, 20)
    is_detailed = mode == "detailed"

    with _gpu_lock:
        embedder = mm.get_siglip()
        query_emb = _get_cached_embed(query, embedder)

        # --- FAISS search para todas as câmeras ---------------------------------
        if is_detailed:
            # Precisamos dos embeddings dos candidatos para o MMR
            all_candidates = []   # (score, FrameRecord, embedding, camera_id)
            for cam in cameras:
                try:
                    store = load_store(cam.output_dir)
                    for score, rec, emb in store.search_with_embeddings(query_emb, top_k=retrieval_k):
                        all_candidates.append((score, rec, emb, cam.camera_id))
                except Exception:
                    continue

            if not all_candidates:
                return []

            # Ordena globalmente e limita ao retrieval_k
            all_candidates.sort(key=lambda x: x[0], reverse=True)
            all_candidates = all_candidates[:retrieval_k]

            # Mapa rec → camera_id ANTES do MMR (mesmo objeto Python)
            rec_to_cam = {id(r): cid for _, r, _, cid in all_candidates}

            # MMR cross-câmera
            reranker = MmrReranker(RerankerConfig(lambda_=mmr_lambda))
            mmr_input = [(s, r, e) for s, r, e, _ in all_candidates]
            reranked = reranker.rerank(query_emb, mmr_input, k=top_k)
            # reranked: List[Tuple[float, FrameRecord]]

            # Extrai frames agrupados por vídeo (abre cada arquivo uma vez)
            by_video: dict = defaultdict(list)
            for i, (_, rec) in enumerate(reranked):
                if rec.video_path:
                    by_video[rec.video_path].append((i, rec.timestamp_sec))

            images: dict = {}
            for vpath, items in by_video.items():
                timestamps = [ts for _, ts in items]
                indices = [idx for idx, _ in items]
                try:
                    for frame_idx, frame in zip(indices, extract_frames_at(vpath, timestamps)):
                        images[frame_idx] = frame
                except Exception:
                    pass

            captionable = [i for i in range(len(reranked)) if i in images]

            captions: dict = {}
            if captionable:
                describer = mm.get_qwen()
                frame_list = [images[i] for i in captionable]
                cap_texts = describer.describe_for_query_batch(frame_list, query)
                captions = dict(zip(captionable, cap_texts))

            confidences: dict = {}
            if captions:
                st_model = mm.get_sentence_transformer()
                texts = [query] + list(captions.values())
                embs = st_model.encode(texts, normalize_embeddings=True)
                q_emb_st = embs[0]
                for j, (idx, _) in enumerate(captions.items()):
                    confidences[idx] = float(np.dot(q_emb_st, embs[j + 1]))

            results = []
            for i, (score, rec) in enumerate(reranked):
                results.append({
                    "camera_id": rec_to_cam.get(id(rec), "unknown"),
                    "timestamp_sec": rec.timestamp_sec,
                    "score": float(score),
                    "video_path": rec.video_path,
                    "description": captions.get(i),
                    "confidence": confidences.get(i),
                })
            return results

        else:
            # FAST: pesquisa direta sem embeddings dos candidatos
            all_fast = []  # (score, FrameRecord, camera_id)
            for cam in cameras:
                try:
                    store = load_store(cam.output_dir)
                    for score, rec in store.search(query_emb, top_k=top_k):
                        all_fast.append((float(score), rec, cam.camera_id))
                except Exception:
                    continue

            all_fast.sort(key=lambda x: x[0], reverse=True)
            return [
                {
                    "camera_id": cam_id,
                    "timestamp_sec": rec.timestamp_sec,
                    "score": score,
                    "video_path": rec.video_path,
                    "description": None,
                    "confidence": None,
                }
                for score, rec, cam_id in all_fast[:top_k]
            ]


# ── Endpoint ─────────────────────────────────────────────────────────────────

@router.post("", response_model=SearchResponse, dependencies=[Depends(require_api_key)])
async def search(
    req: SearchRequest,
    db: CameraDB = Depends(get_camera_db),
) -> SearchResponse:
    all_cameras = db.list_all()
    if not all_cameras:
        raise HTTPException(status_code=404, detail="Nenhuma câmera indexada")

    if req.camera_ids:
        cameras = [c for c in all_cameras if c.camera_id in req.camera_ids]
        if not cameras:
            raise HTTPException(status_code=404, detail="Nenhuma das câmeras solicitadas encontrada")
    else:
        cameras = all_cameras

    t0 = time.perf_counter()
    raw_results = await run_in_threadpool(
        _sync_search, req.query, cameras, req.top_k, req.mode, req.mmr_lambda
    )
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    frame_results = [
        FrameResult(
            camera_id=r["camera_id"],
            timestamp_sec=r["timestamp_sec"],
            score=r["score"],
            frame_url=f"/api/v1/frames/{r['camera_id']}/{int(r['timestamp_sec'] * 1000)}",
            description=r.get("description"),
            confidence=r.get("confidence"),
        )
        for r in raw_results
    ]

    return SearchResponse(
        query_time_ms=elapsed_ms,
        mode=req.mode,
        results=frame_results,
    )
