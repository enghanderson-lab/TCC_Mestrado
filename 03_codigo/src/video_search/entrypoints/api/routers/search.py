"""Endpoint de busca semântica multi-câmera (ver application/search_service.py)."""

import time

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool

from ....application.camera_service import CameraService
from ....application.search_service import SearchService
from ..auth import require_api_key
from ..deps import get_camera_service, get_search_service
from ..schemas import FrameResult, SearchRequest, SearchResponse

router = APIRouter(prefix="/search", tags=["search"], dependencies=[Depends(require_api_key)])


@router.post("", response_model=SearchResponse)
async def search(
    req: SearchRequest,
    cameras: CameraService = Depends(get_camera_service),
    service: SearchService = Depends(get_search_service),
) -> SearchResponse:
    if not cameras.list():
        raise HTTPException(status_code=404, detail="Nenhuma câmera indexada")
    selected = cameras.select(req.camera_ids)
    if not selected:
        raise HTTPException(status_code=404, detail="Nenhuma das câmeras solicitadas encontrada")

    t0 = time.perf_counter()
    results = await run_in_threadpool(
        service.search, req.query, selected,
        mode=req.mode, top_k=req.top_k, mmr_lambda=req.mmr_lambda,
    )
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    return SearchResponse(
        query_time_ms=elapsed_ms,
        mode=req.mode,
        results=[
            FrameResult(
                camera_id=r.camera_id,
                timestamp_sec=r.timestamp_sec,
                score=r.score,
                frame_url=f"/api/v1/frames/{r.camera_id}/{int(r.timestamp_sec * 1000)}",
                description=r.caption,
                confidence=r.confidence,
            )
            for r in results
        ],
    )
