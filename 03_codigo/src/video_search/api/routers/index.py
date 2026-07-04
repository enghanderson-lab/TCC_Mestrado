"""Endpoints de indexação + WebSocket de progresso."""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import cv2
from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect

from ..auth import require_api_key
from ..camera_db import CameraDB, CameraRecord
from ..deps import get_camera_db, get_job_store_dep
from ..job_store import IndexJob, JobStore
from ..schemas import IndexStartRequest, IndexStatusResponse

router = APIRouter(prefix="/index", tags=["indexing"])

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="vs-index")


def _estimate_frames(video_path: str, interval: float) -> int:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    duration = n_frames / fps if fps > 0 else 0.0
    return max(1, int(duration / interval))


def _index_worker(
    job: IndexJob,
    video_path: str,
    output_dir: str,
    interval: float,
    batch_size: int,
    camera_id: str,
    db: CameraDB,
) -> None:
    from video_search.indexing.pipeline import run_index

    try:
        def on_progress(n: int) -> None:
            job.frames_indexed = n

        summary = run_index(
            video_path=video_path,
            output_dir=output_dir,
            interval=interval,
            batch_size=batch_size,
            on_progress=on_progress,
            should_cancel=job.cancel_event.is_set,
        )

        if job.cancel_event.is_set():
            job.status = "cancelled"
        else:
            job.frames_indexed = summary.frame_count
            job.status = "done"
            db.upsert(CameraRecord(
                camera_id=camera_id,
                video_path=video_path,
                output_dir=output_dir,
                frame_count=summary.frame_count,
                indexed_at=datetime.now(timezone.utc).isoformat(),
                status="indexed",
            ))
        job.finished_at = time.monotonic()

    except Exception as exc:
        if job.cancel_event.is_set():
            job.status = "cancelled"
        else:
            job.status = "error"
            job.error_msg = str(exc)
        job.finished_at = time.monotonic()


# ── REST ────────────────────────────────────────────────────────────────────

@router.post("/start", response_model=IndexStatusResponse, dependencies=[Depends(require_api_key)])
async def start_index(
    req: IndexStartRequest,
    db: CameraDB = Depends(get_camera_db),
    jobs: JobStore = Depends(get_job_store_dep),
) -> IndexStatusResponse:
    if not Path(req.video_path).exists():
        raise HTTPException(status_code=400, detail=f"Arquivo não encontrado: {req.video_path}")

    output_dir = str(Path("index") / req.camera_id)
    estimated = _estimate_frames(req.video_path, req.interval)
    job = jobs.create(camera_id=req.camera_id, frames_estimated=estimated)

    _executor.submit(
        _index_worker,
        job, req.video_path, output_dir, req.interval, req.batch_size, req.camera_id, db,
    )

    return IndexStatusResponse(**job.to_dict())


@router.get("/{job_id}/status", response_model=IndexStatusResponse, dependencies=[Depends(require_api_key)])
async def job_status(
    job_id: str,
    jobs: JobStore = Depends(get_job_store_dep),
) -> IndexStatusResponse:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job não encontrado")
    return IndexStatusResponse(**job.to_dict())


@router.delete("/{job_id}", dependencies=[Depends(require_api_key)])
async def cancel_job(
    job_id: str,
    jobs: JobStore = Depends(get_job_store_dep),
) -> dict:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job não encontrado")
    if job.status != "running":
        raise HTTPException(status_code=400, detail=f"Job não está rodando (status={job.status})")
    job.request_cancel()
    return {"cancelled": job_id}


# ── WebSocket ────────────────────────────────────────────────────────────────

@router.websocket("/ws/{job_id}")
async def ws_progress(
    websocket: WebSocket,
    job_id: str,
    api_key: str = Query(default=""),
) -> None:
    """Stream de progresso do job de indexação.

    Conectar com: ws://host/api/v1/index/ws/{job_id}?api_key=<key>
    Mensagens JSON a cada 0.5 s até o job terminar.
    """
    from video_search.api.auth import _API_KEY

    await websocket.accept()

    if api_key != _API_KEY:
        await websocket.send_json({"error": "unauthorized"})
        await websocket.close(code=1008)
        return

    jobs = get_job_store_dep()

    try:
        while True:
            job = jobs.get(job_id)
            if job is None:
                await websocket.send_json({"error": "job not found"})
                break

            await websocket.send_json(job.to_dict())

            if job.status in ("done", "error", "cancelled"):
                break

            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass
    finally:
        await websocket.close()
