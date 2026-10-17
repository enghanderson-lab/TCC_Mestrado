"""Endpoints de indexação + WebSocket de progresso (ver application/index_jobs.py)."""

import asyncio
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool

from ....application.index_jobs import IndexJobService, JobNotRunning
from ....bootstrap import Container
from ....domain.entities import JobStatus
from ..auth import is_valid_key, require_api_key
from ..deps import get_container, get_job_service
from ..schemas import IndexStartRequest, IndexStatusResponse

router = APIRouter(prefix="/index", tags=["indexing"])

_ALLOWED_VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv")


# ── REST ────────────────────────────────────────────────────────────────────

@router.post("/start", response_model=IndexStatusResponse, dependencies=[Depends(require_api_key)])
async def start_index(
    req: IndexStartRequest,
    jobs: IndexJobService = Depends(get_job_service),
) -> IndexStatusResponse:
    if not Path(req.video_path).exists():
        raise HTTPException(status_code=400, detail=f"Arquivo não encontrado: {req.video_path}")
    job = jobs.start(req.camera_id, req.video_path, interval=req.interval, batch_size=req.batch_size)
    return IndexStatusResponse(**job.to_dict())


@router.post("/upload", response_model=IndexStatusResponse, dependencies=[Depends(require_api_key)])
async def upload_and_index(
    camera_id: str = Form(...),
    file: UploadFile = File(...),
    interval: float = Form(default=2.0, ge=0.1, le=60.0),
    batch_size: int = Form(default=16, ge=1, le=64),
    jobs: IndexJobService = Depends(get_job_service),
    c: Container = Depends(get_container),
) -> IndexStatusResponse:
    """Recebe um arquivo de vídeo via multipart, salva em disco e inicia a
    indexação em background — mesmo fluxo de /index/start, mas para clientes
    (ex.: browser) que não têm acesso ao filesystem do servidor para apontar
    um video_path diretamente.
    """
    filename = file.filename or ""
    if not filename.lower().endswith(_ALLOWED_VIDEO_EXT):
        raise HTTPException(
            status_code=400,
            detail=f"Formato não suportado. Use um dos seguintes: {', '.join(_ALLOWED_VIDEO_EXT)}",
        )

    upload_dir = c.settings.upload_dir
    upload_dir.mkdir(parents=True, exist_ok=True)
    # Path(...).name descarta qualquer diretório vindo do cliente no filename.
    dest = upload_dir / f"{camera_id}_{Path(filename).name}"

    def _save() -> None:
        with open(dest, "wb") as out:
            shutil.copyfileobj(file.file, out)

    await run_in_threadpool(_save)

    if not dest.exists() or dest.stat().st_size == 0:
        raise HTTPException(status_code=400, detail="Falha ao salvar o vídeo enviado")

    job = jobs.start(camera_id, str(dest), interval=interval, batch_size=batch_size)
    return IndexStatusResponse(**job.to_dict())


@router.get("/{job_id}/status", response_model=IndexStatusResponse, dependencies=[Depends(require_api_key)])
async def job_status(job_id: str, jobs: IndexJobService = Depends(get_job_service)) -> IndexStatusResponse:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job não encontrado")
    return IndexStatusResponse(**job.to_dict())


@router.delete("/{job_id}", dependencies=[Depends(require_api_key)])
async def cancel_job(job_id: str, jobs: IndexJobService = Depends(get_job_service)) -> dict:
    try:
        job = jobs.cancel(job_id)
    except JobNotRunning as exc:
        raise HTTPException(status_code=400, detail=f"Job não está rodando (status={exc})")
    if job is None:
        raise HTTPException(status_code=404, detail="Job não encontrado")
    return {"cancelled": job_id}


# ── WebSocket ────────────────────────────────────────────────────────────────

@router.websocket("/ws/{job_id}")
async def ws_progress(websocket: WebSocket, job_id: str, api_key: str = Query(default="")) -> None:
    """Stream de progresso do job de indexação.

    Conectar com: ws://host/api/v1/index/ws/{job_id}?api_key=<key>
    (browsers não enviam headers customizados no handshake de WS).
    Mensagens JSON a cada 0.5 s até o job terminar.
    """
    await websocket.accept()

    if not is_valid_key(websocket.app, api_key):
        await websocket.send_json({"error": "unauthorized"})
        await websocket.close(code=1008)
        return

    jobs: IndexJobService = websocket.app.state.container.jobs
    try:
        while True:
            job = jobs.get(job_id)
            if job is None:
                await websocket.send_json({"error": "job not found"})
                break
            await websocket.send_json(job.to_dict())
            if job.status in JobStatus.FINISHED:
                break
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass
    finally:
        await websocket.close()
