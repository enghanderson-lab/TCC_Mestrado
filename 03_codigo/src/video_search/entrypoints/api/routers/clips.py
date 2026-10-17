"""Serve um trecho (clip) de vídeo em MP4 ao redor de uma detecção
(ver adapters/media/pyav_clips.py)."""

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response

from ....adapters.media.pyav_clips import ClipError, extract_clip_h264, probe_fps
from ....application.camera_service import CameraService
from ..auth import require_api_key
from ..deps import get_camera_service

router = APIRouter(prefix="/clips", tags=["clips"], dependencies=[Depends(require_api_key)])


@router.get("/{camera_id}/{timestamp_ms}", response_class=Response)
async def get_clip(
    camera_id: str,
    timestamp_ms: int,
    duration: float = Query(default=30.0, ge=1.0, le=120.0),
    lead_frames: int = Query(default=3, ge=0, le=100),
    cameras: CameraService = Depends(get_camera_service),
) -> Response:
    """Retorna `duration` segundos de vídeo em MP4 (H.264), iniciando
    `lead_frames` quadros antes do timestamp informado."""
    camera = cameras.get(camera_id)
    if camera is None:
        raise HTTPException(status_code=404, detail="Câmera não encontrada")
    video_path = camera.video_path
    if not Path(video_path).exists():
        raise HTTPException(status_code=404, detail="Arquivo de vídeo não encontrado")

    try:
        fps = await run_in_threadpool(probe_fps, video_path)
        start_sec = max(0.0, timestamp_ms / 1000.0 - lead_frames / fps)
        clip_bytes = await run_in_threadpool(extract_clip_h264, video_path, start_sec, duration)
    except ClipError as exc:
        raise HTTPException(status_code=404, detail=f"Não foi possível extrair o trecho de vídeo: {exc}")

    if not clip_bytes:
        raise HTTPException(status_code=404, detail="Não foi possível extrair o trecho de vídeo no timestamp informado")
    return Response(content=clip_bytes, media_type="video/mp4")
