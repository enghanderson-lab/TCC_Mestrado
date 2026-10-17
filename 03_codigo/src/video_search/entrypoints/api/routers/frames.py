"""Serve frames extraídos on-demand a partir do vídeo original."""

import io
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response

from ....application.camera_service import CameraService
from ....bootstrap import Container
from ..auth import require_api_key
from ..deps import get_camera_service, get_container

router = APIRouter(prefix="/frames", tags=["frames"], dependencies=[Depends(require_api_key)])


@router.get("/{camera_id}/{timestamp_ms}", response_class=Response)
async def get_frame(
    camera_id: str,
    timestamp_ms: int,
    cameras: CameraService = Depends(get_camera_service),
    c: Container = Depends(get_container),
) -> Response:
    """Retorna o frame mais próximo ao timestamp em milissegundos como JPEG."""
    camera = cameras.get(camera_id)
    if camera is None:
        raise HTTPException(status_code=404, detail="Câmera não encontrada")
    if not Path(camera.video_path).exists():
        raise HTTPException(status_code=404, detail="Arquivo de vídeo não encontrado")

    def _jpeg() -> bytes:
        image = c.frames.frames_at(camera.video_path, [timestamp_ms / 1000.0])[0]
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=85)
        return buf.getvalue()

    try:
        jpeg_bytes = await run_in_threadpool(_jpeg)
    except (FileNotFoundError, ValueError):
        raise HTTPException(status_code=404, detail="Frame não encontrado no timestamp")
    return Response(content=jpeg_bytes, media_type="image/jpeg")
