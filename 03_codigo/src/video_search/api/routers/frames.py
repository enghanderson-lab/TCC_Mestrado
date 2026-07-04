"""Serve frames extraídos on-demand a partir do vídeo original."""

import io
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from fastapi.concurrency import run_in_threadpool
from PIL import Image

from ..auth import require_api_key
from ..camera_db import CameraDB
from ..deps import get_camera_db

router = APIRouter(prefix="/frames", tags=["frames"])


def _extract_frame_jpeg(video_path: str, timestamp_sec: float) -> bytes:
    from video_search.media.frame_extractor import extract_frames_at

    frames = list(extract_frames_at(video_path, [timestamp_sec]))
    if not frames:
        return b""
    img: Image.Image = frames[0]
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


@router.get(
    "/{camera_id}/{timestamp_ms}",
    response_class=Response,
    dependencies=[Depends(require_api_key)],
)
async def get_frame(
    camera_id: str,
    timestamp_ms: int,
    db: CameraDB = Depends(get_camera_db),
) -> Response:
    """Retorna o frame mais próximo ao timestamp em milissegundos como JPEG."""
    rec = db.get(camera_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Câmera não encontrada")

    video_path = rec.video_path
    if not Path(video_path).exists():
        raise HTTPException(status_code=404, detail="Arquivo de vídeo não encontrado")

    timestamp_sec = timestamp_ms / 1000.0
    jpeg_bytes = await run_in_threadpool(_extract_frame_jpeg, video_path, timestamp_sec)

    if not jpeg_bytes:
        raise HTTPException(status_code=404, detail="Frame não encontrado no timestamp")

    return Response(content=jpeg_bytes, media_type="image/jpeg")
