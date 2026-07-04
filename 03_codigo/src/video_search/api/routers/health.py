import torch
from fastapi import APIRouter

from ..deps import get_camera_db, get_job_store_dep
from ..schemas import HealthResponse

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    from video_search.models.model_manager import get_model_manager
    mm = get_model_manager()

    vram_used = vram_total = None
    if torch.cuda.is_available():
        vram_used = round(torch.cuda.memory_allocated() / 1024**2, 1)
        vram_total = round(torch.cuda.get_device_properties(0).total_memory / 1024**2, 1)

    return HealthResponse(
        status="ok",
        models_loaded=mm._loaded,
        vram_used_mb=vram_used,
        vram_total_mb=vram_total,
        active_jobs=get_job_store_dep().active_count(),
    )
