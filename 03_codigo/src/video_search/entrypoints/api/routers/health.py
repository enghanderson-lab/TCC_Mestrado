import torch
from fastapi import APIRouter, Depends

from ....bootstrap import Container
from ..deps import get_container
from ..schemas import HealthResponse

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health(c: Container = Depends(get_container)) -> HealthResponse:
    vram_used = vram_total = None
    if torch.cuda.is_available():
        vram_used = round(torch.cuda.memory_allocated() / 1024**2, 1)
        vram_total = round(torch.cuda.get_device_properties(0).total_memory / 1024**2, 1)

    return HealthResponse(
        status="ok",
        models_loaded=c.models.is_loaded,
        vram_used_mb=vram_used,
        vram_total_mb=vram_total,
        active_jobs=c.active_jobs(),
    )
