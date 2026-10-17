from typing import List

from fastapi import APIRouter, Depends, HTTPException

from ....application.camera_service import CameraService
from ..auth import require_api_key
from ..deps import get_camera_service
from ..schemas import CameraInfo

router = APIRouter(prefix="/cameras", tags=["cameras"], dependencies=[Depends(require_api_key)])


@router.get("", response_model=List[CameraInfo])
async def list_cameras(cameras: CameraService = Depends(get_camera_service)) -> List[CameraInfo]:
    return [CameraInfo(**vars(c)) for c in cameras.list()]


@router.delete("/{camera_id}")
async def delete_camera(camera_id: str, cameras: CameraService = Depends(get_camera_service)) -> dict:
    if not cameras.delete(camera_id):
        raise HTTPException(status_code=404, detail="Camera not found")
    return {"deleted": camera_id}
