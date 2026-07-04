import shutil
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, HTTPException

from ..auth import require_api_key
from ..camera_db import CameraDB
from ..deps import get_camera_db
from ..schemas import CameraInfo

router = APIRouter(prefix="/cameras", tags=["cameras"])


@router.get("", response_model=List[CameraInfo], dependencies=[Depends(require_api_key)])
async def list_cameras(db: CameraDB = Depends(get_camera_db)) -> List[CameraInfo]:
    return [CameraInfo(**vars(c)) for c in db.list_all()]


@router.delete("/{camera_id}", dependencies=[Depends(require_api_key)])
async def delete_camera(
    camera_id: str,
    db: CameraDB = Depends(get_camera_db),
) -> dict:
    rec = db.get(camera_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Camera not found")

    index_path = Path(rec.output_dir)
    if index_path.exists():
        shutil.rmtree(index_path)

    db.delete(camera_id)
    return {"deleted": camera_id}
