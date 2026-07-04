import os
import threading
from pathlib import Path

from .camera_db import CameraDB
from .job_store import JobStore, get_job_store

_camera_db: CameraDB | None = None
_db_lock = threading.Lock()


def get_camera_db() -> CameraDB:
    global _camera_db
    if _camera_db is None:
        with _db_lock:
            if _camera_db is None:
                db_path = Path(os.getenv("VS_DB_PATH", "cameras.db"))
                _camera_db = CameraDB(db_path)
    return _camera_db


def get_job_store_dep() -> JobStore:
    return get_job_store()
