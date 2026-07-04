import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class IndexJob:
    job_id: str
    camera_id: str
    status: str = "running"          # running | done | error | cancelled
    frames_indexed: int = 0
    frames_estimated: int = 0
    error_msg: Optional[str] = None
    started_at: float = field(default_factory=time.monotonic)
    finished_at: Optional[float] = None

    @property
    def elapsed_sec(self) -> float:
        end = self.finished_at or time.monotonic()
        return round(end - self.started_at, 2)

    @property
    def fps(self) -> float:
        elapsed = self.elapsed_sec
        return round(self.frames_indexed / elapsed, 1) if elapsed > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "camera_id": self.camera_id,
            "status": self.status,
            "frames_indexed": self.frames_indexed,
            "frames_estimated": self.frames_estimated,
            "elapsed_sec": self.elapsed_sec,
            "fps": self.fps,
            "error": self.error_msg,
        }


class JobStore:
    def __init__(self) -> None:
        self._jobs: Dict[str, IndexJob] = {}
        self._lock = threading.Lock()

    def create(self, camera_id: str, frames_estimated: int = 0) -> IndexJob:
        job = IndexJob(
            job_id=str(uuid.uuid4()),
            camera_id=camera_id,
            frames_estimated=frames_estimated,
        )
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> Optional[IndexJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def active_count(self) -> int:
        with self._lock:
            return sum(1 for j in self._jobs.values() if j.status == "running")

    def all(self) -> List[IndexJob]:
        with self._lock:
            return list(self._jobs.values())


_job_store = JobStore()


def get_job_store() -> JobStore:
    return _job_store
