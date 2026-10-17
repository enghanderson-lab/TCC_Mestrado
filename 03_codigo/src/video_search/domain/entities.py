"""Entidades do domínio: estruturas de dados puras, sem dependência de
modelos, banco ou framework web."""

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from PIL.Image import Image


@dataclass
class Frame:
    """Frame amostrado de um vídeo."""
    index: int
    timestamp_sec: float
    image: Image


@dataclass
class FrameRecord:
    """Metadados persistidos de um frame indexado."""
    index: int
    timestamp_sec: float
    video: str
    video_path: str = ""
    caption: str = ""


@dataclass
class VideoSource:
    """Uma "câmera" = um arquivo de vídeo com identidade e pasta de índice."""
    camera_id: str
    video_path: str
    output_dir: str


@dataclass
class Camera:
    """Câmera registrada (já indexada)."""
    camera_id: str
    video_path: str
    output_dir: str
    frame_count: int = 0
    indexed_at: Optional[str] = None
    status: str = "indexed"


@dataclass
class IndexingResult:
    """Contabilidade de uma câmera ao fim da indexação."""
    camera_id: str
    output_dir: str
    frames_read: int = 0
    frames_accepted: int = 0
    frames_dropped_motion: int = 0
    frames_dropped_similarity: int = 0


@dataclass
class SearchResult:
    camera_id: str
    frame_index: int
    timestamp_sec: float
    video: str
    video_path: str
    score: float
    caption: Optional[str] = None
    confidence: Optional[float] = None


class JobStatus:
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"
    FINISHED = (DONE, ERROR, CANCELLED)


@dataclass
class IndexJob:
    camera_id: str
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: str = JobStatus.RUNNING
    frames_indexed: int = 0
    frames_estimated: int = 0
    error_msg: Optional[str] = None
    # Relógio de parede (e não monotônico) para sobreviver à persistência.
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False, compare=False)

    def request_cancel(self) -> None:
        """Sinaliza cancelamento cooperativo ao worker de indexação, que
        verifica `cancel_event` periodicamente e encerra a leitura."""
        self.cancel_event.set()
        if self.status == JobStatus.RUNNING:
            self.status = JobStatus.CANCELLED

    def finish(self, status: str, error_msg: Optional[str] = None) -> None:
        self.status = status
        self.error_msg = error_msg
        self.finished_at = time.time()

    @property
    def elapsed_sec(self) -> float:
        end = self.finished_at or time.time()
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
