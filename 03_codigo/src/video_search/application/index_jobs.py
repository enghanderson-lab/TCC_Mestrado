"""Caso de uso: jobs de indexação em background (usado pela API).

Cada job roda numa thread do pool e é persistido no JobRepository: o status
sobrevive a um reinício da API (jobs que estavam rodando são marcados como
erro na inicialização). O progresso é mantido em memória enquanto o job
roda e gravado no repositório no máximo a cada `progress_persist_sec`.
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from ..domain.entities import Camera, IndexJob, JobStatus, VideoSource
from ..domain.ports import CameraRepository, FrameSource, JobRepository
from .indexing_service import IndexingService

logger = logging.getLogger(__name__)


class JobNotRunning(Exception):
    pass


class IndexJobService:
    def __init__(
        self,
        indexing: IndexingService,
        frames: FrameSource,
        cameras: CameraRepository,
        jobs: JobRepository,
        index_root: Path,
        max_workers: int = 2,
        progress_persist_sec: float = 1.0,
    ) -> None:
        self._indexing = indexing
        self._frames = frames
        self._cameras = cameras
        self._jobs = jobs
        self._index_root = Path(index_root)
        self._progress_persist_sec = progress_persist_sec
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="vs-index")
        self._live: Dict[str, IndexJob] = {}
        self._lock = threading.Lock()

        interrupted = self._jobs.mark_interrupted()
        if interrupted:
            logger.warning("%d job(s) de indexacao interrompido(s) na execucao anterior", interrupted)

    def start(self, camera_id: str, video_path: str, interval: float = 2.0, batch_size: int = 16) -> IndexJob:
        source = VideoSource(
            camera_id=camera_id,
            video_path=video_path,
            output_dir=str(self._index_root / camera_id),
        )
        job = IndexJob(
            camera_id=camera_id,
            frames_estimated=self._frames.estimate_samples(video_path, interval),
        )
        self._jobs.save(job)
        with self._lock:
            self._live[job.job_id] = job
        self._executor.submit(self._run, job, source, interval, batch_size)
        return job

    def get(self, job_id: str) -> Optional[IndexJob]:
        with self._lock:
            job = self._live.get(job_id)
        return job or self._jobs.get(job_id)

    def cancel(self, job_id: str) -> Optional[IndexJob]:
        """None se o job não existir; JobNotRunning se já terminou."""
        job = self.get(job_id)
        if job is None:
            return None
        if job.status != JobStatus.RUNNING:
            raise JobNotRunning(job.status)
        job.request_cancel()
        self._jobs.save(job)
        return job

    def active_count(self) -> int:
        with self._lock:
            return sum(1 for j in self._live.values() if j.status == JobStatus.RUNNING)

    def shutdown(self) -> None:
        """Cancela os jobs em andamento e espera as threads terminarem."""
        with self._lock:
            running = list(self._live.values())
        for job in running:
            job.request_cancel()
        self._executor.shutdown(wait=True)

    def _run(self, job: IndexJob, source: VideoSource, interval: float, batch_size: int) -> None:
        last_persist = time.monotonic()

        def on_progress(_camera_id: str, count: int) -> None:
            nonlocal last_persist
            job.frames_indexed = count
            now = time.monotonic()
            if now - last_persist >= self._progress_persist_sec:
                last_persist = now
                self._jobs.save(job)

        try:
            result = self._indexing.index_video(
                source, interval=interval, batch_size=batch_size,
                on_progress=on_progress, should_cancel=job.cancel_event.is_set,
            )
            if job.cancel_event.is_set():
                job.finish(JobStatus.CANCELLED)
            else:
                job.frames_indexed = result.frames_accepted
                self._cameras.upsert(Camera(
                    camera_id=source.camera_id,
                    video_path=source.video_path,
                    output_dir=source.output_dir,
                    frame_count=result.frames_accepted,
                    indexed_at=datetime.now(timezone.utc).isoformat(),
                ))
                job.finish(JobStatus.DONE)
        except Exception as exc:
            if job.cancel_event.is_set():
                job.finish(JobStatus.CANCELLED)
            else:
                logger.exception("job %s falhou", job.job_id)
                job.finish(JobStatus.ERROR, str(exc))
        finally:
            self._jobs.save(job)
            with self._lock:
                self._live.pop(job.job_id, None)
