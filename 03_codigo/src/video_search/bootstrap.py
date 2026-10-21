"""Composition root: o único lugar que conhece as implementações concretas.

Monta os adapters (SigLIP, Qwen, NLI de confiança, FAISS, OpenCV,
SQLite, executor de GPU) e os injeta nos serviços de `application/`. CLI e
API criam um `Container` cada e usam apenas seus serviços.

Os serviços são criados sob demanda (e uma única vez): a CLI de indexação
não abre o banco SQLite, e o banco não carrega modelos. O mesmo vale para os
modelos: a busca recebe Qwen e NLI por `_OnDemand`, então uma busca FAST (e a
indexação) só carrega o SigLIP.
"""

import threading
from typing import Callable, Dict, Optional, TypeVar

from .adapters.gpu_executor import GpuExecutor
from .adapters.media.opencv_frames import OpenCVFrameSource
from .adapters.models.model_config import ModelConfig
from .adapters.models.model_manager import ModelManager
from .adapters.storage.file_index import FileIndexRepository
from .adapters.storage.sqlite import SQLiteDatabase
from .adapters.storage.sqlite_cameras import SQLiteCameraRepository
from .adapters.storage.sqlite_jobs import SQLiteJobRepository
from .application.camera_service import CameraService
from .application.index_jobs import IndexJobService
from .application.indexing_service import IndexingService
from .application.search_service import SearchService
from .config import Settings
from .domain.tiling import ZoomConfig

T = TypeVar("T")


class _OnDemand:
    """Representa um modelo que só é carregado no primeiro uso de um atributo."""

    def __init__(self, load: Callable[[], object]) -> None:
        self._load = load

    def __getattr__(self, name: str):
        return getattr(self._load(), name)


class Container:
    def __init__(self, settings: Optional[Settings] = None, model_config: Optional[ModelConfig] = None) -> None:
        self.settings = settings or Settings.from_env()
        self.models = ModelManager(model_config)
        self.gpu = GpuExecutor()
        self.frames = OpenCVFrameSource()
        self.indexes = FileIndexRepository()
        self._instances: Dict[str, object] = {}
        self._lock = threading.RLock()

    def _once(self, name: str, factory: Callable[[], T]) -> T:
        with self._lock:
            if name not in self._instances:
                self._instances[name] = factory()
            return self._instances[name]  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Serviços
    # ------------------------------------------------------------------

    @property
    def search(self) -> SearchService:
        return self._once("search", lambda: SearchService(
            text_embedder=self.models.get_siglip(),
            captioner=_OnDemand(self.models.get_qwen),
            scorer=_OnDemand(self.models.get_scorer),
            indexes=self.indexes,
            frames=self.frames,
            gpu=self.gpu,
            image_embedder=self.models.get_siglip(),
            zoom=ZoomConfig(),
        ))

    @property
    def indexing(self) -> IndexingService:
        return self._once("indexing", lambda: IndexingService(
            embedder=self.models.get_siglip(),
            frames=self.frames,
            indexes=self.indexes,
            gpu=self.gpu,
        ))

    @property
    def jobs(self) -> IndexJobService:
        return self._once("jobs", lambda: IndexJobService(
            indexing=self.indexing,
            frames=self.frames,
            cameras=self._camera_repository,
            jobs=SQLiteJobRepository(self._db),
            index_root=self.settings.index_root,
        ))

    @property
    def cameras(self) -> CameraService:
        return self._once("cameras", lambda: CameraService(self._camera_repository, self.indexes))

    # ------------------------------------------------------------------

    @property
    def _db(self) -> SQLiteDatabase:
        return self._once("db", lambda: SQLiteDatabase(self.settings.db_path))

    @property
    def _camera_repository(self) -> SQLiteCameraRepository:
        return self._once("camera_repository", lambda: SQLiteCameraRepository(self._db))

    def active_jobs(self) -> int:
        """Jobs de indexação rodando; 0 se o serviço de jobs nem foi criado
        (sem forçar o carregamento dos modelos)."""
        jobs = self._instances.get("jobs")
        return jobs.active_count() if jobs is not None else 0  # type: ignore[attr-defined]

    def close(self) -> None:
        """Encerra jobs, a thread da GPU e libera os modelos."""
        jobs = self._instances.get("jobs")
        if jobs is not None:
            jobs.shutdown()  # type: ignore[attr-defined]
        self.gpu.shutdown()
        self.models.unload_models()

    def __enter__(self) -> "Container":
        return self

    def __exit__(self, *_) -> None:
        self.close()
