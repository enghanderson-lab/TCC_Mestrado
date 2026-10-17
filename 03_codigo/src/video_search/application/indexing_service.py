"""Caso de uso: indexar um ou vários vídeos/câmeras.

Usa a AsyncIndexPipeline: N threads leitoras (uma por câmera) alimentam a
mesma fila de frames e o BatchCollector combina frames de câmeras diferentes
num único encode_images(), maximizando o lote efetivo na GPU. Vantagens
sobre processar câmera por câmera:
- GPU nunca fica ociosa esperando uma câmera lenta preencher o lote
- Motion-filter e embedding-filter correm em paralelo com a GPU
- Back-pressure bounded evita explosão de RAM
"""

import logging
from typing import Callable, List, Optional, Sequence

from ..config import EmbeddingFilterConfig, MotionDetectionConfig, MultiIndexConfig
from ..domain.entities import IndexingResult, VideoSource
from ..domain.ports import FrameSource, GpuRunner, ImageEmbedder, IndexRepository
from .indexing_pipeline import AsyncIndexPipeline, PipelineConfig

logger = logging.getLogger(__name__)

ProgressFn = Callable[[str, int], None]  # (camera_id, frames aceitos até agora)


class IndexingService:
    def __init__(
        self,
        embedder: ImageEmbedder,
        frames: FrameSource,
        indexes: IndexRepository,
        gpu: GpuRunner,
    ) -> None:
        self._embedder = embedder
        self._frames = frames
        self._indexes = indexes
        self._gpu = gpu

    def index_video(
        self,
        source: VideoSource,
        interval: float = 2.0,
        batch_size: int = 16,
        limit: int = 0,
        on_progress: Optional[ProgressFn] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
        profiler=None,           # Optional[ProfilingContext]
    ) -> IndexingResult:
        """Indexa todos os frames amostrados de um vídeo (sem filtros de
        movimento/similaridade).

        `should_cancel`: checado pelo leitor de frames; quando retorna True,
        a leitura para e o resultado parcial é salvo normalmente.
        """
        return self._run(
            [source],
            PipelineConfig(batch_size=batch_size),
            MotionDetectionConfig(enabled=False),
            EmbeddingFilterConfig(enabled=False),
            interval, limit, on_progress, should_cancel, profiler,
        )[0]

    def index_cameras(
        self,
        sources: Sequence[VideoSource],
        config: Optional[MultiIndexConfig] = None,
        interval: float = 2.0,
        on_progress: Optional[ProgressFn] = None,
        profiler=None,
    ) -> List[IndexingResult]:
        """Indexa várias câmeras concorrentemente com motion detection, lote
        cross-câmera e filtro de similaridade (ver configs/multi_index.yaml)."""
        config = config or MultiIndexConfig()
        results = self._run(
            sources,
            PipelineConfig(
                batch_size=config.batch.batch_size if config.batch.enabled else 1,
                batch_timeout_ms=config.batch.timeout_ms if config.batch.enabled else 0,
            ),
            config.motion_detection,
            config.embedding_filter,
            interval, 0, on_progress, None, profiler,
        )
        for res in results:
            logger.info(
                "[%s] %d/%d frames aceitos | motion=%d similarity=%d",
                res.camera_id, res.frames_accepted, res.frames_read,
                res.frames_dropped_motion, res.frames_dropped_similarity,
            )
        return results

    def _run(self, sources, pipeline_config, motion, similarity, interval, limit,
             on_progress, should_cancel, profiler) -> List[IndexingResult]:
        pipeline = AsyncIndexPipeline(
            embedder=self._embedder,
            frame_source=self._frames,
            new_index=self._indexes.new,
            gpu=self._gpu,
            config=pipeline_config,
            motion_config=motion,
            embedding_config=similarity,
            profiler=profiler,
        )
        return pipeline.run(
            list(sources), interval_sec=interval, limit=limit,
            on_progress=on_progress, should_cancel=should_cancel,
        )
