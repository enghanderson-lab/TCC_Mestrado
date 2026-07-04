"""Indexação de múltiplos vídeos/câmeras concorrentemente.

Usa AsyncIndexPipeline: N threads leitoras (uma por câmera) alimentam a
mesma fila de frames. O BatchCollector combina frames de câmeras diferentes
num único encode_images(), maximizando o batch efetivo na GPU.

Comparado à versão anterior (um BatchDispatcher por câmera em sequência):
- GPU nunca fica ociosa esperando uma câmera lenta preencher o batch
- Motion-filter e embedding-filter correm em paralelo com a GPU
- Back-pressure bounded evita explosão de RAM

`_make_embedder` permanece como factory patcheável para testes (sem GPU).
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from .async_pipeline import AsyncIndexPipeline, PipelineConfig, PipelineSource
from ..utils.config import MultiIndexConfig
from .pipeline import _gpu_lock

logger = logging.getLogger(__name__)


@dataclass
class CameraSource:
    """Uma "câmera" = um arquivo de vídeo com identidade e pasta de saída."""
    camera_id: str
    video_path: str
    output_dir: str


@dataclass
class CameraSummary:
    camera_id: str
    output_dir: str
    frame_count: int
    frames_read: int
    frames_discarded_motion: int
    frames_discarded_similarity: int


# ------------------------------------------------------------------
# Factory patcheável — permite monkeypatch nos testes sem GPU/ModelManager
# ------------------------------------------------------------------

def _make_embedder():
    """Retorna o SigLIPEmbedder residente via ModelManager."""
    from ..models.model_manager import get_model_manager
    return get_model_manager().get_siglip()


def run_multi_index(
    sources: List[CameraSource],
    config: Optional[MultiIndexConfig] = None,
    interval: float = 2.0,
    on_progress: Optional[Callable[[str, int], None]] = None,
    profiling_dir: Optional[str] = None,
) -> List[CameraSummary]:
    """Processa todas as câmeras em paralelo com uma pipeline orientada a filas.

    O SigLIP compartilhado recebe batches cross-câmera: frames de câmeras
    diferentes são agrupados num mesmo lote GPU em vez de processar câmera
    por câmera (que desperdiçaria capacidade quando uma câmera está lenta).

    Os modelos (SigLIP) são obtidos via ModelManager e mantidos residentes;
    o gerenciamento do ciclo de vida é do ModelManager / do chamador.

    Args:
        profiling_dir: pasta onde salvar relatório + gráfico de profiling.
                       None (padrão) → nenhum arquivo gerado.
    """
    from ..profiling import ProfilingContext, HardwareMonitor, generate_report

    config = config or MultiIndexConfig()
    embedder = _make_embedder()

    profiler = ProfilingContext()
    hw_monitor = HardwareMonitor().start() if profiling_dir is not None else None

    pipeline_sources = [
        PipelineSource(
            camera_id=src.camera_id,
            video_path=src.video_path,
            output_dir=src.output_dir,
        )
        for src in sources
    ]

    pipeline = AsyncIndexPipeline(
        embedder=embedder,
        config=PipelineConfig(
            batch_size=config.batch.batch_size if config.batch.enabled else 1,
            batch_timeout_ms=config.batch.timeout_ms if config.batch.enabled else 0,
        ),
        motion_config=config.motion_detection,
        embedding_config=config.embedding_filter,
        gpu_lock=_gpu_lock,
        profiler=profiler,
    )

    pipeline_results = pipeline.run(
        pipeline_sources,
        interval_sec=interval,
        on_progress=on_progress,
    )

    if hw_monitor is not None:
        hw_monitor.stop()

    if profiling_dir is not None:
        hw = hw_monitor.summary() if hw_monitor else None
        generate_report(profiler, label="indexing", hardware=hw, profiling_dir=profiling_dir)

    summaries = []
    for res in pipeline_results:
        logger.info(
            "[%s] %d/%d frames aceitos | motion=%d similarity=%d",
            res.camera_id,
            res.frames_accepted,
            res.frames_read,
            res.frames_dropped_motion,
            res.frames_dropped_similarity,
        )
        summaries.append(CameraSummary(
            camera_id=res.camera_id,
            output_dir=res.output_dir,
            frame_count=res.frames_accepted,
            frames_read=res.frames_read,
            frames_discarded_motion=res.frames_dropped_motion,
            frames_discarded_similarity=res.frames_dropped_similarity,
        ))

    return summaries
