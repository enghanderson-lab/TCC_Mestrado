"""Indexacao de multiplos videos/cameras concorrentemente, com filtro de
movimento, batching cross-camera de SigLIP (`BatchDispatcher`) e filtro de
similaridade de embeddings antes de legendar (Qwen2.5-VL).

Ponto de entrada NOVO e independente de `run_index`/`run_search`
(`pipeline.py`) -- nao altera nada la, so reaproveita `_gpu_lock`,
`extract_frames`, `EmbeddingStore`/`FrameRecord` e acessa os modelos via
ModelManager. A saida (`embeddings.npy`+`records.json`+`metadata.json`)
fica no mesmo formato de hoje, populando o campo `caption` do `FrameRecord`
(ja existente, sem uso em `run_index`), entao `run_search`/CLI continuam
funcionando sem mudanca sobre os indices gerados aqui.

`_make_embedder` e `_make_describer` sao funcoes de modulo-nivel patcheaveis
nos testes (monkeypatch.setattr), evitando que o teste precise tocar
os modelos reais."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import torch

from .batch_dispatcher import BatchDispatcher
from .config import MultiIndexConfig
from .embedding_filter import EmbeddingSimilarityFilter
from .embedding_store import EmbeddingStore, FrameRecord
from .frame_extractor import extract_frames
from .motion_filter import MotionFilter
from .pipeline import _gpu_lock

logger = logging.getLogger(__name__)


@dataclass
class CameraSource:
    """Uma "camera" = um video_path (arquivo OU, no futuro, uma URL
    rtsp://... -- `cv2.VideoCapture` e agnostico, ver `frame_extractor.py`)
    + identidade + pasta de saida."""

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
# Factories patcheaveis — permitem monkeypatch nos testes sem tocar
# nos modelos reais nem no ModelManager.
# ------------------------------------------------------------------

def _make_embedder():
    """Retorna o SigLIPEmbedder residente via ModelManager."""
    from .model_manager import get_model_manager
    return get_model_manager().get_siglip()


def _make_describer():
    """Retorna o Qwen2VLDescriber residente via ModelManager."""
    from .model_manager import get_model_manager
    return get_model_manager().get_qwen()


def _process_camera(
    source: CameraSource,
    interval: float,
    motion_filter: MotionFilter,
    embedding_filter: EmbeddingSimilarityFilter,
    dispatcher: BatchDispatcher,
    describer,
    on_progress: Optional[Callable[[str, int], None]],
) -> CameraSummary:
    store = EmbeddingStore()
    video_name = Path(source.video_path).name
    resolved_video_path = str(Path(source.video_path).resolve())

    frames_read = 0
    discarded_motion = 0
    discarded_similarity = 0
    t_motion = t_batch_wait = t_caption = 0.0

    for frame in extract_frames(source.video_path, interval_sec=interval):
        frames_read += 1

        t0 = time.monotonic()
        accepted = motion_filter.should_accept(source.camera_id, frame.image)
        t_motion += time.monotonic() - t0
        if not accepted:
            discarded_motion += 1
            continue

        t0 = time.monotonic()
        embedding = dispatcher.submit(frame.image).result()
        t_batch_wait += time.monotonic() - t0

        if not embedding_filter.should_accept(source.camera_id, embedding):
            discarded_similarity += 1
            continue

        t0 = time.monotonic()
        with _gpu_lock:
            caption = describer.describe(frame.image)
        t_caption += time.monotonic() - t0

        record = FrameRecord(
            index=frame.index,
            timestamp_sec=frame.timestamp_sec,
            video=video_name,
            video_path=resolved_video_path,
            caption=caption,
        )
        store.add(embedding, record)
        if on_progress is not None:
            on_progress(source.camera_id, len(store))

    store.save(source.output_dir)

    motion_pct = 100 * discarded_motion / frames_read if frames_read else 0.0
    similarity_pct = 100 * discarded_similarity / frames_read if frames_read else 0.0
    logger.info(
        "[%s] %d/%d frames aceitos | motion: %d descartados (%.1f%%) | "
        "similarity: %d descartados (%.1f%%) | tempo motion=%.2fs espera_batch=%.2fs legenda=%.2fs",
        source.camera_id,
        len(store),
        frames_read,
        discarded_motion,
        motion_pct,
        discarded_similarity,
        similarity_pct,
        t_motion,
        t_batch_wait,
        t_caption,
    )

    return CameraSummary(
        camera_id=source.camera_id,
        output_dir=source.output_dir,
        frame_count=len(store),
        frames_read=frames_read,
        frames_discarded_motion=discarded_motion,
        frames_discarded_similarity=discarded_similarity,
    )


def run_multi_index(
    sources: List[CameraSource],
    config: Optional[MultiIndexConfig] = None,
    interval: float = 2.0,
    on_progress: Optional[Callable[[str, int], None]] = None,
) -> List[CameraSummary]:
    """Processa todas as `sources` concorrentemente (1 thread leitora por
    camera), compartilhando UM `SigLIPEmbedder` e UM `BatchDispatcher` entre
    elas -- e isso que de fato lota frames de cameras diferentes no mesmo
    `encode_images()`, em vez de serializar video a video como `run_index`
    faria se chamado 8 vezes.

    `_gpu_lock` (reaproveitado de `pipeline.py`) e tomado so por chamada de
    GPU (um flush do dispatcher, uma chamada de `describe()`) -- nao pela
    duracao inteira desta funcao -- entao a leitura/motion-filter (CPU) das
    N cameras progride em paralelo enquanto a GPU processa um lote por vez.

    Os modelos (SigLIP e Qwen2.5-VL) sao obtidos via ModelManager e nao
    sao deletados ao final — o gerenciamento do ciclo de vida e
    responsabilidade do ModelManager (ou do chamador via unload_models()).
    """
    config = config or MultiIndexConfig()

    embedder = _make_embedder()
    describer = _make_describer()

    motion_filter = MotionFilter(config.motion_detection)
    embedding_filter = EmbeddingSimilarityFilter(config.embedding_filter)

    batch_size = config.batch.batch_size if config.batch.enabled else 1
    timeout_ms = config.batch.timeout_ms if config.batch.enabled else 0
    dispatcher: BatchDispatcher = BatchDispatcher(
        process_fn=embedder.encode_images,
        batch_size=batch_size,
        timeout_ms=timeout_ms,
        gpu_lock=_gpu_lock,
    )
    dispatcher.start()

    try:
        with ThreadPoolExecutor(max_workers=len(sources), thread_name_prefix="camera") as executor:
            futures = [
                executor.submit(
                    _process_camera,
                    source,
                    interval,
                    motion_filter,
                    embedding_filter,
                    dispatcher,
                    describer,
                    on_progress,
                )
                for source in sources
            ]
            summaries = [future.result() for future in futures]
    finally:
        dispatcher.stop()

    return summaries
