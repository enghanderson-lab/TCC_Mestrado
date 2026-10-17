"""Pipeline de indexação orientada a filas (produtor-consumidor).

Cada estágio corre em thread dedicada e se comunica via filas bounded.
Back-pressure é automático: se a GPU não acompanha, os readers pausam
sem consumir RAM extra.

Fluxo:

    VideoReader(s) ─[q_frames]─► MotionFilter ─[q_motion]─► BatchCollector
                                                                    │
                                                              [q_batches]
                                                                    │
                                                           SigLIPWorker (GPU)
                                                                    │
                                                          [q_embeddings]
                                                                    │
                                                         EmbeddingFilter ─[q_accepted]─► StoreWriter
"""

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
from PIL import Image

from ..config import EmbeddingFilterConfig, MotionDetectionConfig
from ..domain.entities import FrameRecord, IndexingResult, VideoSource
from ..domain.filters import EmbeddingSimilarityFilter, MotionFilter
from ..domain.ports import FrameSource, GpuRunner, ImageEmbedder, VectorIndex

logger = logging.getLogger(__name__)

_DONE = object()  # sentinel de fim de stream


# ---------------------------------------------------------------------------
# Configuração pública
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Parâmetros da pipeline assíncrona."""
    # Estágio GPU
    batch_size: int = 16
    batch_timeout_ms: int = 100   # flush parcial após este tempo sem itens
    # Tamanhos das filas (em itens / batches)
    frames_queue_size: int = 64   # back-pressure sobre os readers
    motion_queue_size: int = 32
    batch_queue_size: int = 4     # cada slot é um List[_FrameItem]
    embedding_queue_size: int = 32
    accepted_queue_size: int = 16


@dataclass
class StageMetrics:
    name: str
    frames_in: int = 0
    frames_out: int = 0
    frames_dropped: int = 0
    total_time_s: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, n_in: int = 1, n_out: int = 1, elapsed: float = 0.0) -> None:
        with self._lock:
            self.frames_in += n_in
            self.frames_out += n_out
            self.frames_dropped += n_in - n_out
            self.total_time_s += elapsed

    @property
    def throughput_fps(self) -> float:
        return self.frames_out / self.total_time_s if self.total_time_s > 0 else 0.0

    def summary(self) -> str:
        return (
            f"[{self.name}] in={self.frames_in} out={self.frames_out} "
            f"drop={self.frames_dropped} {self.throughput_fps:.1f}fps "
            f"t={self.total_time_s:.2f}s"
        )


# ---------------------------------------------------------------------------
# Items internos das filas
# ---------------------------------------------------------------------------

@dataclass
class _FrameItem:
    camera_id: str
    index: int
    timestamp_sec: float
    image: Image.Image
    video_name: str
    video_path: str


@dataclass
class _EmbeddingItem:
    camera_id: str
    index: int
    timestamp_sec: float
    embedding: np.ndarray
    video_name: str
    video_path: str


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class AsyncIndexPipeline:
    """Pipeline de indexação com estágios paralelos e GPU batch contínuo.

    Depende só de portas: `embedder` (ImageEmbedder), `frame_source`
    (FrameSource), `new_index` (fábrica de VectorIndex vazio) e `gpu`
    (GpuRunner, prioridade INDEXING).

    Para múltiplas câmeras: N threads leitoras alimentam a mesma fila de
    entrada; o BatchCollector maximiza o tamanho de lote efetivo combinando
    frames de câmeras diferentes num único `encode_images()`.

    Cada câmera mantém estado de MotionFilter e EmbeddingFilter independente
    via camera_id.

    Parâmetro `profiler` (Optional[ProfilingContext]): quando fornecido,
    cada worker registra timings granulares e contadores. Quando None,
    nenhuma medição é feita — custo zero no caminho crítico.
    """

    def __init__(
        self,
        embedder: ImageEmbedder,
        frame_source: FrameSource,
        new_index: Callable[[], VectorIndex],
        gpu: GpuRunner,
        config: Optional[PipelineConfig] = None,
        motion_config: Optional[MotionDetectionConfig] = None,
        embedding_config: Optional[EmbeddingFilterConfig] = None,
        profiler=None,          # Optional[ProfilingContext]
    ) -> None:
        self._embedder = embedder
        self._frame_source = frame_source
        self._new_index = new_index
        self._gpu = gpu
        self._cfg = config or PipelineConfig()
        self._motion_cfg = motion_config or MotionDetectionConfig()
        self._emb_cfg = embedding_config or EmbeddingFilterConfig()
        self._profiler = profiler

    def run(
        self,
        sources: List[VideoSource],
        interval_sec: float = 2.0,
        limit: int = 0,
        on_progress: Optional[Callable[[str, int], None]] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> List[IndexingResult]:
        """Indexa todas as fontes em paralelo.

        Args:
            sources:      uma entrada por vídeo/câmera.
            interval_sec: intervalo de amostragem entre frames (segundos).
            limit:        para após este nº de frames amostrados por câmera (0=sem limite).
            on_progress:  callback(camera_id, n_aceitos) chamado a cada frame salvo.
            should_cancel: checado periodicamente pelos leitores; quando
                           retorna True, a leitura para e o pipeline drena e
                           encerra normalmente (sem exceção).

        Returns:
            Um IndexingResult por câmera; os índices já estão salvos em disco.
        """
        if not sources:
            return []

        cfg = self._cfg
        n = len(sources)
        p = self._profiler  # atalho; None → sem profiling

        # Filas bounded entre estágios
        q_frames: queue.Queue = queue.Queue(maxsize=cfg.frames_queue_size)
        q_motion: queue.Queue = queue.Queue(maxsize=cfg.motion_queue_size)
        q_batches: queue.Queue = queue.Queue(maxsize=cfg.batch_queue_size)
        q_embeddings: queue.Queue = queue.Queue(maxsize=cfg.embedding_queue_size)
        q_accepted: queue.Queue = queue.Queue(maxsize=cfg.accepted_queue_size)

        metrics = {name: StageMetrics(name=name)
                   for name in ("reader", "motion", "batch", "siglip", "emb_filter", "store")}
        results: Dict[str, IndexingResult] = {
            s.camera_id: IndexingResult(camera_id=s.camera_id, output_dir=s.output_dir)
            for s in sources
        }
        stores: Dict[str, VectorIndex] = {s.camera_id: self._new_index() for s in sources}
        results_lock = threading.Lock()
        done_event = threading.Event()
        error_slot: List[Optional[BaseException]] = [None]

        motion_filter = MotionFilter(self._motion_cfg)
        emb_filter = EmbeddingSimilarityFilter(self._emb_cfg)

        workers = [
            *(threading.Thread(
                target=self._reader_worker,
                args=(src, interval_sec, limit, q_frames, metrics["reader"],
                      results, results_lock, error_slot, p, should_cancel),
                name=f"reader-{src.camera_id}",
                daemon=True,
            ) for src in sources),
            threading.Thread(
                target=self._motion_worker,
                args=(n, motion_filter, q_frames, q_motion, metrics["motion"],
                      results, results_lock, error_slot, p),
                name="motion",
                daemon=True,
            ),
            threading.Thread(
                target=self._batch_collector_worker,
                args=(q_motion, q_batches, metrics["batch"], error_slot, p),
                name="batch-collector",
                daemon=True,
            ),
            threading.Thread(
                target=self._siglip_worker,
                args=(q_batches, q_embeddings, metrics["siglip"], error_slot, should_cancel),
                name="siglip",
                daemon=True,
            ),
            threading.Thread(
                target=self._emb_filter_worker,
                args=(emb_filter, q_embeddings, q_accepted, metrics["emb_filter"],
                      results, results_lock, error_slot, p),
                name="emb-filter",
                daemon=True,
            ),
            threading.Thread(
                target=self._store_worker,
                args=(q_accepted, stores, metrics["store"],
                      results, results_lock, on_progress, done_event, error_slot, p),
                name="store-writer",
                daemon=True,
            ),
        ]

        for t in workers:
            t.start()

        done_event.wait()

        for t in workers:
            t.join(timeout=5.0)

        if error_slot[0] is not None:
            raise RuntimeError("pipeline de indexação falhou") from error_slot[0]

        for camera_id, store in stores.items():
            t0 = time.perf_counter()
            store.save(results[camera_id].output_dir)
            if p:
                p.add_time("store_save", time.perf_counter() - t0)
            results[camera_id].frames_accepted = len(store)
            if p:
                p.count("frames_indexed", len(store))

        for m in metrics.values():
            logger.info(m.summary())

        return list(results.values())

    # ------------------------------------------------------------------
    # Workers
    # ------------------------------------------------------------------

    def _reader_worker(
        self,
        source: VideoSource,
        interval_sec: float,
        limit: int,
        q_out: queue.Queue,
        metrics: StageMetrics,
        results: Dict[str, IndexingResult],
        results_lock: threading.Lock,
        error_slot: List,
        profiler,  # Optional[ProfilingContext]
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> None:
        """Lê frames do vídeo e os enfileira, medindo extração separada da fila."""
        video_name = Path(source.video_path).name
        video_path = str(Path(source.video_path).resolve())
        count = 0
        try:
            gen = iter(self._frame_source.iter_frames(source.video_path, interval_sec=interval_sec))
            while True:
                # Mede apenas a extração do frame (next() sobre o gerador)
                t_extract = time.perf_counter()
                try:
                    frame = next(gen)
                except StopIteration:
                    break
                if profiler:
                    profiler.add_time("frame_extract", time.perf_counter() - t_extract, 1)

                if error_slot[0] is not None:
                    break
                if limit and frame.index >= limit:
                    break
                if should_cancel is not None and should_cancel():
                    logger.info("indexação de %s cancelada", source.camera_id)
                    break

                t0 = time.monotonic()
                q_out.put(_FrameItem(
                    camera_id=source.camera_id,
                    index=frame.index,
                    timestamp_sec=frame.timestamp_sec,
                    image=frame.image,
                    video_name=video_name,
                    video_path=video_path,
                ))
                count += 1
                metrics.record(1, 1, time.monotonic() - t0)
                if profiler:
                    profiler.count("frames_read")
        except Exception as exc:
            logger.exception("reader %s falhou", source.camera_id)
            error_slot[0] = exc
        finally:
            with results_lock:
                results[source.camera_id].frames_read = count
            q_out.put(_DONE)

    @staticmethod
    def _motion_worker(
        n_readers: int,
        motion_filter: MotionFilter,
        q_in: queue.Queue,
        q_out: queue.Queue,
        metrics: StageMetrics,
        results: Dict[str, IndexingResult],
        results_lock: threading.Lock,
        error_slot: List,
        profiler,
    ) -> None:
        """Aplica motion-filter; descarta frames estáticos antes da GPU."""
        done_count = 0
        try:
            while True:
                item = q_in.get()
                if item is _DONE:
                    done_count += 1
                    if done_count == n_readers:
                        q_out.put(_DONE)
                        return
                    continue
                if error_slot[0] is not None:
                    continue

                t0 = time.monotonic()
                t_p = time.perf_counter()
                accepted = motion_filter.should_accept(item.camera_id, item.image)
                elapsed = time.monotonic() - t0

                if profiler:
                    profiler.add_time("motion_filter", time.perf_counter() - t_p, 1)

                if accepted:
                    q_out.put(item)
                    metrics.record(1, 1, elapsed)
                else:
                    with results_lock:
                        results[item.camera_id].frames_dropped_motion += 1
                    metrics.record(1, 0, elapsed)
                    if profiler:
                        profiler.count("frames_motion_dropped")
        except Exception as exc:
            logger.exception("motion-worker falhou")
            error_slot[0] = exc
            q_out.put(_DONE)

    def _batch_collector_worker(
        self,
        q_in: queue.Queue,
        q_out: queue.Queue,
        metrics: StageMetrics,
        error_slot: List,
        profiler,
    ) -> None:
        """Acumula itens em lotes de até batch_size OU até batch_timeout_ms."""
        batch_size = self._cfg.batch_size
        timeout_s = self._cfg.batch_timeout_ms / 1000.0
        try:
            while True:
                first = q_in.get()
                if first is _DONE:
                    q_out.put(_DONE)
                    return

                batch: List[_FrameItem] = [first]
                deadline = time.monotonic() + timeout_s
                t0 = time.monotonic()
                done_seen = False

                while len(batch) < batch_size:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        item = q_in.get(timeout=remaining)
                        if item is _DONE:
                            done_seen = True
                            break
                        batch.append(item)
                    except queue.Empty:
                        break

                q_out.put(batch)
                elapsed = time.monotonic() - t0
                metrics.record(len(batch), len(batch), elapsed)
                if profiler:
                    profiler.add_time("batch_collect", elapsed, len(batch))
                    profiler.record_batch(len(batch))
                logger.debug("batch lote=%d", len(batch))

                if done_seen:
                    q_out.put(_DONE)
                    return
        except Exception as exc:
            logger.exception("batch-collector falhou")
            error_slot[0] = exc
            q_out.put(_DONE)

    def _siglip_worker(
        self,
        q_in: queue.Queue,
        q_out: queue.Queue,
        metrics: StageMetrics,
        error_slot: List,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> None:
        """Processa lotes de imagens na GPU com SigLIP.

        Após um cancelamento, os lotes ainda enfileirados são descartados em
        vez de ocupar a GPU (o leitor já parou; só falta drenar as filas).

        Timings granulares (preprocess vs infer) são delegados ao embedder
        quando um profiler está presente — o embedder recebe `profiler=self._profiler`
        em encode_images().
        """
        p = self._profiler
        try:
            while True:
                batch = q_in.get()
                if batch is _DONE:
                    q_out.put(_DONE)
                    return
                if error_slot[0] is not None:
                    continue
                if should_cancel is not None and should_cancel():
                    metrics.record(len(batch), 0)
                    continue

                t0 = time.monotonic()
                all_embs = self._gpu.run(
                    self._embedder.encode_images,
                    [item.image for item in batch],
                    profiler=p,
                    priority=self._gpu.INDEXING,
                )
                elapsed = time.monotonic() - t0

                for item, emb in zip(batch, all_embs):
                    q_out.put(_EmbeddingItem(
                        camera_id=item.camera_id,
                        index=item.index,
                        timestamp_sec=item.timestamp_sec,
                        embedding=emb,
                        video_name=item.video_name,
                        video_path=item.video_path,
                    ))
                metrics.record(len(batch), len(batch), elapsed)
                logger.debug("siglip lote=%d t=%.3fs", len(batch), elapsed)
        except Exception as exc:
            logger.exception("siglip-worker falhou")
            error_slot[0] = exc
            q_out.put(_DONE)

    @staticmethod
    def _emb_filter_worker(
        emb_filter: EmbeddingSimilarityFilter,
        q_in: queue.Queue,
        q_out: queue.Queue,
        metrics: StageMetrics,
        results: Dict[str, IndexingResult],
        results_lock: threading.Lock,
        error_slot: List,
        profiler,
    ) -> None:
        """Descarta embeddings redundantes (similaridade ≥ threshold com último aceito)."""
        try:
            while True:
                item = q_in.get()
                if item is _DONE:
                    q_out.put(_DONE)
                    return
                if error_slot[0] is not None:
                    continue

                t0 = time.monotonic()
                t_p = time.perf_counter()
                accepted = emb_filter.should_accept(item.camera_id, item.embedding)
                elapsed = time.monotonic() - t0

                if profiler:
                    profiler.add_time("emb_filter", time.perf_counter() - t_p, 1)

                if accepted:
                    q_out.put(item)
                    metrics.record(1, 1, elapsed)
                else:
                    with results_lock:
                        results[item.camera_id].frames_dropped_similarity += 1
                    metrics.record(1, 0, elapsed)
                    if profiler:
                        profiler.count("frames_sim_dropped")
        except Exception as exc:
            logger.exception("emb-filter falhou")
            error_slot[0] = exc
            q_out.put(_DONE)

    @staticmethod
    def _store_worker(
        q_in: queue.Queue,
        stores: Dict[str, VectorIndex],
        metrics: StageMetrics,
        results: Dict[str, IndexingResult],
        results_lock: threading.Lock,
        on_progress: Optional[Callable[[str, int], None]],
        done_event: threading.Event,
        error_slot: List,
        profiler,
    ) -> None:
        """Grava embeddings aceitos no store de cada câmera."""
        try:
            while True:
                item = q_in.get()
                if item is _DONE:
                    return

                t0 = time.monotonic()
                t_p = time.perf_counter()
                record = FrameRecord(
                    index=item.index,
                    timestamp_sec=item.timestamp_sec,
                    video=item.video_name,
                    video_path=item.video_path,
                )
                stores[item.camera_id].add(item.embedding, record)
                elapsed = time.monotonic() - t0

                if profiler:
                    profiler.add_time("store_write", time.perf_counter() - t_p, 1)

                metrics.record(1, 1, elapsed)

                if on_progress is not None:
                    on_progress(item.camera_id, len(stores[item.camera_id]))
        except Exception as exc:
            logger.exception("store-writer falhou")
            error_slot[0] = exc
        finally:
            done_event.set()
