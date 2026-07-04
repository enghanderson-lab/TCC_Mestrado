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
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from PIL import Image

from ..media.frame_extractor import extract_frames
from ..media.motion_filter import MotionFilter
from .embedding_filter import EmbeddingSimilarityFilter
from .embedding_store import EmbeddingStore, FrameRecord
from .faiss_store import FAISS_AVAILABLE, FaissStore
from ..utils.config import EmbeddingFilterConfig, MotionDetectionConfig

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
    # Store: FAISS se disponível, numpy legacy caso contrário
    use_faiss: bool = True


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


@dataclass
class PipelineSource:
    camera_id: str
    video_path: str
    output_dir: str


@dataclass
class PipelineResult:
    camera_id: str
    output_dir: str
    frames_read: int = 0
    frames_accepted: int = 0
    frames_dropped_motion: int = 0
    frames_dropped_similarity: int = 0
    metrics: Dict[str, "StageMetrics"] = field(default_factory=dict)


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

    Para múltiplas câmeras: N threads leitoras alimentam a mesma fila de
    entrada; o BatchCollector maximiza o tamanho de lote efetivo combinando
    frames de câmeras diferentes num único `encode_images()`.

    Cada câmera mantém estado de MotionFilter e EmbeddingFilter independente
    via camera_id.

    Parâmetro `profiler` (Optional[ProfilingContext]): quando fornecido,
    cada worker registra timings granulares e contadores de cache. Quando None,
    nenhuma medição é feita — custo zero no caminho crítico.
    """

    def __init__(
        self,
        embedder,
        config: Optional[PipelineConfig] = None,
        motion_config: Optional[MotionDetectionConfig] = None,
        embedding_config: Optional[EmbeddingFilterConfig] = None,
        gpu_lock: Optional[threading.Lock] = None,
        embedding_cache=None,   # Optional[EmbeddingCache]
        profiler=None,          # Optional[ProfilingContext]
    ) -> None:
        self._embedder = embedder
        self._cfg = config or PipelineConfig()
        self._motion_cfg = motion_config or MotionDetectionConfig()
        self._emb_cfg = embedding_config or EmbeddingFilterConfig()
        self._gpu_lock = gpu_lock or threading.Lock()
        self._cache = embedding_cache
        self._profiler = profiler

    def run(
        self,
        sources: List[PipelineSource],
        interval_sec: float = 2.0,
        limit: int = 0,
        on_progress: Optional[Callable[[str, int], None]] = None,
    ) -> List[PipelineResult]:
        """Indexa todas as fontes em paralelo.

        Args:
            sources:      uma entrada por vídeo/câmera.
            interval_sec: intervalo de amostragem entre frames (segundos).
            limit:        para após este nº de frames amostrados por câmera (0=sem limite).
            on_progress:  callback(camera_id, n_aceitos) chamado a cada frame salvo.

        Returns:
            Uma PipelineResult por câmera; os indexes já estão salvos em disco.
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
        results: Dict[str, PipelineResult] = {
            s.camera_id: PipelineResult(camera_id=s.camera_id, output_dir=s.output_dir)
            for s in sources
        }
        use_faiss = cfg.use_faiss and FAISS_AVAILABLE
        stores: Dict[str, Any] = {
            s.camera_id: (FaissStore() if use_faiss else EmbeddingStore())
            for s in sources
        }
        results_lock = threading.Lock()
        done_event = threading.Event()
        error_slot: List[Optional[BaseException]] = [None]

        motion_filter = MotionFilter(self._motion_cfg)
        emb_filter = EmbeddingSimilarityFilter(self._emb_cfg)

        workers = [
            *(threading.Thread(
                target=self._reader_worker,
                args=(src, interval_sec, limit, q_frames, metrics["reader"],
                      results, results_lock, error_slot, p),
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
                args=(q_batches, q_embeddings, metrics["siglip"], error_slot),
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
            results[camera_id].metrics = dict(metrics)
            if p:
                p.count("frames_indexed", len(store))

        for m in metrics.values():
            logger.info(m.summary())

        return list(results.values())

    # ------------------------------------------------------------------
    # Workers
    # ------------------------------------------------------------------

    @staticmethod
    def _reader_worker(
        source: PipelineSource,
        interval_sec: float,
        limit: int,
        q_out: queue.Queue,
        metrics: StageMetrics,
        results: Dict[str, PipelineResult],
        results_lock: threading.Lock,
        error_slot: List,
        profiler,  # Optional[ProfilingContext]
    ) -> None:
        """Lê frames do vídeo e os enfileira, medindo extração separada da fila."""
        video_name = Path(source.video_path).name
        video_path = str(Path(source.video_path).resolve())
        count = 0
        try:
            gen = iter(extract_frames(source.video_path, interval_sec=interval_sec))
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
        results: Dict[str, PipelineResult],
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
    ) -> None:
        """Processa lotes de imagens na GPU com SigLIP.

        Timings granulares (preprocess vs infer) são delegados ao embedder
        quando um profiler está presente — o embedder recebe `profiler=self._profiler`
        em encode_images().
        """
        cache = self._cache
        p = self._profiler
        try:
            while True:
                batch = q_in.get()
                if batch is _DONE:
                    q_out.put(_DONE)
                    return
                if error_slot[0] is not None:
                    continue

                t0 = time.monotonic()

                if cache is not None:
                    from .embedding_cache import EmbeddingCache

                    # Hash das imagens (CPU, barato)
                    t_hash = time.perf_counter()
                    hashes = [EmbeddingCache.hash_image(item.image) for item in batch]
                    if p:
                        p.add_time("cache_lookup", time.perf_counter() - t_hash, len(batch))

                    # Busca em lote no SQLite
                    t_lookup = time.perf_counter()
                    cached_map = cache.get_batch(hashes)
                    if p:
                        p.add_time("cache_lookup", time.perf_counter() - t_lookup)

                    miss_indices = [i for i, h in enumerate(hashes) if cached_map[h] is None]
                    miss_images = [batch[i].image for i in miss_indices]
                    n_hits = len(batch) - len(miss_images)

                    if p:
                        p.count("cache_hits", n_hits)
                        p.count("cache_misses", len(miss_images))

                    if miss_images:
                        with self._gpu_lock:
                            new_embs = self._embedder.encode_images(
                                miss_images, profiler=p
                            )
                        t_store = time.perf_counter()
                        for i, emb in zip(miss_indices, new_embs):
                            cache.put(hashes[i], emb)
                            cached_map[hashes[i]] = emb
                        if p:
                            p.add_time("cache_store", time.perf_counter() - t_store,
                                       len(miss_images))

                    all_embs = [cached_map[h] for h in hashes]
                    gpu_calls = len(miss_images)
                else:
                    with self._gpu_lock:
                        all_embs = list(self._embedder.encode_images(
                            [item.image for item in batch], profiler=p
                        ))
                    gpu_calls = len(batch)

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
                logger.debug("siglip lote=%d gpu_calls=%d t=%.3fs",
                             len(batch), gpu_calls, elapsed)
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
        results: Dict[str, PipelineResult],
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
        stores: Dict[str, EmbeddingStore],
        metrics: StageMetrics,
        results: Dict[str, PipelineResult],
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
