"""Dispatcher de batches entre multiplas fontes concorrentes: acumula itens
de uma fila compartilhada e os processa em lotes (uma chamada de
`process_fn` por lote), maximizando o tamanho de lote efetivo quando ha
varias fontes produzindo itens ao mesmo tempo (ex.: N videos/cameras sendo
lidos em paralelo, ver `multi_index.py`).

Desacoplado de SigLIP/imagens de proposito: `process_fn` e injetada, o que
permite testar com fakes (sem GPU/cv2) e mante-lo reutilizavel."""

import logging
import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Callable, Generic, List, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class _Item(Generic[T, R]):
    payload: T
    future: "Future" = field(default_factory=Future)


class BatchDispatcher(Generic[T, R]):
    """Uma instancia = uma thread dedicada que drena uma `queue.Queue`
    compartilhada, despachando lotes para `process_fn`. Os "produtores"
    (ex.: threads leitoras de N videos) chamam `.submit(payload)` e recebem
    um `Future` que resolve quando o lote daquele item for processado."""

    def __init__(
        self,
        process_fn: Callable[[List[T]], List[R]],
        batch_size: int = 16,
        timeout_ms: int = 200,
        gpu_lock: Optional[threading.Lock] = None,
    ) -> None:
        self._process_fn = process_fn
        self._batch_size = max(1, batch_size)
        self._timeout_s = timeout_ms / 1000.0
        self._gpu_lock = gpu_lock or threading.Lock()
        self._queue: "queue.Queue[_Item]" = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="batch-dispatcher", daemon=True)
        self._thread.start()

    def submit(self, payload: T) -> "Future":
        """Chamado pelas threads produtoras. Nao bloqueia: enfileira e
        retorna um `Future` que o chamador resolve com `.result()`."""
        item: _Item = _Item(payload=payload)
        self._queue.put(item)
        return item.future

    def stop(self) -> None:
        """Sinaliza parada; a thread de batching drena o que resta na fila
        antes de terminar (nenhum item enfileirado e perdido)."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()

    def _collect_batch(self) -> List[_Item]:
        """Bloqueia (com poll curto, para `stop()` ser responsivo) ate o
        primeiro item da fila; depois acumula mais itens ate `batch_size`
        OU até `timeout_ms` desde a chegada do primeiro item, o que ocorrer
        primeiro. Retorna lista vazia se parado e a fila estiver vazia."""
        batch: List[_Item] = []
        while not batch:
            try:
                first = self._queue.get(timeout=0.1)
            except queue.Empty:
                if self._stop_event.is_set():
                    return []
                continue
            batch.append(first)

        deadline = time.monotonic() + self._timeout_s
        while len(batch) < self._batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                batch.append(self._queue.get(timeout=remaining))
            except queue.Empty:
                break
        return batch

    def _run(self) -> None:
        while not (self._stop_event.is_set() and self._queue.empty()):
            batch = self._collect_batch()
            if not batch:
                continue
            t0 = time.monotonic()
            try:
                with self._gpu_lock:
                    results = self._process_fn([item.payload for item in batch])
                for item, result in zip(batch, results):
                    item.future.set_result(result)
            except Exception as exc:  # um lote ruim nao deve travar o dispatcher
                for item in batch:
                    item.future.set_exception(exc)
            logger.info("lote executado tamanho=%d tempo=%.3fs", len(batch), time.monotonic() - t0)
