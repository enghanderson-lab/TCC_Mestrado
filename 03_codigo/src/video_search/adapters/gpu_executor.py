"""Executor dedicado da GPU com fila de prioridade.

Substitui o antigo `_gpu_lock` global: todas as chamadas de modelo na GPU
(SigLIP, Qwen) passam por uma única thread, que é a dona da GPU. Isso
serializa o acesso à VRAM como antes, mas com prioridade — uma busca
(`SEARCH`) passa na frente dos lotes de indexação (`INDEXING`) que estiverem
na fila, então buscar durante uma indexação espera no máximo o lote em
execução, e não a fila inteira.
"""

import itertools
import logging
import queue
import threading
from concurrent.futures import Future
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

_STOP = float("inf")  # prioridade do sentinela: roda depois de tudo que já está na fila


class GpuExecutor:
    SEARCH = 0
    INDEXING = 10

    def __init__(self) -> None:
        self._queue: "queue.PriorityQueue" = queue.PriorityQueue()
        # Desempate FIFO entre itens de mesma prioridade (e evita comparar funções).
        self._seq = itertools.count()
        self._thread = threading.Thread(target=self._loop, name="gpu-executor", daemon=True)
        self._thread.start()

    def run(self, fn: Callable[..., T], *args, priority: int = INDEXING, **kwargs) -> T:
        """Executa `fn(*args, **kwargs)` na thread da GPU e retorna o
        resultado (ou relança a exceção). Chamadas a partir da própria thread
        da GPU executam direto, para não haver deadlock."""
        if threading.current_thread() is self._thread:
            return fn(*args, **kwargs)
        future: Future = Future()
        self._queue.put((priority, next(self._seq), fn, args, kwargs, future))
        return future.result()

    def shutdown(self) -> None:
        """Termina a thread após executar o que já está enfileirado."""
        if self._thread.is_alive():
            self._queue.put((_STOP, next(self._seq), None, (), {}, None))
            self._thread.join()

    def _loop(self) -> None:
        while True:
            _, _, fn, args, kwargs, future = self._queue.get()
            if fn is None:
                return
            try:
                future.set_result(fn(*args, **kwargs))
            except BaseException as exc:  # noqa: BLE001 -- repassado ao chamador
                future.set_exception(exc)
