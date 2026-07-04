"""Contexto de profiling thread-safe para a pipeline de indexação e busca.

Cada stage é medido com `with ctx.stage("nome"):` ou `ctx.add_time(nome, dt)`.
Contadores e tamanhos de batch são registrados via `ctx.count()` / `ctx.record_batch()`.
O objeto é passado opcionalmente por toda a pilha — quando None, o código de negócio
não executa nenhuma medição (custo zero).
"""

import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Generator, List


@dataclass
class StageStats:
    total_sec: float = 0.0
    calls: int = 0
    items: int = 0

    @property
    def mean_sec(self) -> float:
        return self.total_sec / self.calls if self.calls > 0 else 0.0

    @property
    def items_per_sec(self) -> float:
        return self.items / self.total_sec if (self.total_sec > 0 and self.items > 0) else 0.0


class ProfilingContext:
    """Coleta métricas durante a execução de uma pipeline.

    Thread-safe: múltiplos workers podem chamar add_time / count
    concorrentemente. O lock só é mantido para a atualização dos dicts
    (microssegundos), não durante a execução do stage em si.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._wall_start = time.perf_counter()
        self._stages: Dict[str, StageStats] = defaultdict(StageStats)
        self._counters: Dict[str, int] = defaultdict(int)
        self._batch_sizes: List[int] = []

    # ------------------------------------------------------------------
    # Medição de tempo
    # ------------------------------------------------------------------

    @contextmanager
    def stage(self, name: str, items: int = 0) -> Generator:
        """Context manager que mede o tempo de um bloco e o acumula em `name`."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add_time(name, time.perf_counter() - t0, items)

    def add_time(self, name: str, elapsed: float, items: int = 0) -> None:
        """Acumula tempo diretamente (use quando já tem o delta calculado)."""
        with self._lock:
            s = self._stages[name]
            s.total_sec += elapsed
            s.calls += 1
            s.items += items

    # ------------------------------------------------------------------
    # Contadores
    # ------------------------------------------------------------------

    def count(self, name: str, n: int = 1) -> None:
        with self._lock:
            self._counters[name] += n

    def record_batch(self, size: int) -> None:
        with self._lock:
            self._batch_sizes.append(size)

    # ------------------------------------------------------------------
    # Leitura (snapshots imutáveis)
    # ------------------------------------------------------------------

    @property
    def wall_elapsed(self) -> float:
        return time.perf_counter() - self._wall_start

    def snapshot_stages(self) -> Dict[str, StageStats]:
        with self._lock:
            return {k: StageStats(v.total_sec, v.calls, v.items)
                    for k, v in self._stages.items()}

    def snapshot_counters(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._counters)

    def snapshot_batch_sizes(self) -> List[int]:
        with self._lock:
            return list(self._batch_sizes)
