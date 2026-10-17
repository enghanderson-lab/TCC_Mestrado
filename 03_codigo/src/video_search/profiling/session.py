"""Sessão de profiling: cria o contexto, amostra hardware e gera o
relatório ao final, só quando `profiling_dir` é fornecido."""

from contextlib import contextmanager
from typing import Iterator, Optional

from .context import ProfilingContext
from .hardware import HardwareMonitor
from .report import generate_report


@contextmanager
def profiling_session(label: str, profiling_dir: Optional[str]) -> Iterator[Optional[ProfilingContext]]:
    """Gera `<profiling_dir>/profiling_<ts>_<label>.{txt,png}` se o bloco
    terminar sem exceção. Com `profiling_dir=None`, entrega None — os
    serviços não medem nada (custo zero)."""
    if profiling_dir is None:
        yield None
        return
    profiler = ProfilingContext()
    monitor = HardwareMonitor().start()
    try:
        yield profiler
    finally:
        monitor.stop()
    generate_report(profiler, label=label, hardware=monitor.summary(), profiling_dir=profiling_dir)
