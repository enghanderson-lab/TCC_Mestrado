"""Monitor de hardware: amostragem periódica de CPU, RAM, VRAM e uso de GPU.

Roda em thread daemon — `start()` / `stop()` delimitam o período de coleta.
Degrada graciosamente quando psutil ou nvidia-ml-py não estão instalados.

VRAM é lida via `torch.cuda.memory_allocated()` (sem dep extra).
Utilização GPU % requer nvidia-ml-py (`pip install nvidia-ml-py`).
CPU e RAM requerem psutil (`pip install psutil`).
"""

import threading
import time
from dataclasses import dataclass, field
from typing import List

try:
    import psutil as _psutil
    _PROC = _psutil.Process()
    PSUTIL_AVAILABLE = True
except ImportError:
    _psutil = None
    _PROC = None
    PSUTIL_AVAILABLE = False

try:
    import pynvml as _pynvml
    _pynvml.nvmlInit()
    _NVML_HANDLE = _pynvml.nvmlDeviceGetHandleByIndex(0)
    PYNVML_AVAILABLE = True
except Exception:
    _pynvml = None
    _NVML_HANDLE = None
    PYNVML_AVAILABLE = False


@dataclass
class HardwareSample:
    ts: float
    cpu_pct: float
    ram_mb: float
    vram_mb: float
    gpu_util_pct: float


@dataclass
class HardwareSummary:
    cpu_mean: float = 0.0
    cpu_peak: float = 0.0
    ram_mean_mb: float = 0.0
    ram_peak_mb: float = 0.0
    vram_mean_mb: float = 0.0
    vram_peak_mb: float = 0.0
    gpu_util_mean: float = 0.0
    gpu_util_peak: float = 0.0
    n_samples: int = 0
    psutil_available: bool = field(default_factory=lambda: PSUTIL_AVAILABLE)
    pynvml_available: bool = field(default_factory=lambda: PYNVML_AVAILABLE)


class HardwareMonitor:
    """Amostragem periódica de recursos de hardware em thread daemon.

    Uso::

        monitor = HardwareMonitor(interval_sec=0.5).start()
        # ... executa a pipeline ...
        monitor.stop()
        summary = monitor.summary()
    """

    def __init__(self, interval_sec: float = 0.5) -> None:
        self._interval = interval_sec
        self._samples: List[HardwareSample] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="HardwareMonitor"
        )

    def start(self) -> "HardwareMonitor":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            s = self._sample()
            with self._lock:
                self._samples.append(s)

    @staticmethod
    def _sample() -> HardwareSample:
        cpu = ram = vram = gpu = 0.0

        if PSUTIL_AVAILABLE and _PROC is not None:
            try:
                cpu = _psutil.cpu_percent(interval=None)
                ram = _PROC.memory_info().rss / 1_048_576
            except Exception:
                pass

        try:
            import torch
            if torch.cuda.is_available():
                vram = torch.cuda.memory_allocated() / 1_048_576
        except Exception:
            pass

        if PYNVML_AVAILABLE and _NVML_HANDLE is not None:
            try:
                util = _pynvml.nvmlDeviceGetUtilizationRates(_NVML_HANDLE)
                gpu = float(util.gpu)
            except Exception:
                pass

        return HardwareSample(
            ts=time.perf_counter(),
            cpu_pct=cpu,
            ram_mb=ram,
            vram_mb=vram,
            gpu_util_pct=gpu,
        )

    def summary(self) -> HardwareSummary:
        with self._lock:
            samples = list(self._samples)
        if not samples:
            return HardwareSummary(n_samples=0)
        n = len(samples)
        return HardwareSummary(
            cpu_mean=sum(s.cpu_pct for s in samples) / n,
            cpu_peak=max(s.cpu_pct for s in samples),
            ram_mean_mb=sum(s.ram_mb for s in samples) / n,
            ram_peak_mb=max(s.ram_mb for s in samples),
            vram_mean_mb=sum(s.vram_mb for s in samples) / n,
            vram_peak_mb=max(s.vram_mb for s in samples),
            gpu_util_mean=sum(s.gpu_util_pct for s in samples) / n,
            gpu_util_peak=max(s.gpu_util_pct for s in samples),
            n_samples=n,
        )
