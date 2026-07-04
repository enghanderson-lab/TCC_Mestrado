"""Gerador de relatório de profiling em texto + gráfico.

Chamado automaticamente ao final de run_multi_index() e run_search()
quando `profiling_dir` é fornecido. Salva:
    <profiling_dir>/profiling_<timestamp>_<label>.txt
    <profiling_dir>/profiling_<timestamp>_<label>.png

Se `profiling_dir` for None, apenas imprime o relatório via logger.
"""

import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Optional

from .context import ProfilingContext
from .hardware import HardwareSummary

logger = logging.getLogger(__name__)

# Mapa stage → categoria (igual ao chart.py, sem importar de lá para evitar
# dep circular quando matplotlib não está instalado)
_STAGE_CATEGORY = {
    "frame_extract":        "IO",
    "store_save":           "IO",
    "load_store":           "IO",
    "frame_extract_search": "IO",
    "siglip_preprocess":    "GPU",
    "siglip_infer":         "GPU",
    "siglip_text_encode":   "GPU",
    "qwen_preprocess":      "GPU",
    "qwen_infer":           "GPU",
    "confidence_score":     "GPU",
    "faiss_search":         "GPU",
    "motion_filter":        "CPU",
    "emb_filter":           "CPU",
    "mmr_rerank":           "CPU",
    "batch_collect":        "CPU",
    "cache_lookup":         "Cache",
    "cache_store":          "Cache",
    "store_write":          "Cache",
}

# Nomes amigáveis
_STAGE_LABEL = {
    "frame_extract":        "Extração de frames",
    "store_save":           "Salvar índice em disco",
    "load_store":           "Carregar índice (FAISS)",
    "frame_extract_search": "Extrair frames para Qwen",
    "siglip_preprocess":    "SigLIP — pré-processamento",
    "siglip_infer":         "SigLIP — inferência GPU",
    "siglip_text_encode":   "SigLIP — encode texto",
    "qwen_preprocess":      "Qwen — pré-processamento",
    "qwen_infer":           "Qwen — inferência GPU",
    "confidence_score":     "Score de confiança (ST)",
    "faiss_search":         "Busca vetorial (FAISS)",
    "motion_filter":        "Motion Filter",
    "emb_filter":           "Embedding Filter",
    "mmr_rerank":           "MMR Reranker",
    "batch_collect":        "Montagem de batch",
    "cache_lookup":         "Cache — leitura (SQLite)",
    "cache_store":          "Cache — escrita (SQLite)",
    "store_write":          "Store — add embedding",
}

_BAR_WIDTH = 30


def _bar(fraction: float, width: int = _BAR_WIDTH) -> str:
    filled = round(fraction * width)
    return "█" * filled + "░" * (width - filled)


def _mb(mb: float) -> str:
    if mb >= 1024:
        return f"{mb / 1024:.1f} GB"
    return f"{mb:.0f} MB"


def generate_report(
    ctx: ProfilingContext,
    label: str = "indexing",
    hardware: Optional[HardwareSummary] = None,
    profiling_dir: Optional[str] = None,
) -> str:
    """Gera o relatório completo de profiling.

    Args:
        ctx:           contexto com timings e contadores coletados.
        label:         "indexing" ou "search" — aparece no nome do arquivo.
        hardware:      resumo do HardwareMonitor (None se não usado).
        profiling_dir: pasta onde salvar .txt e .png (None → só log).

    Returns:
        O texto do relatório (string).
    """
    stages     = ctx.snapshot_stages()
    counters   = ctx.snapshot_counters()
    batch_sizes = ctx.snapshot_batch_sizes()
    wall       = ctx.wall_elapsed

    # Ordena stages por tempo decrescente
    sorted_stages = sorted(stages.items(), key=lambda kv: kv[1].total_sec, reverse=True)

    # ------------------------------------------------------------------ #
    # Monta texto
    # ------------------------------------------------------------------ #
    sep  = "═" * 62
    sep2 = "─" * 62
    lines = [
        "",
        f"╔{sep}╗",
        f"║  PROFILING — {label.upper():<46}║",
        f"╠{sep}╣",
    ]

    # Headline
    frames_indexed = counters.get("frames_indexed", 0)
    fps = frames_indexed / wall if wall > 0 and frames_indexed > 0 else 0.0
    lines += [
        f"║  Tempo total : {wall:>8.2f}s{'':<35}║",
        f"║  Frames proc.: {frames_indexed:>8d}    FPS efetivo: {fps:>6.2f}{'':<14}║",
        f"╚{sep}╝",
        "",
        "TEMPO POR ETAPA",
        sep2,
    ]

    total_measured = sum(s.total_sec for _, s in sorted_stages)

    for name, stats in sorted_stages:
        if stats.total_sec < 0.001:
            continue
        pct = stats.total_sec / wall * 100 if wall > 0 else 0.0
        frac = min(stats.total_sec / wall, 1.0) if wall > 0 else 0.0
        label_str = _STAGE_LABEL.get(name, name)
        cat = _STAGE_CATEGORY.get(name, "?")
        throughput = ""
        if stats.items > 0 and stats.total_sec > 0:
            throughput = f"  {stats.items_per_sec:>6.1f} it/s"
        lines.append(
            f"  {label_str:<34}  {stats.total_sec:>7.3f}s  {pct:>5.1f}%  [{cat}]{throughput}"
        )

    unmeasured = max(0.0, wall - total_measured)
    if unmeasured > 0.5:
        lines.append(
            f"  {'(overhead / fila / outros)':<34}  {unmeasured:>7.3f}s  "
            f"{unmeasured / wall * 100:>5.1f}%"
        )

    # Gráfico ASCII inline (top 8)
    lines += ["", "DISTRIBUIÇÃO (top 8)", sep2]
    top8 = [(n, s) for n, s in sorted_stages if s.total_sec >= 0.001][:8]
    for name, stats in top8:
        frac = min(stats.total_sec / wall, 1.0) if wall > 0 else 0.0
        pct = stats.total_sec / wall * 100 if wall > 0 else 0.0
        short = _STAGE_LABEL.get(name, name)[:28]
        lines.append(f"  {short:<28}  {_bar(frac, 20)}  {pct:>5.1f}%")

    # Contadores
    lines += ["", "CONTADORES", sep2]
    frames_read   = counters.get("frames_read", frames_indexed)
    drop_motion   = counters.get("frames_motion_dropped", 0)
    drop_sim      = counters.get("frames_sim_dropped", 0)
    cache_hits    = counters.get("cache_hits", 0)
    cache_misses  = counters.get("cache_misses", 0)
    qwen_calls    = counters.get("qwen_calls", 0)
    cache_total   = cache_hits + cache_misses
    cache_pct     = cache_hits / cache_total * 100 if cache_total > 0 else 0.0
    motion_pct    = drop_motion / frames_read * 100 if frames_read > 0 else 0.0
    sim_pct       = drop_sim / frames_read * 100 if frames_read > 0 else 0.0

    lines += [
        f"  Frames lidos              : {frames_read:>6d}",
        f"  Descartados — motion      : {drop_motion:>6d}  ({motion_pct:.1f}%)",
        f"  Descartados — similaridade: {drop_sim:>6d}  ({sim_pct:.1f}%)",
        f"  Embeddings indexados      : {frames_indexed:>6d}",
        f"  Cache hits                : {cache_hits:>6d}  ({cache_pct:.1f}% do total)",
        f"  Cache misses              : {cache_misses:>6d}",
        f"  Chamadas ao Qwen          : {qwen_calls:>6d}",
    ]

    # Qwen: tempo médio por chamada
    if qwen_calls > 0 and "qwen_infer" in stages and stages["qwen_infer"].total_sec > 0:
        mean_qwen = stages["qwen_infer"].total_sec / stages["qwen_infer"].calls
        lines.append(f"  Tempo médio por batch Qwen : {mean_qwen:.3f}s")

    # Batches
    if batch_sizes:
        mean_b = sum(batch_sizes) / len(batch_sizes)
        lines += [
            "",
            "BATCHES (SigLIP)",
            sep2,
            f"  Total: {len(batch_sizes)}  |  Médio: {mean_b:.1f}  |  "
            f"Mín: {min(batch_sizes)}  |  Máx: {max(batch_sizes)}",
        ]

    # Hardware
    if hardware is not None and hardware.n_samples > 0:
        lines += ["", "HARDWARE", sep2]
        lines += [
            f"  N amostras (intervalo 0,5 s): {hardware.n_samples}",
        ]
        if hardware.psutil_available:
            lines += [
                f"  CPU  : médio={hardware.cpu_mean:>5.1f}%  pico={hardware.cpu_peak:>5.1f}%",
                f"  RAM  : médio={_mb(hardware.ram_mean_mb):>9}  pico={_mb(hardware.ram_peak_mb):>9}",
            ]
        else:
            lines.append("  CPU/RAM: psutil não instalado  (pip install psutil)")
        lines.append(
            f"  VRAM : médio={_mb(hardware.vram_mean_mb):>9}  pico={_mb(hardware.vram_peak_mb):>9}"
        )
        if hardware.pynvml_available:
            lines.append(
                f"  GPU  : médio={hardware.gpu_util_mean:>5.1f}%  pico={hardware.gpu_util_peak:>5.1f}%"
            )
        else:
            lines.append("  GPU% : nvidia-ml-py não instalado  (pip install nvidia-ml-py)")

    # Throughput
    lines += ["", "THROUGHPUT", sep2]
    if "siglip_infer" in stages and stages["siglip_infer"].items > 0:
        lines.append(
            f"  SigLIP infer: {stages['siglip_infer'].items_per_sec:.1f} frames/s"
        )
    if "qwen_infer" in stages and stages["qwen_infer"].items > 0:
        lines.append(
            f"  Qwen infer  : {stages['qwen_infer'].items_per_sec:.1f} frames/s"
        )
    if fps > 0:
        lines.append(f"  Pipeline E2E: {fps:.2f} frames/s aceitos/s")

    lines += ["", sep2, ""]
    text = "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Salva / loga
    # ------------------------------------------------------------------ #
    logger.info(text)

    if profiling_dir is not None:
        out_dir = Path(profiling_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        txt_path = out_dir / f"profiling_{ts}_{label}.txt"
        png_path = out_dir / f"profiling_{ts}_{label}.png"

        txt_path.write_text(text, encoding="utf-8")
        logger.info("[profiling] relatório salvo em %s", txt_path)

        try:
            from .chart import save_chart
            stage_secs = {n: s.total_sec for n, s in stages.items() if s.total_sec > 0}
            save_chart(stage_secs, wall, png_path, title=f"Tempo por etapa — {label}")
            logger.info("[profiling] gráfico salvo em %s", png_path)
        except Exception as exc:
            logger.warning("[profiling] não foi possível gerar o gráfico: %s", exc)

    return text
