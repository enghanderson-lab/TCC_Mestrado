"""Gráfico de barras horizontais do tempo por etapa da pipeline.

Design conforme o método data-viz (palette.md):
  - Forma: barras horizontais (magnitudes com rótulos longos)
  - Cor: 4 categorias funcionais em hues CVD-safe (slots 1-4 da paleta)
  - Legenda obrigatória para ≥ 2 séries
  - Labels diretos nas pontas (regra de alívio para aqua/amarelo < 3:1)
  - Gridlines: hairline, #e1e0d9, recessivas
  - Superfície: #fcfcfb (light)

Paleta validada (palette.md — CVD worst-adjacent ΔE 24.2 light):
  slot 1 blue  #2a78d6  → IO (leitura de vídeo / disco)
  slot 2 aqua  #1baf7a  → GPU (SigLIP / Qwen — inferência)
  slot 3 yellow #eda100 → CPU (filtros / MMR / pré-proc)
  slot 4 green #008300  → Cache (SQLite read/write)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Mapeamento stage → categoria funcional
STAGE_CATEGORY: Dict[str, str] = {
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

# Paleta categorical (slots 1-4, ordem CVD-safe)
_COLORS = {
    "IO":    "#2a78d6",
    "GPU":   "#1baf7a",
    "CPU":   "#eda100",
    "Cache": "#008300",
}
_SURFACE    = "#fcfcfb"
_GRID       = "#e1e0d9"
_TEXT_PRI   = "#0b0b0b"
_TEXT_SEC   = "#52514e"
_TEXT_MUTED = "#898781"


def save_chart(
    stages: Dict[str, float],
    total_sec: float,
    output_path: Path,
    title: str = "Tempo por etapa da pipeline",
) -> None:
    """Gera e salva o gráfico de barras horizontais.

    Args:
        stages:      {stage_name: seconds} — inclui apenas stages com t > 0.
        total_sec:   tempo de parede total (para calcular %).
        output_path: caminho do arquivo .png a ser salvo.
        title:       título do gráfico.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        return  # matplotlib não instalado → pula silenciosamente

    # Ordena por tempo decrescente (barra mais longa no topo)
    sorted_stages: List[Tuple[str, float, str]] = sorted(
        [
            (name, secs, STAGE_CATEGORY.get(name, "IO"))
            for name, secs in stages.items()
            if secs > 0
        ],
        key=lambda x: x[1],
        reverse=True,
    )

    if not sorted_stages:
        return

    n = len(sorted_stages)
    fig_height = max(4.5, n * 0.52 + 2.0)
    fig, ax = plt.subplots(figsize=(11, fig_height))
    fig.set_facecolor(_SURFACE)
    ax.set_facecolor(_SURFACE)

    labels  = [s[0] for s in sorted_stages]
    values  = [s[1] for s in sorted_stages]
    cats    = [s[2] for s in sorted_stages]
    colors  = [_COLORS.get(c, _COLORS["IO"]) for c in cats]

    y_pos = list(range(n))
    bars  = ax.barh(y_pos, values, height=0.55, color=colors, zorder=3)

    # Labels diretos na ponta (obrigatório para slots aqua/amarelo < 3:1)
    x_max = max(values) if values else 1.0
    for i, (val, _bar) in enumerate(zip(values, bars)):
        pct = val / total_sec * 100 if total_sec > 0 else 0.0
        ax.text(
            val + x_max * 0.015, i,
            f"{val:.2f}s  ({pct:.1f}%)",
            va="center", ha="left",
            fontsize=8, color=_TEXT_PRI,
        )

    # Y: nomes dos stages
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9, color=_TEXT_PRI)
    ax.invert_yaxis()

    # X: eixo em segundos
    ax.set_xlabel("Tempo (s)", fontsize=9, color=_TEXT_SEC)
    ax.tick_params(axis="x", colors=_TEXT_MUTED, labelsize=8)
    ax.tick_params(axis="y", colors=_TEXT_MUTED, length=0)

    # Gridlines: hairline, recessivas, só X
    ax.xaxis.grid(True, color=_GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)

    # Sem spines (visual limpo)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Margem extra à direita para os labels
    ax.set_xlim(right=x_max * 1.45)

    # Título
    ax.set_title(
        f"{title}\ntotal: {total_sec:.1f}s",
        fontsize=10, fontweight="semibold", color=_TEXT_PRI,
        loc="left", pad=10,
    )

    # Legenda (obrigatória para ≥ 2 séries presentes)
    present_cats = {c for c in cats}
    legend_handles = [
        mpatches.Patch(color=_COLORS[cat], label=cat)
        for cat in ("IO", "GPU", "CPU", "Cache")
        if cat in present_cats
    ]
    if len(present_cats) >= 2:
        ax.legend(
            handles=legend_handles,
            loc="lower right",
            fontsize=8,
            frameon=False,
            labelcolor=_TEXT_PRI,
        )

    plt.tight_layout(pad=1.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=_SURFACE)
    plt.close(fig)
