"""Grava o resultado de uma avaliação em disco:

    <dir>/metricas.json   métricas, IC 95%, τ e configuração
    <dir>/consultas.csv   um resultado por linha (consulta, posição, confiança, acerto)
    <dir>/roc.png         curva ROC por consulta com o ponto de operação
"""

import csv
import json
import math
from dataclasses import asdict
from pathlib import Path

from ..application.evaluation import EvaluationReport


def _finite(x):
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def write_report(report: EvaluationReport, out_dir) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mt = report.metrics

    summary = {
        "metricas": {
            "deteccao_confirmada": asdict(mt.detection),
            "taxa_falso_alarme": asdict(mt.false_alarm),
            "auroc_confianca": asdict(mt.auroc),
        },
        "ponto_de_operacao": {
            "tau": _finite(mt.tau),
            "origem": mt.tau_source,
            "deteccao_alvo": report.config.target_tpr,
            "alvo_atingivel": mt.target_reached,
        },
        "diagnostico": {
            "recall_at_k": mt.recall_at_k,
            "recall_candidatos_siglip": mt.retrieval_recall,
            "consultas_positivas": mt.n_positive,
            "consultas_negativas": mt.n_negative,
            "resultados_com_confianca": mt.n_results_scored,
            "resultados_sem_confianca": mt.n_results_without_confidence,
        },
        "configuracao": asdict(report.config),
        "curva_roc": [{"tau": _finite(t), "fpr": f, "tpr": p} for t, f, p in mt.roc_points],
    }
    (out / "metricas.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with open(out / "consultas.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["id", "consulta", "camera", "tipo", "alvo_nos_candidatos", "posicao",
                    "t_seg", "score_siglip", "confianca", "correto", "legenda"])
        for o in report.outcomes:
            a = o.annotation
            tipo = "positiva" if a.positive else "negativa"
            hit = "" if o.retrieval_hit is None else ("sim" if o.retrieval_hit else "não")
            if not o.results:
                w.writerow([a.query_id, a.query, a.camera, tipo, hit, "", "", "", "", "", ""])
            for r in o.results:
                w.writerow([a.query_id, a.query, a.camera, tipo, hit, r.rank, f"{r.timestamp_sec:.2f}",
                            f"{r.score:.4f}", "" if r.confidence is None else f"{r.confidence:.4f}",
                            "sim" if r.correct else "não", r.caption or ""])

    _plot_roc(report, out / "roc.png")
    return out


def _plot_roc(report: EvaluationReport, path: Path) -> None:
    mt = report.metrics
    if not mt.roc_points:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fpr = [p[1] for p in mt.roc_points] + [1.0]
    tpr = [p[2] for p in mt.roc_points] + [mt.roc_points[-1][2]]
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.plot([0, 1], [0, 1], color="#b8bfc6", linewidth=1, linestyle="--", label="acaso")
    ax.step(fpr, tpr, where="post", color="#0d6b70", linewidth=2, label="busca (por consulta)")
    if mt.detection.value is not None and mt.false_alarm.value is not None:
        ax.scatter([mt.false_alarm.value], [mt.detection.value], color="#8a3b12", zorder=3,
                   label=f"ponto de operação (τ={mt.tau:.3f})" if math.isfinite(mt.tau) else "ponto de operação")
    ax.axhline(report.config.target_tpr, color="#8a3b12", linewidth=0.8, alpha=0.4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Taxa de falso alarme (consultas negativas)")
    ax.set_ylabel("Detecção confirmada (consultas positivas)")
    au = mt.auroc.value
    ax.set_title("Curva ROC da confiança" + (f" — AUROC por resultado {au:.3f}" if au is not None else ""),
                 fontsize=10, loc="left")
    ax.grid(True, color="#e1e5e8", linewidth=0.8)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
