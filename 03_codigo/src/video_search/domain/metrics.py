"""Métricas de avaliação da confiança (funções puras, só numpy).

A confiança é tratada como escore de decisão de um detector binário ("o alvo
está neste resultado?"), no arcabouço da análise ROC (Fawcett, 2006) e do
protocolo de detecção de erros/OOD por escore de confiança (Hendrycks &
Gimpel, ICLR 2017; Liang et al., ICLR 2018):

1. Detecção confirmada  — TPR no ponto de operação τ: fração das consultas
   positivas com um resultado correto no top-K e confiança ≥ τ.
2. Taxa de falso alarme — FPR no mesmo τ: fração das consultas negativas
   (objeto ausente, negativos adversariais à la POPE) com algum resultado de
   confiança ≥ τ. Com τ escolhido para TPR ≥ alvo, é o "FPR a TPR fixo".
3. AUROC da confiança   — probabilidade de um resultado correto ter
   confiança maior que um incorreto (Hanley & McNeil, 1982).

Para não estimar 1 e 2 com o mesmo dado usado para escolher τ (o que daria
TPR ≈ alvo por construção), `cross_validated_decisions` escolhe τ por
validação cruzada estratificada. Intervalos de confiança por bootstrap
reamostrando consultas (Efron & Tibshirani, 1993).
"""

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

NO_DETECTION = float("-inf")  # escore de uma consulta positiva sem resultado correto no top-K


def in_spans(timestamp_sec: float, spans: Sequence[Tuple[float, float]], tolerance_sec: float) -> bool:
    """True se `timestamp_sec` cai em algum trecho [ini, fim] ± tolerância."""
    return any(start - tolerance_sec <= timestamp_sec <= end + tolerance_sec for start, end in spans)


@dataclass
class OperatingPoint:
    tau: float
    reached: bool  # False se nem com τ mínimo a detecção alcança o alvo


def threshold_for_tpr(positive_scores: Sequence[float], target_tpr: float) -> OperatingPoint:
    """Maior τ com TPR ≥ `target_tpr` — o que minimiza falsos alarmes
    mantendo a detecção exigida (critério de Neyman-Pearson invertido).

    Consultas positivas sem resultado correto (NO_DETECTION) nunca são
    detectadas; se elas impedem atingir o alvo, devolve o τ de detecção
    máxima e `reached=False`."""
    scores = np.asarray(positive_scores, dtype=float)
    finite = np.sort(scores[np.isfinite(scores)])[::-1]
    needed = max(1, math.ceil(target_tpr * len(scores) - 1e-9))
    if len(finite) == 0:
        return OperatingPoint(tau=math.inf, reached=False)
    if len(finite) < needed:
        return OperatingPoint(tau=float(finite[-1]), reached=False)
    return OperatingPoint(tau=float(finite[needed - 1]), reached=True)


def auroc(labels: Sequence[bool], scores: Sequence[float]) -> Optional[float]:
    """Área sob a curva ROC pela estatística de Mann-Whitney (empates valem
    0,5). None se faltar uma das classes."""
    y = np.asarray(labels, dtype=bool)
    s = np.asarray(scores, dtype=float)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = _average_ranks(s)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_vals = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1  # posto médio (1-based) do grupo empatado
        i = j + 1
    return ranks


def cross_validated_decisions(
    positive_scores: Sequence[float],
    negative_scores: Sequence[float],
    target_tpr: float,
    folds: int = 5,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """Decide cada consulta com um τ calibrado nas demais (k-fold
    estratificado: positivos e negativos repartidos separadamente).

    Returns:
        (detectado por consulta positiva, falso alarme por consulta
         negativa, τ escolhido em cada fold)
    """
    pos = np.asarray(positive_scores, dtype=float)
    neg = np.asarray(negative_scores, dtype=float)
    rng = np.random.default_rng(seed)
    k = max(2, min(folds, len(pos)))
    pos_fold = rng.permutation(np.arange(len(pos)) % k)
    neg_fold = rng.permutation(np.arange(len(neg)) % k)

    detected = np.zeros(len(pos), dtype=bool)
    false_alarm = np.zeros(len(neg), dtype=bool)
    taus = []
    for f in range(k):
        train = pos[pos_fold != f]
        if len(train) == 0:
            continue
        tau = threshold_for_tpr(train, target_tpr).tau
        taus.append(tau)
        detected[pos_fold == f] = pos[pos_fold == f] >= tau
        false_alarm[neg_fold == f] = neg[neg_fold == f] >= tau
    return detected, false_alarm, taus


def bootstrap_ci(
    statistic: Callable[[np.ndarray, np.ndarray], Optional[float]],
    n_pos: int,
    n_neg: int,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Optional[Tuple[float, float]]:
    """IC percentil (1-α) reamostrando consultas com reposição, positivas e
    negativas separadamente. `statistic(idx_pos, idx_neg)` recalcula a
    métrica sobre os índices sorteados; reamostras em que ela é indefinida
    são descartadas."""
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        idx_pos = rng.integers(0, n_pos, n_pos) if n_pos else np.array([], dtype=int)
        idx_neg = rng.integers(0, n_neg, n_neg) if n_neg else np.array([], dtype=int)
        v = statistic(idx_pos, idx_neg)
        if v is not None and not math.isnan(v):
            values.append(v)
    if not values:
        return None
    lo, hi = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def roc_points(positive_scores: Sequence[float], negative_scores: Sequence[float]) -> List[Tuple[float, float, float]]:
    """Curva ROC por consulta: (τ, FPR, TPR) para cada escore distinto,
    do τ mais alto ao mais baixo, começando em (∞, 0, 0)."""
    pos = np.asarray(positive_scores, dtype=float)
    neg = np.asarray(negative_scores, dtype=float)
    taus = sorted({float(s) for s in np.concatenate([pos, neg]) if np.isfinite(s)}, reverse=True)
    points = [(math.inf, 0.0, 0.0)]
    for tau in taus:
        points.append((tau, float(np.mean(neg >= tau)) if len(neg) else 0.0,
                       float(np.mean(pos >= tau)) if len(pos) else 0.0))
    return points
