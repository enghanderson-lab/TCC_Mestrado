"""Caso de uso: avaliar a confiança da busca contra um ground truth anotado.

Formato da anotação (JSONL, uma consulta por linha):

    {"id": "p01", "query": "uma van branca", "camera": "Teste13", "spans": [[12.0, 18.5]]}
    {"id": "n01", "query": "uma escada no teto do carro", "camera": "Teste13", "spans": []}

- `camera`: pasta do índice dentro de `index_root` (ex.: index/Teste13).
- `spans`: trechos [início, fim] em segundos onde o alvo aparece. Lista
  vazia = consulta negativa (o objeto NÃO está no vídeo).

Cada consulta roda no modo DETAILED do SearchService (o mesmo da CLI e da
API). As métricas estão definidas em `domain/metrics.py`.
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from ..domain import metrics as m
from ..domain.entities import Camera
from .search_service import SearchMode, SearchService, default_retrieval_k


@dataclass
class Annotation:
    query_id: str
    query: str
    camera: str
    spans: List[Tuple[float, float]]

    @property
    def positive(self) -> bool:
        return bool(self.spans)


def load_annotations(path) -> List[Annotation]:
    """Lê e valida o JSONL; erros citam a linha."""
    annotations, seen = [], set()
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                spans = [(float(a), float(b)) for a, b in row.get("spans", [])]
                ann = Annotation(str(row["id"]), str(row["query"]).strip(), str(row["camera"]), spans)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{lineno}: linha inválida ({exc})") from exc
            if not ann.query:
                raise ValueError(f"{path}:{lineno}: 'query' vazia")
            if any(a > b for a, b in spans):
                raise ValueError(f"{path}:{lineno}: trecho com início maior que o fim")
            if ann.query_id in seen:
                raise ValueError(f"{path}:{lineno}: id '{ann.query_id}' repetido")
            seen.add(ann.query_id)
            annotations.append(ann)
    if not annotations:
        raise ValueError(f"{path}: nenhuma consulta anotada")
    return annotations


@dataclass
class EvaluationConfig:
    top_k: int = 5
    tolerance_sec: float = 2.0      # = intervalo de amostragem padrão da indexação
    target_tpr: float = 0.90
    tau: Optional[float] = None     # None = escolhe τ por validação cruzada
    folds: int = 5
    n_boot: int = 2000
    seed: int = 0
    retrieval_k: Optional[int] = None
    mmr_lambda: float = 0.7


@dataclass
class ResultRow:
    rank: int
    timestamp_sec: float
    score: float
    confidence: Optional[float]
    caption: Optional[str]
    correct: bool


@dataclass
class QueryOutcome:
    annotation: Annotation
    results: List[ResultRow]
    retrieval_hit: Optional[bool]   # alvo entre os candidatos do SigLIP (só positivas)

    @property
    def decision_score(self) -> float:
        """Escore da consulta para as métricas 1 e 2: maior confiança entre
        os resultados corretos (positiva) ou entre todos (negativa)."""
        pool = [r.confidence for r in self.results
                if r.confidence is not None and (r.correct or not self.annotation.positive)]
        return max(pool) if pool else m.NO_DETECTION


@dataclass
class Estimate:
    value: Optional[float]
    ci95: Optional[Tuple[float, float]] = None


@dataclass
class EvaluationMetrics:
    detection: Estimate             # métrica 1: TPR no ponto de operação
    false_alarm: Estimate           # métrica 2: FPR no mesmo ponto
    auroc: Estimate                 # métrica 3
    tau: float                      # τ recomendado (calibrado em todos os dados)
    tau_source: str                 # "fixo" | "validação cruzada (k=..)"
    target_reached: bool
    recall_at_k: Optional[float]    # positivas com resultado correto no top-K, com qualquer confiança
    retrieval_recall: Optional[float]  # positivas com o alvo entre os candidatos do SigLIP
    n_positive: int
    n_negative: int
    n_results_scored: int
    n_results_without_confidence: int
    roc_points: List[Tuple[float, float, float]] = field(default_factory=list)  # (τ, FPR, TPR) por consulta


@dataclass
class EvaluationReport:
    config: EvaluationConfig
    outcomes: List[QueryOutcome]
    metrics: EvaluationMetrics


def evaluate(
    search: SearchService,
    annotations: Sequence[Annotation],
    index_root: Path,
    config: Optional[EvaluationConfig] = None,
    on_query: Optional[Callable[[int, int, Annotation], None]] = None,
) -> EvaluationReport:
    config = config or EvaluationConfig()
    index_root = Path(index_root)
    missing = sorted({a.camera for a in annotations if not (index_root / a.camera / "records.json").exists()})
    if missing:
        raise FileNotFoundError(f"índice não encontrado em {index_root} para: {', '.join(missing)}")

    candidates_k = config.retrieval_k or default_retrieval_k(config.top_k)
    outcomes = []
    for i, ann in enumerate(annotations, 1):
        if on_query:
            on_query(i, len(annotations), ann)
        camera = Camera(ann.camera, "", str(index_root / ann.camera))
        results = search.search(
            ann.query, [camera], mode=SearchMode.DETAILED, top_k=config.top_k,
            retrieval_k=config.retrieval_k, mmr_lambda=config.mmr_lambda,
        )
        rows = [
            ResultRow(rank, r.timestamp_sec, r.score, r.confidence, r.caption,
                      ann.positive and m.in_spans(r.timestamp_sec, ann.spans, config.tolerance_sec))
            for rank, r in enumerate(results, 1)
        ]
        retrieval_hit = None
        if ann.positive:
            # Diagnóstico: o alvo chegou a estar entre os candidatos que o
            # SigLIP passa ao MMR? (FAST não chama o VLM; a query já está em cache.)
            pool = search.search(ann.query, [camera], mode=SearchMode.FAST, top_k=candidates_k)
            retrieval_hit = any(m.in_spans(r.timestamp_sec, ann.spans, config.tolerance_sec) for r in pool)
        outcomes.append(QueryOutcome(ann, rows, retrieval_hit))

    return EvaluationReport(config, outcomes, compute_metrics(outcomes, config))


def compute_metrics(outcomes: Sequence[QueryOutcome], config: EvaluationConfig) -> EvaluationMetrics:
    pos = [o for o in outcomes if o.annotation.positive]
    neg = [o for o in outcomes if not o.annotation.positive]
    pos_scores = np.array([o.decision_score for o in pos], dtype=float)
    neg_scores = np.array([o.decision_score for o in neg], dtype=float)

    # --- Métricas 1 e 2: decisão por consulta no ponto de operação -------
    full_point = m.threshold_for_tpr(pos_scores, config.target_tpr) if len(pos) else m.OperatingPoint(math.inf, False)
    if config.tau is not None:
        tau, tau_source = config.tau, "fixo"
        detected, false_alarm = pos_scores >= tau, neg_scores >= tau
        target_reached = bool(len(pos)) and float(detected.mean()) >= config.target_tpr
    else:
        detected, false_alarm, _ = m.cross_validated_decisions(
            pos_scores, neg_scores, config.target_tpr, config.folds, config.seed)
        tau, tau_source = full_point.tau, f"validação cruzada (k={max(2, min(config.folds, len(pos)))})"
        target_reached = full_point.reached

    def mean_or_none(x: np.ndarray) -> Optional[float]:
        return float(x.mean()) if len(x) else None

    detection = Estimate(mean_or_none(detected), m.bootstrap_ci(
        lambda ip, _in: float(detected[ip].mean()) if len(ip) else None,
        len(pos), len(neg), config.n_boot, seed=config.seed) if len(pos) else None)
    fa = Estimate(mean_or_none(false_alarm), m.bootstrap_ci(
        lambda _ip, ineg: float(false_alarm[ineg].mean()) if len(ineg) else None,
        len(pos), len(neg), config.n_boot, seed=config.seed + 1) if len(neg) else None)

    # --- Métrica 3: AUROC por resultado, reamostrando consultas -------------
    per_query = [
        ([r.correct for r in o.results if r.confidence is not None],
         [r.confidence for r in o.results if r.confidence is not None])
        for o in outcomes
    ]
    pos_idx = [i for i, o in enumerate(outcomes) if o.annotation.positive]
    neg_idx = [i for i, o in enumerate(outcomes) if not o.annotation.positive]

    def auroc_of(queries: Sequence[int]) -> Optional[float]:
        labels = [y for q in queries for y in per_query[q][0]]
        scores = [s for q in queries for s in per_query[q][1]]
        return m.auroc(labels, scores)

    au = auroc_of(range(len(outcomes)))
    au_ci = None
    if au is not None:
        au_ci = m.bootstrap_ci(
            lambda ip, ineg: auroc_of([pos_idx[i] for i in ip] + [neg_idx[i] for i in ineg]),
            len(pos_idx), len(neg_idx), config.n_boot, seed=config.seed + 2)

    n_scored = sum(len(s) for _, s in per_query)
    n_total = sum(len(o.results) for o in outcomes)
    return EvaluationMetrics(
        detection=detection,
        false_alarm=fa,
        auroc=Estimate(au, au_ci),
        tau=tau,
        tau_source=tau_source,
        target_reached=target_reached,
        recall_at_k=mean_or_none(np.array([any(r.correct for r in o.results) for o in pos])),
        retrieval_recall=mean_or_none(np.array([bool(o.retrieval_hit) for o in pos])),
        n_positive=len(pos),
        n_negative=len(neg),
        n_results_scored=n_scored,
        n_results_without_confidence=n_total - n_scored,
        roc_points=m.roc_points(pos_scores, neg_scores) if len(pos) and len(neg) else [],
    )
