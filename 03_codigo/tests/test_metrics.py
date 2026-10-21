import math

import numpy as np
import pytest

from video_search.domain import metrics as m


def test_in_spans_with_tolerance():
    spans = [(10.0, 20.0), (50.0, 55.0)]
    assert m.in_spans(15.0, spans, 0.0)
    assert m.in_spans(8.5, spans, 2.0)
    assert not m.in_spans(7.9, spans, 2.0)
    assert m.in_spans(57.0, spans, 2.0)
    assert not m.in_spans(30.0, spans, 2.0)
    assert not m.in_spans(15.0, [], 2.0)


def test_threshold_for_tpr_picks_highest_tau_reaching_target():
    pos = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]

    point = m.threshold_for_tpr(pos, 0.9)

    assert point.reached
    assert point.tau == pytest.approx(0.1)  # 9 de 10 positivas com confiança >= 0,1
    assert np.mean(np.asarray(pos) >= point.tau) == pytest.approx(0.9)


def test_threshold_unreachable_when_positives_were_never_retrieved():
    pos = [0.9, 0.8, m.NO_DETECTION, m.NO_DETECTION]  # metade nem apareceu no top-K

    point = m.threshold_for_tpr(pos, 0.9)

    assert not point.reached
    assert point.tau == pytest.approx(0.8)  # detecção máxima possível = 50%
    assert np.mean(np.asarray(pos) >= point.tau) == pytest.approx(0.5)


def test_auroc_matches_sklearn_including_ties():
    sklearn = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(3)
    labels = rng.random(200) < 0.4
    scores = np.round(rng.random(200) + labels * 0.3, 1)  # arredondado força empates

    assert m.auroc(labels, scores) == pytest.approx(sklearn.roc_auc_score(labels, scores))


def test_auroc_reference_values():
    assert m.auroc([True, True, False, False], [0.9, 0.8, 0.2, 0.1]) == 1.0
    assert m.auroc([True, True, False, False], [0.1, 0.2, 0.8, 0.9]) == 0.0
    assert m.auroc([True, False], [0.5, 0.5]) == 0.5
    assert m.auroc([True, True], [0.1, 0.2]) is None


def test_cross_validation_does_not_leak_the_threshold():
    # Positivas e negativas separáveis: CV deve detectar ~todas sem falso alarme.
    rng = np.random.default_rng(0)
    pos = rng.uniform(0.7, 1.0, 40)
    neg = rng.uniform(0.0, 0.3, 15)

    detected, false_alarm, taus = m.cross_validated_decisions(pos, neg, 0.9, folds=5, seed=0)

    assert len(taus) == 5
    assert detected.mean() >= 0.8
    assert false_alarm.mean() == 0.0
    assert all(0.7 <= t <= 1.0 for t in taus)


def test_uninformative_confidence_yields_high_false_alarm():
    # Confiança sem relação com a classe: para manter ~90% de detecção o τ
    # fica baixo, e a maioria das consultas negativas vira falso alarme.
    rng = np.random.default_rng(1)
    pos = rng.random(50)
    neg = rng.random(50)

    detected, false_alarm, _ = m.cross_validated_decisions(pos, neg, 0.9, folds=5, seed=1)

    assert detected.mean() >= 0.75
    assert false_alarm.mean() > 0.6


def test_bootstrap_ci_contains_point_estimate():
    detected = np.array([True] * 18 + [False] * 2)

    lo, hi = m.bootstrap_ci(lambda ip, _in: float(detected[ip].mean()), 20, 0, n_boot=1000, seed=0)

    assert lo <= 0.9 <= hi
    assert 0.7 <= lo < hi <= 1.0


def test_roc_points_are_monotonic():
    points = m.roc_points([0.9, 0.7, 0.4], [0.8, 0.2])

    assert points[0] == (math.inf, 0.0, 0.0)
    assert points[-1][1:] == (1.0, 1.0)
    fprs = [p[1] for p in points]
    tprs = [p[2] for p in points]
    assert fprs == sorted(fprs) and tprs == sorted(tprs)
