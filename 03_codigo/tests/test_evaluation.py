"""Avaliação de ponta a ponta: vídeo sintético indexado de verdade (OpenCV +
pipeline + FAISS), busca DETAILED pelo SearchService e as 3 métricas; só os
modelos são fakes (conftest.py)."""

import csv
import json

import pytest

from conftest import FakeCaptioner, FakeImageEmbedder, FakeScorer, FakeTextEmbedder, write_video
from video_search.adapters.evaluation_report import write_report
from video_search.adapters.media.opencv_frames import OpenCVFrameSource
from video_search.adapters.storage.file_index import FileIndexRepository
from video_search.application.evaluation import (
    Annotation, EvaluationConfig, QueryOutcome, ResultRow, compute_metrics, evaluate, load_annotations,
)
from video_search.application.indexing_service import IndexingService
from video_search.application.search_service import SearchService
from video_search.domain.entities import VideoSource

BLACK, WHITE = (0, 0, 0), (255, 255, 255)


@pytest.fixture
def setup(tmp_path, gpu):
    """Câmera "cam": 5 frames pretos (t = 0,0–0,8 s) e 5 brancos (t = 1,0–1,8 s)."""
    indexes, frames = FileIndexRepository(), OpenCVFrameSource()
    video = write_video(tmp_path / "cam.mp4", [BLACK] * 5 + [WHITE] * 5)
    index_root = tmp_path / "index"
    IndexingService(FakeImageEmbedder(), frames, indexes, gpu).index_video(
        VideoSource("cam", video, str(index_root / "cam")), interval=0.2)
    search = SearchService(FakeTextEmbedder(), FakeCaptioner(), FakeScorer(), indexes, frames, gpu)
    return search, index_root


def _annotations(tmp_path, lines):
    path = tmp_path / "consultas.jsonl"
    path.write_text("\n".join(json.dumps(l, ensure_ascii=False) for l in lines) + "\n", encoding="utf-8")
    return load_annotations(path)


def test_perfect_confidence_end_to_end(setup, tmp_path):
    search, index_root = setup
    annotations = _annotations(tmp_path, [
        {"id": "p1", "query": "cena clara", "camera": "cam", "spans": [[1.0, 1.8]]},
        {"id": "p2", "query": "cena escura", "camera": "cam", "spans": [[0.0, 0.8]]},
        {"id": "n1", "query": "uma moto azul", "camera": "cam", "spans": []},
    ])
    config = EvaluationConfig(top_k=5, tolerance_sec=0.0, n_boot=200)

    report = evaluate(search, annotations, index_root, config)
    mt = report.metrics

    assert (mt.n_positive, mt.n_negative) == (2, 1)
    assert mt.detection.value == 1.0
    assert mt.false_alarm.value == 0.0
    assert mt.auroc.value == 1.0
    assert mt.recall_at_k == mt.retrieval_recall == 1.0
    assert mt.target_reached and mt.tau == pytest.approx(1.0)
    p1 = report.outcomes[0]
    assert [r.correct for r in p1.results] == [True] * 5
    assert all(r.caption == "cena clara" for r in p1.results)

    out = write_report(report, tmp_path / "relatorio")
    summary = json.loads((out / "metricas.json").read_text(encoding="utf-8"))
    assert summary["metricas"]["deteccao_confirmada"]["value"] == 1.0
    assert summary["ponto_de_operacao"]["alvo_atingivel"] is True
    with open(out / "consultas.csv", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f, delimiter=";"))
    assert len(rows) == 1 + 15  # cabeçalho + 3 consultas × 5 resultados
    assert (out / "roc.png").stat().st_size > 0


def test_wrong_annotation_shows_up_as_retrieval_miss(setup, tmp_path):
    search, index_root = setup
    # O alvo está anotado num trecho onde a busca não o encontra.
    annotations = _annotations(tmp_path, [
        {"id": "p1", "query": "cena clara", "camera": "cam", "spans": [[1.0, 1.8]]},
        {"id": "p2", "query": "cena clara", "camera": "cam", "spans": [[0.0, 0.2]]},
    ])

    report = evaluate(search, annotations, index_root, EvaluationConfig(tolerance_sec=0.0, n_boot=100))
    mt = report.metrics

    assert mt.recall_at_k == 0.5
    # o alvo de p2 estava entre os candidatos do SigLIP, mas saiu do top-5
    assert mt.retrieval_recall == 1.0
    assert not mt.target_reached          # 90% impossível: p2 não tem resultado correto
    assert mt.false_alarm.value is None   # sem consultas negativas
    # os mesmos frames brancos (confiança 1,0) são corretos em p1 e incorretos
    # em p2: a confiança não separa acerto de erro
    assert mt.auroc.value == 0.5
    assert report.outcomes[1].decision_score == float("-inf")


def test_missing_index_fails_fast(setup, tmp_path):
    search, index_root = setup
    annotations = _annotations(tmp_path, [{"id": "x", "query": "q", "camera": "nao_existe", "spans": []}])
    with pytest.raises(FileNotFoundError, match="nao_existe"):
        evaluate(search, annotations, index_root)


def test_load_annotations_validates(tmp_path):
    bad = tmp_path / "ruim.jsonl"
    bad.write_text('{"id": "a", "query": "x", "camera": "c", "spans": [[5, 1]]}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="linha 1|:1:"):
        load_annotations(bad)

    dup = tmp_path / "dup.jsonl"
    dup.write_text('{"id": "a", "query": "x", "camera": "c"}\n\n{"id": "a", "query": "y", "camera": "c"}\n',
                   encoding="utf-8")
    with pytest.raises(ValueError, match="repetido"):
        load_annotations(dup)


def test_fixed_tau_and_false_alarms():
    def outcome(qid, positive, rows):
        ann = Annotation(qid, qid, "cam", [(0.0, 1.0)] if positive else [])
        return QueryOutcome(ann, [ResultRow(i + 1, 0.0, 0.1, c, "x", ok) for i, (c, ok) in enumerate(rows)],
                            True if positive else None)

    outcomes = [
        outcome("p1", True, [(0.9, True), (0.3, False)]),
        outcome("p2", True, [(0.4, True)]),
        outcome("n1", False, [(0.8, False)]),   # falso alarme com τ = 0,5
        outcome("n2", False, [(0.2, False)]),
    ]

    mt = compute_metrics(outcomes, EvaluationConfig(tau=0.5, n_boot=100))

    assert mt.tau_source == "fixo"
    assert mt.detection.value == 0.5
    assert mt.false_alarm.value == 0.5
    # corretos {0.9, 0.4} × incorretos {0.3, 0.8, 0.2}: 0,9 vence os 3 e 0,4 vence 2 → 5 de 6 pares
    assert mt.auroc.value == pytest.approx(5 / 6)
