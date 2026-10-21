"""Testes de integração com os modelos reais (SigLIP, Qwen2.5-VL 4-bit,
sentence-transformer) montados pelo Container de produção.

Fora da suíte padrão (ver pytest.ini). Rodar com:
    .venv\\Scripts\\python.exe -m pytest -m models -s

Os modelos são carregados uma única vez para o módulo inteiro.
"""

import time
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from conftest import write_video
from video_search.adapters.models.model_config import ModelConfig
from video_search.application.search_service import SearchMode
from video_search.bootstrap import Container
from video_search.config import Settings
from video_search.domain.entities import Camera, VideoSource

pytestmark = [
    pytest.mark.models,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requer GPU CUDA"),
]

RED, BLUE = (0, 0, 255), (255, 0, 0)  # BGR (OpenCV)
REAL_VIDEO = Path(__file__).resolve().parents[1] / "uploads" / "Teste14_teste14.mp4"


@pytest.fixture(scope="module")
def container(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("models")
    c = Container(Settings(db_path=tmp / "db.sqlite", index_root=tmp / "index"),
                  model_config=ModelConfig(enable_warmup=False))
    t0 = time.perf_counter()
    c.models.load_models()
    print(f"\n[carga dos modelos: {time.perf_counter() - t0:.1f}s, "
          f"VRAM alocada={torch.cuda.memory_allocated() / 2**30:.2f} GB]")
    yield c
    c.close()


@pytest.fixture(scope="module")
def red_blue_camera(container, tmp_path_factory):
    """Vídeo sintético: 5 frames vermelhos seguidos de 5 azuis, indexado com
    o SigLIP real."""
    tmp = tmp_path_factory.mktemp("video")
    video = write_video(tmp / "red_blue.mp4", [RED] * 5 + [BLUE] * 5, size=(256, 256))
    out = str(tmp / "idx")
    result = container.indexing.index_video(VideoSource("red_blue", video, out), interval=0.2)
    assert result.frames_accepted == 10
    return Camera("red_blue", video, out)


def _solid(rgb):
    return Image.new("RGB", (256, 256), rgb)


def test_siglip_embeddings_are_normalized_and_multilingual(container):
    siglip = container.models.get_siglip()
    imgs = siglip.encode_images([_solid((220, 20, 20)), _solid((20, 20, 220))])
    texts = siglip.encode_text(["uma imagem vermelha", "uma imagem azul"])

    assert imgs.shape == texts.shape == (2, 768)
    np.testing.assert_allclose(np.linalg.norm(imgs, axis=1), 1.0, atol=1e-3)
    np.testing.assert_allclose(np.linalg.norm(texts, axis=1), 1.0, atol=1e-3)

    sims = texts @ imgs.T  # [texto, imagem]
    print(f"\n[SigLIP sim texto×imagem]\n{np.round(sims, 4)}")
    assert sims[0, 0] > sims[0, 1]  # "vermelha" casa mais com a imagem vermelha
    assert sims[1, 1] > sims[1, 0]  # "azul" casa mais com a imagem azul


def test_qwen_batch_captions_follow_the_image(container):
    qwen = container.models.get_qwen()
    t0 = time.perf_counter()
    captions = container.gpu.run(
        qwen.describe_for_query_batch,
        [_solid((220, 20, 20)), _solid((20, 20, 220))], "uma imagem vermelha",
        priority=container.gpu.SEARCH,
    )
    print(f"\n[Qwen {time.perf_counter() - t0:.1f}s] {captions}")

    assert len(captions) == 2 and all(c.strip() for c in captions)
    assert "vermelh" in captions[0].lower()
    # A imagem azul não pode ser confirmada como vermelha. O prompt pede que a
    # ausência seja dita ("Não há nada vermelho na imagem."), então o critério
    # é a confiança NLI, não uma palavra específica na legenda.
    conf = container.models.get_scorer().score("uma imagem vermelha", captions)
    print(f"  confiança NLI: vermelha={conf[0]:.3f} azul={conf[1]:.3f}")
    assert conf[0] > 0.5 > conf[1]


def test_search_fast_on_real_index(container, red_blue_camera):
    results = container.search.search("uma imagem vermelha", [red_blue_camera], mode=SearchMode.FAST, top_k=3)

    assert len(results) == 3
    assert all(r.frame_index < 5 for r in results)  # os 3 primeiros são frames vermelhos


def test_search_detailed_captions_and_confidence(container, red_blue_camera):
    t0 = time.perf_counter()
    results = container.search.search(
        "uma imagem vermelha", [red_blue_camera], mode=SearchMode.DETAILED, top_k=3, mmr_lambda=0.7,
    )
    elapsed = time.perf_counter() - t0
    for r in results:
        print(f"\n  frame={r.frame_index} score={r.score:.4f} conf={r.confidence:.3f} legenda={r.caption!r}")
    print(f"[busca detailed: {elapsed:.1f}s]")

    assert len(results) == 3
    assert results[0].frame_index < 5
    assert all(r.caption and -1.0 <= r.confidence <= 1.0 for r in results)
    red = [r for r in results if r.frame_index < 5]
    blue = [r for r in results if r.frame_index >= 5]
    # legenda de frame vermelho tem confiança maior que a de frame azul (MMR traz diversidade)
    if red and blue:
        assert max(r.confidence for r in red) > max(r.confidence for r in blue)


@pytest.mark.skipif(not REAL_VIDEO.exists(), reason="video real de teste ausente")
def test_real_video_index_and_detailed_search(container, tmp_path):
    out = str(tmp_path / "real")
    t0 = time.perf_counter()
    indexed = container.indexing.index_video(
        VideoSource("real", str(REAL_VIDEO), out), interval=2.0, limit=12,
    )
    t_index = time.perf_counter() - t0

    t0 = time.perf_counter()
    results = container.search.search(
        "uma pessoa", [Camera("real", str(REAL_VIDEO), out)], mode=SearchMode.DETAILED, top_k=3,
    )
    t_search = time.perf_counter() - t0
    print(f"\n[video real: indexou {indexed.frames_accepted} frames em {t_index:.1f}s; busca detailed {t_search:.1f}s]")
    for r in results:
        print(f"  t={r.timestamp_sec:6.1f}s score={r.score:.4f} conf={r.confidence:.3f} legenda={r.caption!r}")

    assert indexed.frames_accepted == 12
    assert len(results) == 3
    assert all(r.caption for r in results)
    print(f"[pico de VRAM: {torch.cuda.max_memory_allocated() / 2**30:.2f} GB]")


def test_evaluate_confidence_metrics_with_real_models(container, red_blue_camera, tmp_path):
    from video_search.adapters.evaluation_report import write_report
    from video_search.application.evaluation import Annotation, EvaluationConfig, evaluate

    index_root = Path(red_blue_camera.output_dir).parent
    camera = Path(red_blue_camera.output_dir).name
    red, blue = [(0.0, 0.8)], [(1.0, 1.8)]
    annotations = [
        Annotation("p1", "uma imagem vermelha", camera, red),
        Annotation("p2", "uma imagem azul", camera, blue),
        Annotation("p3", "uma tela toda vermelha", camera, red),
        Annotation("n1", "um carro amarelo", camera, []),
        Annotation("n2", "uma pessoa andando de bicicleta", camera, []),
    ]
    t0 = time.perf_counter()
    report = evaluate(container.search, annotations, index_root,
                      EvaluationConfig(top_k=3, tolerance_sec=0.0, n_boot=500))
    out = write_report(report, tmp_path / "avaliacao")
    mt = report.metrics
    print(f"\n[evaluate: {len(annotations)} consultas em {time.perf_counter() - t0:.1f}s]")
    for o in report.outcomes:
        best = o.decision_score
        print(f"  {o.annotation.query_id} {o.annotation.query!r}: escore={best:.3f} "
              f"legendas={[r.caption[:50] for r in o.results[:1]]}")
    print(f"  detecção={mt.detection.value} falso_alarme={mt.false_alarm.value} "
          f"AUROC={mt.auroc.value} τ={mt.tau:.3f} recall@3={mt.recall_at_k}")

    # O teste verifica que a avaliação roda de ponta a ponta com os modelos
    # reais, não a qualidade do sistema. Em 2026-10-17, nestes frames de cor
    # sólida, o Qwen confirmou as consultas negativas ("A imagem mostra uma
    # pessoa andando de bicicleta.") e a AUROC deu 0,17 — o viés de confirmação
    # que a métrica existe para medir.
    assert mt.recall_at_k == 1.0          # SigLIP traz a cor certa no top-3
    for est in (mt.detection, mt.false_alarm, mt.auroc):
        assert est.value is not None and 0.0 <= est.value <= 1.0
        assert est.ci95 is not None
    assert all(r.caption for o in report.outcomes for r in o.results)
    assert (out / "metricas.json").exists() and (out / "roc.png").exists()


def test_nli_scorer_handles_negation(container):
    scorer = container.models.get_scorer()
    query = "uma mulher de rosa sobre uma moto e sem uso do capacete"
    conf = scorer.score(query, [
        "Uma mulher de vestido rosa pilota uma moto vermelha sem capacete.",
        "Não há uma mulher de rosa sobre uma moto e sem uso do capacete na imagem.",
        "Um homem de camisa branca está parado na calçada.",
    ])
    print(f"\n[NLI] afirma={conf[0]:.3f} nega={conf[1]:.3f} fala_de_outra_coisa={conf[2]:.3f}")
    assert conf[0] > 0.9
    assert conf[1] < 0.1   # antes (cosseno): 92,8%
    assert conf[2] < 0.1


TESTE_A1_INDEX = Path(__file__).resolve().parents[1] / "index" / "testeA1"


@pytest.mark.skipif(not (TESTE_A1_INDEX / "records.json").exists(), reason="índice testeA1 ausente")
def test_real_case_woman_in_pink_on_motorcycle(container):
    """Regressão do caso real de 2026-10-18: alvo em t=40 s, pequeno no
    frame 1080p. Antes: t=40 s com 21,3% e "Não há uma mulher..." com 92,8%."""
    query = "uma mulher de rosa sobre uma moto e sem uso do capacete"
    t0 = time.perf_counter()
    results = container.search.search(
        query, [Camera("testeA1", "", str(TESTE_A1_INDEX))], mode=SearchMode.DETAILED, top_k=5)
    print(f"\n[testeA1 detailed: {time.perf_counter() - t0:.1f}s, pico VRAM "
          f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GB]")
    for r in sorted(results, key=lambda r: -r.confidence):
        print(f"  t={r.timestamp_sec:5.1f}s conf={r.confidence:.3f} legenda={r.caption!r}")

    target = [r for r in results if 38.0 <= r.timestamp_sec <= 42.0]
    assert target, "o SigLIP deixou de trazer o frame do alvo"
    best = max(results, key=lambda r: r.confidence)
    assert best is target[0], "a maior confiança não está no frame do alvo"
    for r in results:
        if r.caption.lower().startswith(("não há", "não aparece", "nao ha")):
            assert r.confidence < 0.5
