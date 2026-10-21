"""As 3 correções da confiança: recorte ampliado (tiling + escolha pelo
SigLIP), hipótese do NLI e o contrato do captioner com recortes."""

import numpy as np
import pytest
from PIL import Image

from conftest import FakeCaptioner, FakeScorer, write_video
from video_search.adapters.media.opencv_frames import OpenCVFrameSource
from video_search.adapters.models.nli_scorer import hypothesis_for
from video_search.adapters.storage.file_index import FileIndexRepository
from video_search.application.indexing_service import IndexingService
from video_search.application.search_service import SearchService
from video_search.domain.entities import Camera, VideoSource
from video_search.domain.tiling import ZoomConfig, tile_boxes


def test_tile_boxes_cover_frame_with_overlap():
    boxes = tile_boxes(1920, 1080, ZoomConfig(grid=3, tile_fraction=0.5))

    assert len(boxes) == 9
    assert all((r - l, b - t) == (960, 540) for l, t, r, b in boxes)
    assert min(b[0] for b in boxes) == 0 and max(b[2] for b in boxes) == 1920
    assert min(b[1] for b in boxes) == 0 and max(b[3] for b in boxes) == 1080
    # alvo do testeA1 (~x 450–560, y 420–630) cabe inteiro em algum recorte
    assert any(l <= 450 and r >= 560 and t <= 420 and b >= 630 for l, t, r, b in boxes)


def test_tile_boxes_degenerate_grid():
    assert tile_boxes(100, 50, ZoomConfig(grid=1, tile_fraction=1.0)) == [(0, 0, 100, 50)]


@pytest.mark.parametrize("query, expected", [
    ("uma mulher de rosa sobre uma moto", "A imagem mostra uma mulher de rosa sobre uma moto."),
    ("procure por uma van branca", "A imagem mostra uma van branca."),
    ("Busque um carro com escada no teto.", "A imagem mostra um carro com escada no teto."),
    ("Homem de camisa branca", "A imagem mostra homem de camisa branca."),
])
def test_nli_hypothesis(query, expected):
    assert hypothesis_for(query) == expected


class _CornerEmbedder:
    """Embedding 2-D pela fração de pixels brancos: o recorte com mais
    branco é o mais "parecido" com a query (que aponta para [1, 0])."""

    def encode_images(self, images, profiler=None):
        out = []
        for im in images:
            white = float((np.asarray(im.convert("L")) > 200).mean())
            v = np.array([white, 1.0 - white], dtype=np.float32)
            out.append(v / np.linalg.norm(v))
        return np.stack(out)


class _WhiteQuery:
    def encode_text(self, texts):
        return np.array([[1.0, 0.0]] * len(texts), dtype=np.float32)


def test_detailed_sends_zoom_on_the_target(tmp_path, gpu):
    # frame preto com um quadrado branco pequeno no canto inferior direito
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    frame[140:170, 270:310] = 255
    path = tmp_path / "v.mp4"
    import cv2
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (320, 180))
    for _ in range(3):
        w.write(frame)
    w.release()

    indexes, frames = FileIndexRepository(), OpenCVFrameSource()
    IndexingService(_CornerEmbedder(), frames, indexes, gpu).index_video(
        VideoSource("v", str(path), str(tmp_path / "idx")), interval=0.2)
    captioner = FakeCaptioner()
    service = SearchService(_WhiteQuery(), captioner, FakeScorer(), indexes, frames, gpu,
                            image_embedder=_CornerEmbedder(), zoom=ZoomConfig())

    service.search("cena clara", [Camera("v", str(path), str(tmp_path / "idx"))], mode="detailed", top_k=1)

    zoom = captioner.zooms[0][0]
    assert zoom.size == (160, 90)                      # recorte de 50% do frame
    assert np.asarray(zoom.convert("L")).max() > 200   # o recorte escolhido contém o alvo branco


def test_zoom_requires_image_embedder(gpu):
    with pytest.raises(ValueError, match="image_embedder"):
        SearchService(_WhiteQuery(), FakeCaptioner(), FakeScorer(), FileIndexRepository(),
                      OpenCVFrameSource(), gpu, zoom=ZoomConfig())


def test_without_zoom_captioner_gets_no_crops(tmp_path, gpu):
    video = write_video(tmp_path / "v.mp4", [(255, 255, 255)] * 3)
    indexes, frames = FileIndexRepository(), OpenCVFrameSource()
    IndexingService(_CornerEmbedder(), frames, indexes, gpu).index_video(
        VideoSource("v", video, str(tmp_path / "idx")), interval=0.2)
    captioner = FakeCaptioner()
    service = SearchService(_WhiteQuery(), captioner, FakeScorer(), indexes, frames, gpu)

    service.search("cena clara", [Camera("v", video, str(tmp_path / "idx"))], mode="detailed", top_k=2)

    assert captioner.zooms == [None]
