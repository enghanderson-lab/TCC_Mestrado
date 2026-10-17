import pytest

from conftest import FakeCaptioner, FakeImageEmbedder, FakeScorer, FakeTextEmbedder, write_video
from video_search.adapters.media.opencv_frames import OpenCVFrameSource
from video_search.adapters.storage.file_index import FileIndexRepository
from video_search.application.indexing_service import IndexingService
from video_search.application.search_service import SearchMode, SearchService
from video_search.domain.entities import Camera, VideoSource

GRAY = [(v, v, v) for v in (0, 60, 120, 200, 255)]


@pytest.fixture
def setup(tmp_path, gpu):
    """Duas câmeras indexadas de verdade (pipeline + FAISS), com fakes de modelo."""
    indexes = FileIndexRepository()
    indexing = IndexingService(FakeImageEmbedder(), OpenCVFrameSource(), indexes, gpu)
    cameras = []
    for cam_id, colors in (("cam_a", GRAY), ("cam_b", list(reversed(GRAY)))):
        video = write_video(tmp_path / f"{cam_id}.mp4", colors)
        out = str(tmp_path / f"idx_{cam_id}")
        indexing.index_video(VideoSource(cam_id, video, out), interval=0.2)
        cameras.append(Camera(cam_id, video, out))

    text, captioner = FakeTextEmbedder(), FakeCaptioner()
    service = SearchService(text, captioner, FakeScorer(), indexes, OpenCVFrameSource(), gpu)
    return service, cameras, text, captioner


def test_fast_ranks_globally_across_cameras(setup):
    service, cameras, _, captioner = setup

    results = service.search("cena escura", cameras, mode=SearchMode.FAST, top_k=4)

    assert len(results) == 4
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)
    # o frame preto e o 1o de cam_a e o ultimo de cam_b
    assert {(r.camera_id, r.frame_index) for r in results[:2]} == {("cam_a", 0), ("cam_b", 4)}
    assert all(r.caption is None for r in results)
    assert captioner.calls == []  # FAST nunca chama o VLM


def test_detailed_captions_and_scores_top_k(setup):
    service, cameras, _, captioner = setup

    results = service.search("cena clara", cameras, mode="detailed", top_k=3)

    assert len(results) == 3
    assert captioner.calls == [3]  # um unico lote com os top_k
    assert results[0].caption == "cena clara"
    assert results[0].confidence == pytest.approx(1.0)
    assert all(r.caption is not None and r.confidence is not None for r in results)


def test_query_embedding_is_cached(setup):
    service, cameras, text, _ = setup

    service.search("cena escura", cameras, top_k=1)
    service.search("cena escura", cameras, top_k=2)

    assert text.calls == 1


def test_missing_index_and_missing_video_are_skipped(setup, tmp_path):
    service, cameras, _, _ = setup
    gone = Camera("gone", "x.mp4", str(tmp_path / "nao_existe"))

    fast = service.search("cena escura", [gone] + cameras, top_k=1)
    assert fast[0].camera_id in {"cam_a", "cam_b"}

    # video apagado depois de indexar: busca DETAILED devolve o resultado sem legenda
    for cam in cameras:
        (tmp_path / f"{cam.camera_id}.mp4").unlink()
    detailed = service.search("cena escura", cameras, mode="detailed", top_k=2)
    assert len(detailed) == 2
    assert all(r.caption is None for r in detailed)


def test_no_cameras_returns_empty(setup):
    service, _, _, _ = setup
    assert service.search("qualquer", [], mode="detailed") == []
