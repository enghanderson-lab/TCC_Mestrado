import threading
import time

import pytest

from conftest import FakeImageEmbedder, write_video
from video_search.adapters.gpu_executor import GpuExecutor
from video_search.adapters.media.opencv_frames import OpenCVFrameSource
from video_search.adapters.storage.file_index import FileIndexRepository
from video_search.application.indexing_service import IndexingService
from video_search.config import MultiIndexConfig
from video_search.domain.entities import VideoSource

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


@pytest.fixture
def service(gpu):
    return IndexingService(FakeImageEmbedder(), OpenCVFrameSource(), FileIndexRepository(), gpu)


def test_index_cameras_with_independent_filter_state(tmp_path, service):
    # accept(black,1o) reject(black) reject(black) accept(white) reject(white) accept(black)
    video_a = write_video(tmp_path / "cam_a.mp4", [BLACK, BLACK, BLACK, WHITE, WHITE, BLACK])
    # accept(white,1o) reject(white) accept(black) accept(white) accept(black) accept(white) accept(black)
    video_b = write_video(tmp_path / "cam_b.mp4", [WHITE, WHITE, BLACK, WHITE, BLACK, WHITE, BLACK])
    sources = [
        VideoSource("cam_a", video_a, str(tmp_path / "out_a")),
        VideoSource("cam_b", video_b, str(tmp_path / "out_b")),
    ]

    results = {r.camera_id: r for r in service.index_cameras(sources, MultiIndexConfig(), interval=0.2)}

    a, b = results["cam_a"], results["cam_b"]
    assert (a.frames_read, a.frames_accepted, a.frames_dropped_motion) == (6, 3, 3)
    assert (b.frames_read, b.frames_accepted, b.frames_dropped_motion) == (7, 6, 1)
    for r in results.values():
        assert r.frames_read == r.frames_accepted + r.frames_dropped_motion + r.frames_dropped_similarity

    stored = FileIndexRepository().load(str(tmp_path / "out_a"))
    assert stored.model_name == "siglip"
    assert len(stored) == 3


def test_index_video_keeps_every_frame_and_reports_progress(tmp_path, service):
    video = write_video(tmp_path / "v.mp4", [BLACK] * 10)
    progress = []

    result = service.index_video(
        VideoSource("v", video, str(tmp_path / "idx")), interval=0.2,
        on_progress=lambda cam, n: progress.append((cam, n)),
    )

    # sem filtros: frames identicos sao todos indexados
    assert result.frames_read == result.frames_accepted == 10
    assert progress[-1] == ("v", 10)


def test_index_video_limit_and_cancel(tmp_path, service):
    video = write_video(tmp_path / "v.mp4", [BLACK] * 20)

    limited = service.index_video(VideoSource("v", video, str(tmp_path / "lim")), interval=0.2, limit=4)
    assert limited.frames_accepted == 4

    cancelled = service.index_video(
        VideoSource("v", video, str(tmp_path / "can")), interval=0.2, should_cancel=lambda: True,
    )
    assert cancelled.frames_accepted == 0
    assert (tmp_path / "can" / "records.json").exists()  # resultado parcial salvo


def test_embedder_error_propagates(tmp_path, gpu):
    class Broken:
        def encode_images(self, images, profiler=None):
            raise RuntimeError("CUDA out of memory")

    service = IndexingService(Broken(), OpenCVFrameSource(), FileIndexRepository(), gpu)
    video = write_video(tmp_path / "v.mp4", [BLACK] * 3)

    with pytest.raises(RuntimeError, match="pipeline de indexação falhou"):
        service.index_video(VideoSource("v", video, str(tmp_path / "idx")), interval=0.2)


def test_gpu_executor_prioritizes_search_over_queued_indexing():
    gpu = GpuExecutor()
    started, release, order = threading.Event(), threading.Event(), []

    def blocker():
        started.set()
        release.wait()

    try:
        threads = [threading.Thread(target=gpu.run, args=(blocker,), kwargs={"priority": gpu.INDEXING})]
        threads[0].start()
        started.wait()
        # enquanto a GPU esta ocupada, enfileira 2 lotes de indexacao e depois 1 busca
        for name, prio in (("index1", gpu.INDEXING), ("index2", gpu.INDEXING), ("search", gpu.SEARCH)):
            t = threading.Thread(target=gpu.run, args=(order.append, name), kwargs={"priority": prio})
            t.start()
            threads.append(t)
            time.sleep(0.05)
        release.set()
        for t in threads:
            t.join()
    finally:
        gpu.shutdown()

    assert order == ["search", "index1", "index2"]


def test_gpu_executor_reentrant_and_propagates_errors(gpu):
    assert gpu.run(lambda: gpu.run(lambda: 42, priority=gpu.SEARCH), priority=gpu.SEARCH) == 42
    with pytest.raises(ValueError):
        gpu.run(int, "nao-numero", priority=gpu.SEARCH)
