import cv2
import numpy as np

from video_search import multi_index
from video_search.config import MultiIndexConfig
from video_search.embedding_store import EmbeddingStore
from video_search.multi_index import CameraSource, run_multi_index


def _write_video(path, colors, fps=5, size=(64, 64)):
    width, height = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for color in colors:
        frame = np.full((height, width, 3), color, dtype=np.uint8)
        writer.write(frame)
    writer.release()


class FakeEmbedder:
    """Embedding determinístico de 2 dimensões a partir do brilho médio do
    frame, normalizado -- suficiente para o filtro de similaridade distinguir
    frames claramente diferentes (preto vs branco) sem precisar de GPU/SigLIP."""

    def encode_images(self, images):
        vectors = []
        for image in images:
            mean = float(np.asarray(image.convert("L"), dtype=np.float32).mean()) / 255.0
            vec = np.array([mean, 1.0 - mean], dtype=np.float32)
            norm = np.linalg.norm(vec)
            vectors.append(vec / norm if norm > 0 else vec)
        return np.stack(vectors)


class FakeDescriber:
    def describe(self, image, prompt=None, max_new_tokens=64):
        return "fake-caption"


def test_run_multi_index_two_cameras_with_independent_state(tmp_path, monkeypatch):
    monkeypatch.setattr(multi_index, "_make_embedder", lambda: FakeEmbedder())
    monkeypatch.setattr(multi_index, "Qwen2VLDescriber", FakeDescriber)

    black = (0, 0, 0)
    white = (255, 255, 255)

    video_a = tmp_path / "cam_a.mp4"
    # accept(black,1o) reject(black) reject(black) accept(white) reject(white) accept(black)
    _write_video(video_a, [black, black, black, white, white, black])

    video_b = tmp_path / "cam_b.mp4"
    # accept(white,1o) reject(white) accept(black) accept(white) accept(black) accept(white) accept(black)
    _write_video(video_b, [white, white, black, white, black, white, black])

    sources = [
        CameraSource(camera_id="cam_a", video_path=str(video_a), output_dir=str(tmp_path / "out_a")),
        CameraSource(camera_id="cam_b", video_path=str(video_b), output_dir=str(tmp_path / "out_b")),
    ]

    summaries = run_multi_index(sources, config=MultiIndexConfig(), interval=0.2)
    by_id = {s.camera_id: s for s in summaries}

    summary_a = by_id["cam_a"]
    assert summary_a.frames_read == 6
    assert summary_a.frame_count == 3
    assert summary_a.frames_discarded_motion == 3

    summary_b = by_id["cam_b"]
    assert summary_b.frames_read == 7
    assert summary_b.frame_count == 6
    assert summary_b.frames_discarded_motion == 1

    # Contabilidade interna consistente em ambas as cameras.
    for summary in summaries:
        assert summary.frames_read == (
            summary.frame_count + summary.frames_discarded_motion + summary.frames_discarded_similarity
        )

    # Saida no MESMO formato de hoje (run_index) -- carregavel por EmbeddingStore.load,
    # com o campo `caption` (ja existente, sem uso ate aqui) populado.
    store_a = EmbeddingStore.load(tmp_path / "out_a")
    assert store_a.model_name == "siglip"
    assert len(store_a) == 3
    assert all(record.caption == "fake-caption" for record in store_a._records)
