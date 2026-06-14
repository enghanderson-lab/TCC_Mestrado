import cv2
import numpy as np

from video_search.frame_extractor import extract_frames


def _write_synthetic_video(path, fps=10, num_frames=30, size=(32, 32)):
    width, height = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for i in range(num_frames):
        frame = np.full((height, width, 3), fill_value=i % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_extract_frames_at_interval(tmp_path):
    video_path = tmp_path / "synthetic.mp4"
    _write_synthetic_video(video_path, fps=10, num_frames=30)

    frames = list(extract_frames(video_path, interval_sec=1.0))

    assert len(frames) == 3
    assert [f.index for f in frames] == [0, 1, 2]
    assert frames[1].timestamp_sec == 1.0
    assert frames[0].image.size == (32, 32)


def test_extract_frames_missing_file(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        list(extract_frames(tmp_path / "nao_existe.mp4", interval_sec=1.0))
