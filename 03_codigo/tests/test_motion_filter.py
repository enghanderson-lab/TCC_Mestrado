from PIL import Image

from video_search.utils.config import MotionDetectionConfig
from video_search.media.motion_filter import MotionFilter


def _solid_image(color, size=(64, 64)):
    return Image.new("RGB", size, color=color)


def test_first_frame_always_accepted():
    motion_filter = MotionFilter(MotionDetectionConfig())
    assert motion_filter.should_accept("cam1", _solid_image((0, 0, 0))) is True


def test_identical_frame_rejected():
    motion_filter = MotionFilter(MotionDetectionConfig())
    frame = _solid_image((10, 10, 10))
    assert motion_filter.should_accept("cam1", frame) is True
    assert motion_filter.should_accept("cam1", frame) is False


def test_very_different_frame_accepted():
    motion_filter = MotionFilter(MotionDetectionConfig())
    assert motion_filter.should_accept("cam1", _solid_image((0, 0, 0))) is True
    assert motion_filter.should_accept("cam1", _solid_image((255, 255, 255))) is True


def test_per_camera_state_is_independent():
    motion_filter = MotionFilter(MotionDetectionConfig())
    black = _solid_image((0, 0, 0))

    assert motion_filter.should_accept("cam1", black) is True
    # cam2 nunca viu um frame antes -- deve aceitar o primeiro, mesmo que
    # seja identico ao ultimo frame aceito de cam1.
    assert motion_filter.should_accept("cam2", black) is True


def test_disabled_always_accepts():
    motion_filter = MotionFilter(MotionDetectionConfig(enabled=False))
    frame = _solid_image((5, 5, 5))
    assert motion_filter.should_accept("cam1", frame) is True
    assert motion_filter.should_accept("cam1", frame) is True
