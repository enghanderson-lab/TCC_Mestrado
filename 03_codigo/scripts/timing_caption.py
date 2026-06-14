"""Mede o tempo de geracao de legenda (Qwen2-VL) por frame em um video real,
para estimar o custo de indexar com --caption."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_search.frame_extractor import extract_frames  # noqa: E402
from video_search.vlm_describer import Qwen2VLDescriber  # noqa: E402

VIDEO = Path(__file__).resolve().parents[2] / "04_dados" / "raw" / "teste01.mp4"
N_FRAMES = 5
INTERVAL = 2.0


def main():
    describer = Qwen2VLDescriber()
    print(f"Dispositivo: {describer.device}")

    times = []
    for i, frame in enumerate(extract_frames(str(VIDEO), interval_sec=INTERVAL)):
        if i >= N_FRAMES:
            break
        start = time.perf_counter()
        caption = describer.describe(frame.image)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"frame {frame.index} (t={frame.timestamp_sec:.1f}s): {elapsed:.2f}s -> {caption}")

    avg = sum(times) / len(times)
    print(f"\nMedia: {avg:.2f}s/frame")
    for n_frames, interval in [(594, 2.0), (3600, 1.0)]:
        total_min = avg * n_frames / 60
        print(f"  Estimativa p/ {n_frames} frames (interval={interval}s): {total_min:.1f} min")


if __name__ == "__main__":
    main()
