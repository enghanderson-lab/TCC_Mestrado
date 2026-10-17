"""Benchmark antes/depois das otimizacoes de indexacao (motion detection +
batch cross-camera + filtro de similaridade, ver
`application/indexing_service.py`).

"Antes": `IndexingService.index_video` (sem filtros) + legenda via
Qwen2.5-VL em TODOS os frames extraidos -- equivalente ao pipeline sem
nenhuma das 3 otimizacoes (todo frame embedado E legendado).

"Depois": `IndexingService.index_cameras` sobre o MESMO video -- reporta
quantos frames sobrevivem ao filtro de movimento e de similaridade (os
unicos que precisariam de legenda).

Uso:
    python scripts/benchmark_optimizations.py caminho/do/video.mp4 [--interval 2.0]
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_search.bootstrap import Container  # noqa: E402
from video_search.config import MultiIndexConfig  # noqa: E402
from video_search.domain.entities import VideoSource  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="Caminho do video de teste")
    parser.add_argument("--interval", type=float, default=2.0, help="Intervalo entre frames (s)")
    parser.add_argument("--output", default=None, help="Pasta base para os indices gerados (default: scripts/_benchmark_out)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(threadName)s %(name)s %(levelname)s %(message)s")

    output_base = Path(args.output) if args.output else Path(__file__).resolve().parent / "_benchmark_out"
    output_base.mkdir(parents=True, exist_ok=True)

    with Container() as container:
        print("=" * 60)
        print("ANTES (sem otimizacoes) -- index_video + legenda em TODOS os frames")
        print("=" * 60)

        source = VideoSource(camera_id="antes", video_path=args.video, output_dir=str(output_base / "antes"))
        t0 = time.perf_counter()
        old = container.indexing.index_video(source, interval=args.interval)
        t_index_old = time.perf_counter() - t0
        print(f"Indexacao (SigLIP, {old.frames_accepted} frames): {t_index_old:.2f}s")

        describer = container.models.get_qwen()
        t0 = time.perf_counter()
        n_captions = 0
        for frame in container.frames.iter_frames(args.video, interval_sec=args.interval):
            container.gpu.run(describer.describe, frame.image, priority=container.gpu.INDEXING)
            n_captions += 1
        t_vlm_old = time.perf_counter() - t0
        print(f"VLM (legenda em todos os {n_captions} frames): {t_vlm_old:.2f}s ({t_vlm_old / max(n_captions, 1):.2f}s/frame)")
        print(f"TOTAL antes: {t_index_old + t_vlm_old:.2f}s")

        print()
        print("=" * 60)
        print("DEPOIS (com otimizacoes) -- index_cameras (motion + similarity)")
        print("=" * 60)

        source = VideoSource(camera_id="depois", video_path=args.video, output_dir=str(output_base / "depois"))
        t0 = time.perf_counter()
        new = container.indexing.index_cameras([source], config=MultiIndexConfig(), interval=args.interval)[0]
        t_total_new = time.perf_counter() - t0
        print(
            f"frames lidos={new.frames_read} descartados_motion={new.frames_dropped_motion} "
            f"descartados_similarity={new.frames_dropped_similarity} aceitos={new.frames_accepted}"
        )
        print(f"TOTAL depois (indexacao com filtros): {t_total_new:.2f}s")


if __name__ == "__main__":
    main()
