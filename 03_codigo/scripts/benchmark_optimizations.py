"""Benchmark antes/depois das otimizacoes de indexacao (motion detection +
batch cross-camera + filtro de similaridade, ver `multi_index.py`).

"Antes": `run_index` (pipeline.py, inalterado) + legenda via Qwen2.5-VL em
TODOS os frames extraidos -- equivalente ao pipeline sem nenhuma das 3
otimizacoes (todo frame embedado E legendado).

"Depois": `run_multi_index` (multi_index.py) sobre o MESMO video -- so
legenda os frames que sobrevivem ao filtro de movimento e de similaridade.

Uso:
    python scripts/benchmark_optimizations.py caminho/do/video.mp4 [--interval 2.0]
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from video_search.utils.config import MultiIndexConfig  # noqa: E402
from video_search.media.frame_extractor import extract_frames  # noqa: E402
from video_search.indexing.multi_index import CameraSource, run_multi_index  # noqa: E402
from video_search.indexing.pipeline import run_index  # noqa: E402
from video_search.models.vlm_describer import Qwen2VLDescriber  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="Caminho do video de teste")
    parser.add_argument("--interval", type=float, default=2.0, help="Intervalo entre frames (s)")
    parser.add_argument("--output", default=None, help="Pasta base para os indices gerados (default: pasta temporaria)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(threadName)s %(name)s %(levelname)s %(message)s")

    output_base = Path(args.output) if args.output else Path(__file__).resolve().parent / "_benchmark_out"
    output_base.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ANTES (sem otimizacoes) -- run_index + legenda em TODOS os frames")
    print("=" * 60)

    t0 = time.perf_counter()
    old_summary = run_index(args.video, str(output_base / "antes"), interval=args.interval)
    t_index_old = time.perf_counter() - t0
    print(f"Indexacao (SigLIP, {old_summary.frame_count} frames): {t_index_old:.2f}s")

    describer = Qwen2VLDescriber()
    t0 = time.perf_counter()
    n_captions = 0
    for frame in extract_frames(args.video, interval_sec=args.interval):
        describer.describe(frame.image)
        n_captions += 1
    t_vlm_old = time.perf_counter() - t0
    print(f"VLM (legenda em todos os {n_captions} frames): {t_vlm_old:.2f}s ({t_vlm_old / n_captions:.2f}s/frame)")
    print(f"TOTAL antes: {t_index_old + t_vlm_old:.2f}s")

    del describer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print()
    print("=" * 60)
    print("DEPOIS (com otimizacoes) -- run_multi_index (motion + similarity)")
    print("=" * 60)

    source = CameraSource(camera_id="benchmark", video_path=args.video, output_dir=str(output_base / "depois"))
    t0 = time.perf_counter()
    summaries = run_multi_index([source], config=MultiIndexConfig(), interval=args.interval)
    t_total_new = time.perf_counter() - t0
    new_summary = summaries[0]
    print(
        f"frames lidos={new_summary.frames_read} descartados_motion={new_summary.frames_discarded_motion} "
        f"descartados_similarity={new_summary.frames_discarded_similarity} legendados={new_summary.frame_count}"
    )
    print(f"TOTAL depois (indexacao + VLM combinados): {t_total_new:.2f}s")


if __name__ == "__main__":
    main()
