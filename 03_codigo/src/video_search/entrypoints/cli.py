"""CLI: indexar vídeos e buscar frames por descrição em linguagem natural.

A indexação gera apenas embeddings SigLIP (rápido, todos os frames). A
legenda (Qwen2.5-VL) e a confiança textual são calculadas sob demanda,
apenas para os top-K resultados de uma busca.

Adapter de entrada: traduz argumentos para os serviços montados em
`video_search.bootstrap.Container`. Os modelos são carregados uma única vez
por processo.

Exemplos:
    python -m video_search index video.mp4 --output index/video --interval 2
    python -m video_search search "homem de camisa branca" --index index/video
    python -m video_search search "van escolar" --index index/cam1 index/cam2

Profiling:
    python -m video_search index video.mp4 --profiling-dir profiling/
    python -m video_search search "..." --profiling-dir profiling/
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from ..adapters.models.model_config import ModelConfig
from ..application.search_service import SearchMode
from ..bootstrap import Container
from ..config import MultiIndexConfig
from ..domain.entities import Camera, VideoSource
from ..profiling import profiling_session


def cmd_index(args: argparse.Namespace, container: Container) -> None:
    def on_progress(_camera_id: str, frame_count: int) -> None:
        print(f"Processados {frame_count} frames...")

    source = VideoSource(camera_id=Path(args.video).stem, video_path=args.video, output_dir=args.output)
    with profiling_session("indexing", args.profiling_dir) as profiler:
        result = container.indexing.index_video(
            source,
            interval=args.interval,
            batch_size=args.batch_size,
            limit=args.limit,
            on_progress=on_progress,
            profiler=profiler,
        )
    print(
        f"Indice salvo em '{result.output_dir}' ({result.frames_accepted} frames, "
        f"intervalo={args.interval}s)"
    )


def cmd_index_multi(args: argparse.Namespace, container: Container) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(threadName)s %(name)s %(levelname)s %(message)s",
    )
    sources = [
        VideoSource(
            camera_id=Path(video).stem,
            video_path=video,
            output_dir=str(Path(args.output) / Path(video).stem),
        )
        for video in args.videos
    ]
    with profiling_session("indexing", args.profiling_dir) as profiler:
        results = container.indexing.index_cameras(
            sources,
            config=MultiIndexConfig.load(args.config),
            interval=args.interval,
            profiler=profiler,
        )
    for r in results:
        print(
            f"[{r.camera_id}] {r.frames_accepted}/{r.frames_read} frames aceitos -> '{r.output_dir}' "
            f"(motion={r.frames_dropped_motion}, similarity={r.frames_dropped_similarity})"
        )


def cmd_search(args: argparse.Namespace, container: Container) -> None:
    cameras = []
    for index_dir in args.index:
        meta_path = Path(index_dir) / "metadata.json"
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                print(f"[Indice {index_dir}: modelo={json.load(f).get('model_name', 'siglip')}]")
        cameras.append(Camera(camera_id=Path(index_dir).name, video_path="", output_dir=index_dir))

    mode = SearchMode(args.mode)
    print(
        f"[Modo: {mode.value}  top_k={args.top_k}  retrieval_k={args.retrieval_k}  "
        f"reranker_lambda={args.reranker_lambda}]"
    )

    with profiling_session("search", args.profiling_dir) as profiler:
        results = container.search.search(
            args.query,
            cameras,
            mode=mode,
            top_k=args.top_k,
            retrieval_k=args.retrieval_k,
            mmr_lambda=args.reranker_lambda,
            profiler=profiler,
        )
        if profiler:
            profiler.count("frames_indexed", len(results))

    if not results:
        print("Indice vazio.")
        return

    print(f"Resultados para: '{args.query}'")
    for r in results:
        minutes, seconds = divmod(r.timestamp_sec, 60)
        line = (
            f"  t={int(minutes):02d}:{seconds:05.2f}  frame={r.frame_index:6d}  "
            f"retrieval_score={r.score:.4f}"
        )
        if r.confidence is not None:
            line += f"  confianca={r.confidence * 100:.1f}%  legenda='{r.caption}'"
        line += f"  video={r.video}"
        if len(cameras) > 1:
            line += f"  camera={r.camera_id}"
        print(line)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Busca semantica em video via SigLIP + Qwen2.5-VL")
    parser.add_argument(
        "--device",
        default="auto",
        help="Device dos modelos: 'auto' (default), 'cuda:0', 'cpu'",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Pula o warmup dos modelos apos o carregamento",
    )
    parser.add_argument(
        "--profiling-dir",
        default=None,
        metavar="DIR",
        help=(
            "Pasta onde salvar o relatório de profiling (.txt) e o gráfico (.png) "
            "ao final da execução. Omitir desativa a geração de arquivos."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Extrai frames e gera embeddings")
    p_index.add_argument("video", help="Caminho do arquivo de video")
    p_index.add_argument("--output", default="index", help="Pasta onde salvar o indice")
    p_index.add_argument("--interval", type=float, default=2.0, help="Intervalo entre frames (s)")
    p_index.add_argument("--batch-size", type=int, default=16, help="Tamanho do lote para inferencia")
    p_index.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limita o numero de frames processados (0 = sem limite; util para testes)",
    )
    p_index.set_defaults(func=cmd_index)

    p_multi = sub.add_parser(
        "index-multi",
        help="Indexa multiplos videos/cameras concorrentemente (motion detection + batch cross-camera + filtro de similaridade)",
    )
    p_multi.add_argument("videos", nargs="+", help="Caminhos dos videos (um por camera)")
    p_multi.add_argument("--output", default="index", help="Pasta base; cada camera grava em <output>/<camera_id>")
    p_multi.add_argument(
        "--config",
        default=None,
        help="Caminho do YAML de configuracao (default: parametros padrao, ver configs/multi_index.yaml)",
    )
    p_multi.add_argument("--interval", type=float, default=2.0, help="Intervalo entre frames (s)")
    p_multi.set_defaults(func=cmd_index_multi)

    p_search = sub.add_parser("search", help="Busca frames por descricao em texto")
    p_search.add_argument("query", help="Descricao em linguagem natural")
    p_search.add_argument(
        "--index",
        nargs="+",
        default=["index"],
        help="Pasta(s) de indice geradas por 'index'/'index-multi'; varias = busca com ranking global",
    )
    p_search.add_argument("--top-k", type=int, default=5, help="Numero de resultados retornados")
    p_search.add_argument(
        "--mode",
        choices=[m.value for m in SearchMode],
        default=SearchMode.DETAILED.value,
        help=(
            "fast: SigLIP apenas (latência mínima). "
            "detailed: SigLIP → MMR → Qwen (máxima qualidade). "
            "Default: detailed"
        ),
    )
    p_search.add_argument(
        "--retrieval-k",
        type=int,
        default=100,
        help="Candidatos FAISS antes do reranker MMR (modo detailed). Default: 100",
    )
    p_search.add_argument(
        "--reranker-lambda",
        type=float,
        default=0.7,
        help=(
            "Peso MMR entre relevância e diversidade (modo detailed). "
            "1.0=relevância pura, 0.0=diversidade pura. Default: 0.7"
        ),
    )
    p_search.set_defaults(func=cmd_search)

    return parser


def main() -> None:
    # No Windows, stdout nao-interativo (pipe/redirecionamento) cai para a
    # codepage do sistema em vez de UTF-8, e acentos (ç, õ, ã) saem como "?".
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = build_parser()
    args = parser.parse_args()

    model_config = ModelConfig(device=args.device, enable_warmup=not args.no_warmup)
    with Container(model_config=model_config) as container:
        args.func(args, container)


if __name__ == "__main__":
    main()
