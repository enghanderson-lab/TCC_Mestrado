"""CLI: indexar um vídeo e buscar frames por descrição em linguagem natural.

A indexacao gera apenas embeddings SigLIP (rapido, todos os frames). A
legenda (Qwen2.5-VL) e a confianca textual sao calculadas sob demanda,
apenas para os top-K resultados de uma busca.

Os modelos sao carregados UMA UNICA VEZ no inicio do processo via
ModelManager e reutilizados durante toda a execucao. A logica de
indexacao/busca vive em `pipeline.py`.

Exemplos:
    python -m video_search.cli index video.mp4 --output index/video --interval 2
    python -m video_search.cli search "homem de camisa branca" --index index/video

Profiling:
    python -m video_search.cli index video.mp4 --profiling-dir profiling/
    python -m video_search.cli search "..." --profiling-dir profiling/
"""

import argparse
import json
import sys
from pathlib import Path

from .models.model_config import ModelConfig
from .models.model_manager import ModelManager
from .indexing.pipeline import SearchMode, run_index, run_search


def cmd_index(args: argparse.Namespace, mm: ModelManager) -> None:
    def on_progress(frame_count: int) -> None:
        print(f"Processados {frame_count} frames...")

    summary = run_index(
        args.video,
        args.output,
        interval=args.interval,
        batch_size=args.batch_size,
        limit=args.limit,
        on_progress=on_progress,
        model_manager=mm,
        profiling_dir=args.profiling_dir,
    )
    print(
        f"Indice salvo em '{summary.output_dir}' ({summary.frame_count} frames, "
        f"intervalo={args.interval}s)"
    )


def cmd_index_multi(args: argparse.Namespace, mm: ModelManager) -> None:
    import logging

    from .utils.config import MultiIndexConfig
    from .indexing.multi_index import CameraSource, run_multi_index

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(threadName)s %(name)s %(levelname)s %(message)s",
    )

    config = MultiIndexConfig.load(args.config)
    sources = [
        CameraSource(
            camera_id=Path(video).stem,
            video_path=video,
            output_dir=str(Path(args.output) / Path(video).stem),
        )
        for video in args.videos
    ]

    summaries = run_multi_index(
        sources,
        config=config,
        interval=args.interval,
        profiling_dir=args.profiling_dir,
    )
    for s in summaries:
        print(
            f"[{s.camera_id}] {s.frame_count}/{s.frames_read} frames aceitos -> '{s.output_dir}' "
            f"(motion={s.frames_discarded_motion}, similarity={s.frames_discarded_similarity})"
        )


def cmd_search(args: argparse.Namespace, mm: ModelManager) -> None:
    meta_path = Path(args.index) / "metadata.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            print(f"[Indice: modelo={json.load(f).get('model_name', 'siglip')}]")

    # --no-caption é compat legada; --mode tem precedência quando especificado
    mode = SearchMode(args.mode)
    if args.no_caption and args.mode == SearchMode.DETAILED.value:
        mode = SearchMode.FAST

    print(
        f"[Modo: {mode.value}  top_k={args.top_k}  retrieval_k={args.retrieval_k}  "
        f"reranker_lambda={args.reranker_lambda}]"
    )

    results = run_search(
        args.query,
        args.index,
        top_k=args.top_k,
        mode=mode,
        retrieval_k=args.retrieval_k,
        reranker_lambda=args.reranker_lambda,
        model_manager=mm,
        profiling_dir=args.profiling_dir,
    )

    if not results:
        print("Indice vazio.")
        return

    print(f"Resultados para: '{args.query}'")
    for r in results:
        minutes, seconds = divmod(r.timestamp_sec, 60)
        line = (
            f"  t={int(minutes):02d}:{seconds:05.2f}  frame={r.frame_index:6d}  "
            f"retrieval_score={r.retrieval_score:.4f}"
        )
        if r.confidence is not None:
            line += f"  confianca={r.confidence * 100:.1f}%  legenda='{r.caption}'"
        line += f"  video={r.video}"
        print(line)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Busca semantica em video via SigLIP + Qwen2-VL")
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
    p_search.add_argument("--index", default="index", help="Pasta do indice gerado por 'index'")
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
    p_search.add_argument(
        "--no-caption",
        action="store_true",
        help="[Legado] equivalente a --mode fast",
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

    config = ModelConfig(
        device=args.device,
        enable_warmup=not args.no_warmup,
    )
    with ModelManager(config) as mm:
        args.func(args, mm)


if __name__ == "__main__":
    main()
