"""CLI: indexar um vídeo e buscar frames por descrição em linguagem natural.

A indexacao gera apenas embeddings (rapido, todos os frames). A legenda
(Qwen2.5-VL) e a confianca textual sao calculadas sob demanda, apenas para os
top-K resultados de uma busca.

Modelo de embedding: SigLIP multilingual base-patch16-256 (Apache 2.0).
Suporte nativo a PT (sem necessidade de traducao da query).

A logica de indexacao/busca vive em `pipeline.py`, compartilhada com a API
HTTP (`api.py`).

Exemplos:
    python -m video_search.cli index video.mp4 --output index/video --interval 2
    python -m video_search.cli search "homem de camisa branca" --index index/video
"""

import argparse
import sys

from .pipeline import run_index, run_search


def cmd_index(args: argparse.Namespace) -> None:
    def on_progress(frame_count: int) -> None:
        print(f"Processados {frame_count} frames...")

    summary = run_index(
        args.video,
        args.output,
        interval=args.interval,
        batch_size=args.batch_size,
        limit=args.limit,
        on_progress=on_progress,
    )
    print(
        f"Indice salvo em '{summary.output_dir}' ({summary.frame_count} frames, "
        f"modelo={summary.model_name}, intervalo={args.interval}s)"
    )


def cmd_search(args: argparse.Namespace) -> None:
    results = run_search(
        args.query,
        args.index,
        top_k=args.top_k,
        with_caption=not args.no_caption,
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
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Extrai frames e gera embeddings")
    p_index.add_argument("video", help="Caminho do arquivo de video")
    p_index.add_argument("--output", default="index", help="Pasta onde salvar o indice")
    p_index.add_argument("--interval", type=float, default=1.0, help="Intervalo entre frames (s)")
    p_index.add_argument("--batch-size", type=int, default=16, help="Tamanho do lote para inferencia")
    p_index.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limita o numero de frames processados (0 = sem limite; util para testes)",
    )
    p_index.set_defaults(func=cmd_index)

    p_search = sub.add_parser("search", help="Busca frames por descricao em texto")
    p_search.add_argument("query", help="Descricao em linguagem natural")
    p_search.add_argument("--index", default="index", help="Pasta do indice gerado por 'index'")
    p_search.add_argument("--top-k", type=int, default=12, help="Numero de resultados")
    p_search.add_argument(
        "--no-caption",
        action="store_true",
        help="Pula a legenda/confianca via Qwen2-VL (busca apenas por score, mais rapido)",
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
    args.func(args)


if __name__ == "__main__":
    main()
