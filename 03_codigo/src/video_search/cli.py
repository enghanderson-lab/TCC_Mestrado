"""CLI: indexar um vídeo e buscar frames por descrição em linguagem natural.

A indexacao gera apenas embeddings (rapido, todos os frames). A legenda
(Qwen2.5-VL) e a confianca textual sao calculadas sob demanda, apenas para os
top-K resultados de uma busca.

Modelo de embedding: SigLIP multilingual base-patch16-256 (Apache 2.0).
Suporte nativo a PT (sem necessidade de traducao da query).

Exemplos:
    python -m video_search.cli index video.mp4 --output index/video --interval 2
    python -m video_search.cli search "homem de camisa branca" --index index/video
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from .embedding_store import EmbeddingStore, FrameRecord
from .frame_extractor import extract_frame_at, extract_frames
from .siglip_embedder import SigLIPEmbedder

CONFIDENCE_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def cmd_index(args: argparse.Namespace) -> None:
    embedder = SigLIPEmbedder()
    store = EmbeddingStore()
    video_name = Path(args.video).name
    video_path = str(Path(args.video).resolve())

    batch_images = []
    batch_records = []

    def flush() -> None:
        if not batch_images:
            return
        embeddings = embedder.encode_images(batch_images)
        for emb, rec in zip(embeddings, batch_records):
            store.add(emb, rec)
        batch_images.clear()
        batch_records.clear()

    for frame in extract_frames(args.video, interval_sec=args.interval):
        if args.limit and frame.index >= args.limit:
            break
        batch_images.append(frame.image)
        batch_records.append(
            FrameRecord(
                index=frame.index,
                timestamp_sec=frame.timestamp_sec,
                video=video_name,
                video_path=video_path,
            )
        )
        if len(batch_images) >= args.batch_size:
            flush()
            print(f"Processados {len(store)} frames...")
    flush()

    store.save(args.output, model_name="siglip")
    print(f"Indice salvo em '{args.output}' ({len(store)} frames, modelo=siglip, intervalo={args.interval}s)")


def cmd_search(args: argparse.Namespace) -> None:
    store = EmbeddingStore.load(args.index)
    print(f"[Indice: modelo={store.model_name}]")

    embedder = SigLIPEmbedder()
    need_caption = not args.no_caption

    query_embedding = embedder.encode_text([args.query])[0]
    results = store.search(query_embedding, top_k=args.top_k)

    # O embedder de retrieval ja fez seu trabalho; liberar sua VRAM agora da
    # mais espaço para o Qwen2.5-VL durante o generate() de cada legenda.
    del embedder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    describer = None
    text_model = None
    if need_caption:
        from sentence_transformers import SentenceTransformer
        from .vlm_describer import Qwen2VLDescriber

        describer = Qwen2VLDescriber()
        # CPU de proposito: o calculo de confianca so processa 2 frases curtas
        # por resultado, custo irrelevante. Mantendo isso fora da GPU libera
        # ~1GB de VRAM para o Qwen2.5-VL.
        text_model = SentenceTransformer(CONFIDENCE_MODEL, device="cpu")

    if not results:
        print("Indice vazio.")
        return

    print(f"Resultados para: '{args.query}'")
    for score, rec in results:
        minutes, seconds = divmod(rec.timestamp_sec, 60)
        line = (
            f"  t={int(minutes):02d}:{seconds:05.2f}  frame={rec.index:6d}  "
            f"retrieval_score={score:.4f}"
        )
        if describer is not None and rec.video_path:
            image = extract_frame_at(rec.video_path, rec.timestamp_sec)
            caption = describer.describe_for_query(image, args.query)
            text_emb = text_model.encode([args.query, caption], normalize_embeddings=True)
            confidence = float(np.dot(text_emb[0], text_emb[1]))
            line += f"  confianca={confidence * 100:.1f}%  legenda='{caption}'"
        line += f"  video={rec.video}"
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
