"""CLI: indexar um vídeo e buscar frames por descrição em linguagem natural.

A indexacao gera apenas embeddings CLIP (rapido, todos os frames). A legenda
(Qwen2-VL) e a confianca textual associada sao calculadas sob demanda, apenas
para os top-K resultados de uma busca - o VLM so e carregado/usado quando ha
uma consulta.

Exemplos:
    python -m video_search.cli index 04_dados/raw/camera1.mp4 --output index/camera1 --interval 2
    python -m video_search.cli search "homem de camisa branca e bone vermelho" --index index/camera1
"""

import argparse
from pathlib import Path

import numpy as np

from .embedder import ClipEmbedder
from .embedding_store import EmbeddingStore, FrameRecord
from .frame_extractor import extract_frame_at, extract_frames

CONFIDENCE_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def cmd_index(args: argparse.Namespace) -> None:
    embedder = ClipEmbedder()
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

    store.save(args.output)
    print(f"Indice salvo em '{args.output}' ({len(store)} frames, intervalo={args.interval}s)")


def cmd_search(args: argparse.Namespace) -> None:
    embedder = ClipEmbedder()
    store = EmbeddingStore.load(args.index)

    # Carrega o VLM antes do CLIP para poder traduzir a query para ingles.
    # O CLIP foi treinado majoritariamente em ingles; queries em portugues
    # geram embeddings de texto mais fracos, prejudicando o retrieval.
    describer = None
    text_model = None
    if not args.no_caption:
        from sentence_transformers import SentenceTransformer

        from .vlm_describer import Qwen2VLDescriber

        describer = Qwen2VLDescriber()
        text_model = SentenceTransformer(CONFIDENCE_MODEL, device=embedder.device)

    clip_query = args.query
    if describer is not None:
        clip_query = describer.translate_to_english(args.query)
        print(f"[CLIP query (EN): '{clip_query}']")

    query_embedding = embedder.encode_text([clip_query])[0]
    results = store.search(query_embedding, top_k=args.top_k)

    if not results:
        print("Indice vazio.")
        return

    print(f"Resultados para: '{args.query}'")
    for score, rec in results:
        minutes, seconds = divmod(rec.timestamp_sec, 60)
        line = (
            f"  t={int(minutes):02d}:{seconds:05.2f}  frame={rec.index:6d}  "
            f"clip_score={score:.4f}"
        )
        if describer is not None and rec.video_path:
            image = extract_frame_at(rec.video_path, rec.timestamp_sec)
            caption = describer.describe_for_query(image, args.query)
            # Confianca = similaridade semantica (sentence-transformer
            # multilingue) entre a query e a legenda focada na query, gerada
            # sob demanda pelo VLM para este resultado.
            text_emb = text_model.encode([args.query, caption], normalize_embeddings=True)
            confidence = float(np.dot(text_emb[0], text_emb[1]))
            line += f"  confianca_legenda={confidence * 100:.1f}%  legenda='{caption}'"
        line += f"  video={rec.video}"
        print(line)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Busca semantica em video via CLIP + Qwen2-VL")
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
    p_search.add_argument("--top-k", type=int, default=5, help="Numero de resultados")
    p_search.add_argument(
        "--no-caption",
        action="store_true",
        help="Pula a legenda/confianca via Qwen2-VL (busca apenas por score CLIP, mais rapido)",
    )
    p_search.set_defaults(func=cmd_search)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
