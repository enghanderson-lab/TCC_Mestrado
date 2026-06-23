"""Validacao do pipeline final de confianca (legenda condicionada a query +
sentence-transformer) em varias queries, no indice completo de teste01.mp4
(594 frames). Carrega os modelos uma unica vez para evitar overhead repetido."""

import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_search.siglip_embedder import SigLIPEmbedder  # noqa: E402
from video_search.embedding_store import EmbeddingStore  # noqa: E402
from video_search.frame_extractor import extract_frame_at  # noqa: E402
from video_search.vlm_describer import Qwen2VLDescriber  # noqa: E402

INDEX_DIR = str(Path(__file__).resolve().parents[2] / "04_dados" / "index" / "teste01")

QUERIES = [
    "ponte sobre o rio",
    "montanhas cobertas de arvores",
    "estrada com linha amarela no meio",
    "mulher de jaqueta preta andando na calcada",
]

TOP_K = 3


def main():
    embedder = SigLIPEmbedder()
    store = EmbeddingStore.load(INDEX_DIR)
    describer = Qwen2VLDescriber()
    text_model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        device=embedder.device,
    )

    for query in QUERIES:
        query_embedding = embedder.encode_text([query])[0]
        results = store.search(query_embedding, top_k=TOP_K)

        print(f"\n=== '{query}' ===")
        for score, rec in results:
            minutes, seconds = divmod(rec.timestamp_sec, 60)
            image = extract_frame_at(rec.video_path, rec.timestamp_sec)
            caption = describer.describe_for_query(image, query)
            emb = text_model.encode([query, caption], normalize_embeddings=True)
            confidence = float(np.dot(emb[0], emb[1]))
            print(
                f"  t={int(minutes):02d}:{seconds:05.2f}  frame={rec.index:6d}  "
                f"retrieval_score={score:.4f}  confianca_legenda={confidence * 100:.1f}%  "
                f"legenda='{caption}'"
            )


if __name__ == "__main__":
    main()
