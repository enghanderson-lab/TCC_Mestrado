"""Verifica se describe_for_query() apenas 'confirma' a query (vies de
confirmacao, como aconteceu com match_confidence) ou se discrimina
corretamente quando a query NAO corresponde ao frame.

Usa um frame real de teste01.mp4 conhecido por mostrar um carro estacionado
na rua (t=4.0s) e compara a confianca_legenda para:
  - a query correta ("carro estacionado na rua")
  - duas queries claramente erradas (sem relacao com a cena)
"""

import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_search.media.frame_extractor import extract_frame_at  # noqa: E402
from video_search.models.vlm_describer import Qwen2VLDescriber  # noqa: E402

VIDEO_PATH = str(Path(__file__).resolve().parents[2] / "04_dados" / "raw" / "teste01.mp4")
TIMESTAMP_SEC = 4.0  # frame 2: clip_score alto para "carro estacionado na rua"

QUERIES = [
    ("carro estacionado na rua", "correta"),
    ("pessoa cozinhando na cozinha", "errada"),
    ("gato dormindo no sofa", "errada"),
]


def main():
    image = extract_frame_at(VIDEO_PATH, TIMESTAMP_SEC)
    describer = Qwen2VLDescriber()
    text_model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        device=describer.device,
    )

    for query, label in QUERIES:
        caption = describer.describe_for_query(image, query)
        emb = text_model.encode([query, caption], normalize_embeddings=True)
        confidence = float(np.dot(emb[0], emb[1]))
        print(f"[{label}] query='{query}'")
        print(f"  legenda='{caption}'")
        print(f"  confianca_legenda={confidence * 100:.1f}%\n")


if __name__ == "__main__":
    main()
