"""ConfidenceScorer via sentence-transformers: confiança = similaridade de
cosseno texto-texto entre a query e a legenda gerada pelo VLM."""

from typing import List, Sequence

import numpy as np

CONFIDENCE_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


class SentenceTransformerScorer:
    def __init__(self, model) -> None:  # model: sentence_transformers.SentenceTransformer
        self._model = model

    def score(self, query: str, captions: Sequence[str]) -> List[float]:
        if not captions:
            return []
        embeddings = self._model.encode([query, *captions], normalize_embeddings=True)
        query_emb, caption_embs = embeddings[0], embeddings[1:]
        return [float(np.dot(query_emb, emb)) for emb in caption_embs]
