"""Filtro de similaridade de embeddings por camera: descarta (nao legenda)
frames cujo embedding for quase identico ao ultimo embedding ACEITO daquela
camera — evita gastar o VLM em cenas redundantes que passaram pelo filtro de
movimento (ex.: ruido de captura numa cena estatica)."""

import logging
from typing import Dict

import numpy as np

from .config import EmbeddingFilterConfig

logger = logging.getLogger(__name__)


class EmbeddingSimilarityFilter:
    """Mesma forma do `MotionFilter`: estado por `camera_id`, sem lock (cada
    camera e tocada por exatamente uma thread)."""

    def __init__(self, config: EmbeddingFilterConfig) -> None:
        self._config = config
        self._last_embedding: Dict[str, np.ndarray] = {}

    def should_accept(self, camera_id: str, embedding: np.ndarray) -> bool:
        """Embeddings sao L2-normalizados (ver siglip_embedder.py), entao
        produto interno = similaridade de cosseno. Aceita o primeiro
        embedding de cada camera; depois, aceita somente se a similaridade
        com o ultimo aceito for < `similarity_threshold`."""
        if not self._config.enabled:
            return True

        last = self._last_embedding.get(camera_id)
        if last is None:
            self._last_embedding[camera_id] = embedding
            return True

        similarity = float(np.dot(embedding, last))
        accept = similarity < self._config.similarity_threshold
        if accept:
            self._last_embedding[camera_id] = embedding
        else:
            logger.debug("descartado (similarity) camera=%s similarity=%.4f", camera_id, similarity)
        return accept
