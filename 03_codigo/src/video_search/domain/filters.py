"""Filtros por câmera aplicados durante a indexação.

MotionFilter: descarta frames quase idênticos ao último frame ACEITO daquela
câmera, antes de gerar embedding (economiza GPU). Compara contra o último
aceito (não o frame bruto anterior) para que uma deriva lenta de cena ainda
seja detectada eventualmente, em vez de nunca acumular diferença suficiente
entre dois frames consecutivos.

EmbeddingSimilarityFilter: descarta frames cujo embedding for quase idêntico
ao último embedding ACEITO daquela câmera — cenas redundantes que passaram
pelo filtro de movimento (ex.: ruído de captura numa cena estática).

Ambos guardam estado por `camera_id` e não precisam de lock: cada estágio da
pipeline roda numa única thread (ver `application/indexing_pipeline.py`).
"""

import logging
from typing import Dict

import numpy as np
from PIL.Image import Image

from ..config import EmbeddingFilterConfig, MotionDetectionConfig

logger = logging.getLogger(__name__)


class MotionFilter:
    def __init__(self, config: MotionDetectionConfig) -> None:
        self._config = config
        self._last_frame: Dict[str, np.ndarray] = {}

    def _preprocess(self, image: Image) -> np.ndarray:
        resized = image.resize((self._config.resize_width, self._config.resize_height))
        return np.asarray(resized.convert("L"), dtype=np.int16)

    def should_accept(self, camera_id: str, image: Image) -> bool:
        """Aceita o primeiro frame de cada câmera. Depois disso, aceita
        somente se a fração de pixels alterados em relação ao último frame
        aceito for >= `motion_ratio`; atualiza o estado apenas quando aceita."""
        if not self._config.enabled:
            return True

        current = self._preprocess(image)
        last = self._last_frame.get(camera_id)
        if last is None:
            self._last_frame[camera_id] = current
            return True

        diff = np.abs(current - last)
        ratio = float(np.count_nonzero(diff >= self._config.pixel_threshold)) / diff.size
        accept = ratio >= self._config.motion_ratio
        if accept:
            self._last_frame[camera_id] = current
        else:
            logger.debug("descartado (motion) camera=%s ratio=%.4f", camera_id, ratio)
        return accept


class EmbeddingSimilarityFilter:
    def __init__(self, config: EmbeddingFilterConfig) -> None:
        self._config = config
        self._last_embedding: Dict[str, np.ndarray] = {}

    def should_accept(self, camera_id: str, embedding: np.ndarray) -> bool:
        """Embeddings são L2-normalizados, então produto interno =
        similaridade de cosseno. Aceita o primeiro embedding de cada câmera;
        depois, aceita somente se a similaridade com o último aceito for
        < `similarity_threshold`."""
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
