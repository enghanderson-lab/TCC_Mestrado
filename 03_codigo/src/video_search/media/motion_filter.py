"""Filtro de movimento por camera: descarta frames quase identicos ao ultimo
frame ACEITO daquela mesma camera, antes de gerar embedding (economiza GPU).

Compara contra o ultimo frame aceito (nao o frame bruto anterior) para que
uma deriva lenta de cena ainda seja detectada eventualmente, em vez de nunca
acumular diferenca suficiente entre dois frames consecutivos."""

import logging
from typing import Dict

import cv2
import numpy as np
from PIL import Image

from ..utils.config import MotionDetectionConfig

logger = logging.getLogger(__name__)


class MotionFilter:
    """Estado independente por `camera_id`. Nao precisa de lock: cada
    camera e lida por exatamente uma thread (ver `multi_index.py`), entao
    nunca ha acesso concorrente ao estado da MESMA camera."""

    def __init__(self, config: MotionDetectionConfig) -> None:
        self._config = config
        self._last_frame: Dict[str, np.ndarray] = {}

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        resized = image.resize((self._config.resize_width, self._config.resize_height))
        return cv2.cvtColor(np.asarray(resized), cv2.COLOR_RGB2GRAY)

    def should_accept(self, camera_id: str, image: Image.Image) -> bool:
        """Aceita o primeiro frame de cada camera. Depois disso, aceita
        somente se a fracao de pixels alterados em relacao ao ultimo frame
        aceito for >= `motion_ratio`; atualiza o estado apenas quando aceita."""
        if not self._config.enabled:
            return True

        current = self._preprocess(image)
        last = self._last_frame.get(camera_id)
        if last is None:
            self._last_frame[camera_id] = current
            return True

        diff = cv2.absdiff(current, last)
        ratio = float(np.count_nonzero(diff >= self._config.pixel_threshold)) / diff.size
        accept = ratio >= self._config.motion_ratio
        if accept:
            self._last_frame[camera_id] = current
        else:
            logger.debug("descartado (motion) camera=%s ratio=%.4f", camera_id, ratio)
        return accept
