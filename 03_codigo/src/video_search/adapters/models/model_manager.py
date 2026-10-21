"""Gerenciador centralizado de modelos de IA.

Carrega SigLIP, Qwen2.5-VL e o modelo de confianca (NLI) uma unica vez e os
mantem residentes na VRAM/RAM durante todo o ciclo de vida da aplicacao,
eliminando o custo de carregamento a cada busca / indexacao.

Cada modelo e carregado na 1a chamada ao seu get_*(): a CLI de indexacao e a
busca FAST so precisam do SigLIP e nao pagam a carga do Qwen e do NLI (com os
pesos num HDD, a carga dos 3 passa de 3 min). A API chama load_models() no
startup para que a 1a busca ja encontre tudo pronto.

Uma instancia por processo, criada pelo composition root
(`video_search.bootstrap.Container`). Imports de siglip, qwen e do NLI sao
lazy para que importar este modulo nao carregue transformers.
"""

import logging
import threading
from typing import TYPE_CHECKING, Optional

import torch

from .model_config import ModelConfig

if TYPE_CHECKING:
    from .nli_scorer import NliConfidenceScorer
    from .qwen import Qwen2VLDescriber
    from .siglip import SigLIPEmbedder

logger = logging.getLogger(__name__)


class ModelManager:
    """Mantém instâncias únicas de SigLIP, Qwen2.5-VL e do scorer NLI.

    Thread-safe: `_init_lock` serializa a inicializacao; modelos em si sao
    read-only durante inferencia (`@torch.no_grad`); o acesso a GPU e
    serializado pelo `GpuExecutor` (ver adapters/gpu_executor.py).
    """

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        self._config = config or ModelConfig()
        self._init_lock = threading.RLock()
        self._siglip: Optional["SigLIPEmbedder"] = None
        self._qwen: Optional["Qwen2VLDescriber"] = None
        self._scorer: Optional["NliConfidenceScorer"] = None

    # ------------------------------------------------------------------
    # Ciclo de vida
    # ------------------------------------------------------------------

    def load_models(self) -> None:
        """Carrega os 3 modelos. Idempotente."""
        self.get_siglip()
        self.get_qwen()
        self.get_scorer()
        logger.info("[ModelManager] todos os modelos prontos e residentes.")

    def unload_models(self) -> None:
        """Libera pesos de todos os modelos e limpa cache CUDA."""
        with self._init_lock:
            if self._siglip:
                self._siglip.unload()
            if self._qwen:
                self._qwen.unload()
            self._siglip = self._qwen = self._scorer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @property
    def is_loaded(self) -> bool:
        """True quando os 3 modelos estão residentes."""
        return None not in (self._siglip, self._qwen, self._scorer)

    # ------------------------------------------------------------------
    # Acesso aos modelos (carga sob demanda)
    # ------------------------------------------------------------------

    def get_siglip(self) -> "SigLIPEmbedder":
        with self._init_lock:
            if self._siglip is None:
                from .siglip import SigLIPEmbedder

                self._siglip = SigLIPEmbedder(device=self._device(), precision=self._config.siglip_precision)
                if self._config.enable_warmup:
                    self._siglip.warmup()
                logger.info("[ModelManager] SigLIP pronto.")
            return self._siglip

    def get_qwen(self) -> "Qwen2VLDescriber":
        with self._init_lock:
            if self._qwen is None:
                from .qwen import Qwen2VLDescriber

                self._qwen = Qwen2VLDescriber(
                    device=self._device(), quantize=self._config.qwen_precision == "int4"
                )
                if self._config.enable_warmup:
                    self._qwen.warmup()
                logger.info("[ModelManager] Qwen2.5-VL pronto.")
            return self._qwen

    def get_scorer(self) -> "NliConfidenceScorer":
        with self._init_lock:
            if self._scorer is None:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                from .hf_utils import load_offline_first
                from .nli_scorer import NLI_MODEL, NliConfidenceScorer

                self._scorer = NliConfidenceScorer(
                    load_offline_first(AutoModelForSequenceClassification.from_pretrained, NLI_MODEL),
                    load_offline_first(AutoTokenizer.from_pretrained, NLI_MODEL),
                    device=self._config.scorer_device,
                )
                if self._config.enable_warmup:
                    self._scorer.score("warmup", ["warmup"])
                logger.info("[ModelManager] scorer NLI pronto.")
            return self._scorer

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "ModelManager":
        self.load_models()
        return self

    def __exit__(self, *_) -> None:
        self.unload_models()

    def _device(self) -> str:
        if self._config.device == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        return self._config.device
