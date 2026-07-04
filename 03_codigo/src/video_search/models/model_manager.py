"""Gerenciador centralizado de modelos de IA.

Carrega SigLIP, Qwen2.5-VL e SentenceTransformer uma unica vez e os mantem
residentes na VRAM/RAM durante todo o ciclo de vida da aplicacao, eliminando
o custo de carregamento (~4-8 s por modelo) a cada chamada de run_search /
run_index.

Uso tipico (servidor / script de longa duracao):
    from video_search.model_manager import ModelManager, ModelConfig

    with ModelManager(ModelConfig(enable_warmup=True)) as mm:
        # busca e indexacao usam mm internamente
        results = run_search(query, index_dir, model_manager=mm)

Uso via singleton (CLI / modulo):
    from video_search.model_manager import get_model_manager

    mm = get_model_manager()       # cria na 1a chamada, reutiliza nas seguintes
    # load_models() e chamado automaticamente na 1a vez que get_siglip() /
    # get_qwen() / get_sentence_transformer() sao chamados.

Hierarquia de importacao (sem importacoes circulares):
    vlm_abc.py            <- sem imports do projeto
    model_config.py       <- sem imports do projeto
    siglip_embedder.py    <- vlm_abc
    vlm_describer.py      <- vlm_abc
    model_manager.py      <- vlm_abc, model_config
                             + imports lazy de siglip_embedder, vlm_describer,
                               pipeline (CONFIDENCE_MODEL) e sentence_transformers
"""

import logging
import threading
from typing import TYPE_CHECKING, Optional

import torch

from .model_config import ModelConfig
from .vlm_abc import VisionLanguageModel  # noqa: F401 -- reexportado para conveniencia

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

    from .siglip_embedder import SigLIPEmbedder
    from .vlm_describer import Qwen2VLDescriber

logger = logging.getLogger(__name__)


class ModelManager:
    """Mantém instâncias únicas de SigLIP, Qwen2.5-VL e SentenceTransformer.

    Thread-safe: `_init_lock` serializa a inicializacao; modelos em si sao
    read-only durante inferencia (`@torch.no_grad`) e nao precisam de lock.
    O `_gpu_lock` (de pipeline.py) continua sendo usado para serializar as
    chamadas de GPU em run_index / run_search / multi_index.
    """

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        self._config = config or ModelConfig()
        self._init_lock = threading.Lock()
        self._siglip: Optional["SigLIPEmbedder"] = None
        self._qwen: Optional["Qwen2VLDescriber"] = None
        self._sentence_transformer: Optional["SentenceTransformer"] = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Ciclo de vida
    # ------------------------------------------------------------------

    def load_models(self) -> None:
        """Carrega todos os modelos. Idempotente — segundas chamadas sao no-op."""
        with self._init_lock:
            if self._loaded:
                return

            device = self._resolve_device()
            logger.info("[ModelManager] inicializando modelos no device '%s'...", device)

            from .siglip_embedder import SigLIPEmbedder

            self._siglip = SigLIPEmbedder(device=device, precision=self._config.siglip_precision)
            logger.info("[ModelManager] SigLIP pronto.")

            from .vlm_describer import Qwen2VLDescriber

            self._qwen = Qwen2VLDescriber(
                device=device,
                quantize=(self._config.qwen_precision == "int4"),
            )
            logger.info("[ModelManager] Qwen2.5-VL pronto.")

            from sentence_transformers import SentenceTransformer

            from ..utils.hf_utils import load_offline_first
            from ..indexing.pipeline import CONFIDENCE_MODEL

            self._sentence_transformer = load_offline_first(
                SentenceTransformer,
                CONFIDENCE_MODEL,
                device=self._config.sentence_transformer_device,
            )
            logger.info("[ModelManager] SentenceTransformer pronto.")

            if self._config.enable_warmup:
                self._run_warmup()

            if self._config.enable_compile:
                self._run_compile()

            self._loaded = True
            logger.info("[ModelManager] todos os modelos prontos e residentes.")

    def unload_models(self) -> None:
        """Libera pesos de todos os modelos e limpa cache CUDA."""
        with self._init_lock:
            if not self._loaded:
                return
            if self._siglip:
                self._siglip.unload()
                self._siglip = None
            if self._qwen:
                self._qwen.unload()
                self._qwen = None
            self._sentence_transformer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._loaded = False
            logger.info("[ModelManager] modelos liberados.")

    # ------------------------------------------------------------------
    # Acesso aos modelos
    # ------------------------------------------------------------------

    def get_siglip(self) -> "SigLIPEmbedder":
        """Retorna o SigLIPEmbedder carregado. Dispara load_models() se necessario."""
        self._ensure_loaded()
        return self._siglip

    def get_qwen(self) -> "Qwen2VLDescriber":
        """Retorna o Qwen2VLDescriber carregado. Dispara load_models() se necessario."""
        self._ensure_loaded()
        return self._qwen

    def get_sentence_transformer(self) -> "SentenceTransformer":
        """Retorna o SentenceTransformer carregado. Dispara load_models() se necessario."""
        self._ensure_loaded()
        return self._sentence_transformer

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "ModelManager":
        self.load_models()
        return self

    def __exit__(self, *_) -> None:
        self.unload_models()

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------

    def _resolve_device(self) -> str:
        if self._config.device == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        return self._config.device

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load_models()

    def _run_warmup(self) -> None:
        from PIL import Image

        logger.info("[ModelManager] executando warmup dos modelos...")
        dummy = Image.new("RGB", (256, 256), (128, 128, 128))
        self._siglip.warmup()
        self._qwen.warmup()
        self._sentence_transformer.encode(["warmup"])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        logger.info("[ModelManager] warmup concluido.")

    def _run_compile(self) -> None:
        if self._config.qwen_precision == "int4":
            logger.warning(
                "[ModelManager] torch.compile ignorado para Qwen: incompativel com bitsandbytes NF4."
            )
        else:
            try:
                self._qwen.model = torch.compile(self._qwen.model)
                logger.info("[ModelManager] torch.compile aplicado ao Qwen.")
            except Exception as exc:
                logger.warning("[ModelManager] torch.compile falhou para Qwen: %s", exc)
        try:
            self._siglip.model = torch.compile(self._siglip.model)
            logger.info("[ModelManager] torch.compile aplicado ao SigLIP.")
        except Exception as exc:
            logger.warning("[ModelManager] torch.compile falhou para SigLIP: %s", exc)


# ------------------------------------------------------------------
# Singleton de modulo
# ------------------------------------------------------------------

_default_manager: Optional[ModelManager] = None
_singleton_lock = threading.Lock()


def get_model_manager(config: Optional[ModelConfig] = None) -> ModelManager:
    """Retorna o ModelManager singleton do processo.

    Cria a instancia na 1a chamada e a reutiliza em todas as seguintes.
    Para configuracao nao-default, passe `config` na 1a chamada ou instancie
    `ModelManager` diretamente e passe-o aos run_index / run_search.
    Os modelos sao carregados lazily na 1a chamada a get_siglip() /
    get_qwen() / get_sentence_transformer(), ou explicitamente via
    load_models().
    """
    global _default_manager
    with _singleton_lock:
        if _default_manager is None:
            _default_manager = ModelManager(config)
    return _default_manager
