"""Interface abstrata para modelos multimodais gerenciados pelo ModelManager.

Separada de model_manager.py para evitar importacao circular:
    siglip_embedder / vlm_describer  ->  vlm_abc
    model_manager                    ->  vlm_abc, siglip_embedder, vlm_describer
"""

from abc import ABC, abstractmethod


class VisionLanguageModel(ABC):
    """Contrato de ciclo de vida que todos os modelos do pipeline devem cumprir.

    load()    -- carrega pesos para o device; idempotente (segunda chamada e no-op).
    warmup()  -- inferencia ficticia para inicializar kernels CUDA; chamada
                 automaticamente pelo ModelManager se enable_warmup=True.
    infer()   -- ponto de entrada unificado; kwargs variam por modelo (ver docstring
                 das subclasses SigLIPEmbedder e Qwen2VLDescriber).
    unload()  -- libera tensores e limpa cache CUDA; apos isso load() pode
                 ser chamado novamente.
    """

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def warmup(self) -> None: ...

    @abstractmethod
    def infer(self, **kwargs): ...

    @abstractmethod
    def unload(self) -> None: ...
