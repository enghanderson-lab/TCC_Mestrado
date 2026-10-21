"""Configuracao centralizada dos modelos de IA do pipeline.

Unica fonte de verdade para device, precisao e warmup. Passada ao
ModelManager na inicializacao da aplicacao.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class ModelConfig:
    """Parametros de carregamento e execucao dos modelos.

    device
        "auto"    -> cuda:0 se disponivel, senao cpu.
        "cuda:0"  -> GPU explicita.
        "cpu"     -> forcado em CPU (lento, so para testes).

    siglip_precision
        Precisao dos pesos do encoder de retrieval (SigLIP).
        "bf16" e o default e funciona em todas as GPUs Ampere+.

    qwen_precision
        Precisao do VLM de legendagem (Qwen2.5-VL-3B):
        "int4"  NF4 4-bit via bitsandbytes (~2.4 GB VRAM). Default.
        "bf16"  Pesos completos (~6 GB VRAM, sem quantizacao).

    enable_warmup
        Se True, executa uma inferencia ficticia apos carregar cada modelo
        para inicializar kernels CUDA. Elimina a latencia extra da primeira
        busca real (tipicamente 1-3 s no primeiro generate() do Qwen).

    scorer_device
        Device do modelo de confianca (NLI, mDeBERTa-v3-base). "cpu" por
        default: em float32 pontua 5 legendas em ~0,25 s e nao disputa VRAM
        com SigLIP/Qwen.
    """

    device: str = "auto"
    siglip_precision: Literal["bf16", "fp16", "fp32"] = "bf16"
    qwen_precision: Literal["int4", "bf16"] = "int4"
    enable_warmup: bool = True
    scorer_device: str = "cpu"
