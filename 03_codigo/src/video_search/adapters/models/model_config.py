"""Configuracao centralizada dos modelos de IA do pipeline.

Unica fonte de verdade para device, precisao, warmup e compile.
Passada ao ModelManager na inicializacao da aplicacao.
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
        "int8"  LLM.int8 via bitsandbytes (~3.6 GB VRAM).
        "bf16"  Pesos completos (~6 GB VRAM, sem quantizacao).

    enable_warmup
        Se True, executa uma inferencia ficticia apos carregar cada modelo
        para inicializar kernels CUDA. Elimina a latencia extra da primeira
        busca real (tipicamente 1-3 s no primeiro generate() do Qwen).

    enable_compile
        Se True, aplica torch.compile() ao SigLIP. Pode melhorar throughput
        em batchs repetidos apos uma compilacao inicial (~10-30 s). Ignorado
        para Qwen quando qwen_precision="int4" (incompativel com bitsandbytes).

    sentence_transformer_device
        Device para o modelo de confianca (sentence-transformers). "cpu" por
        default: o modelo e leve e nao vale competir com SigLIP/Qwen pela VRAM.
    """

    device: str = "auto"
    siglip_precision: Literal["bf16", "fp16", "fp32"] = "bf16"
    qwen_precision: Literal["int4", "int8", "bf16"] = "int4"
    enable_warmup: bool = True
    enable_compile: bool = False
    sentence_transformer_device: str = "cpu"
