"""Wrapper para Qwen2.5-VL-3B-Instruct (Apache 2.0): legenda de frames
condicionada a query da busca.

Modelo 3B escolhido para caber em GPUs de 6-8GB VRAM (RTX 3060/4060):
  - 3B em bfloat16 ocupa ~6GB, deixando pouca margem para KV-cache,
    ativacoes e os outros modelos (embedder de retrieval, sentence-
    transformer) que tambem disputam VRAM durante a busca.
  - Por isso o default e carregar em 4-bit (bitsandbytes, NF4) — cai para
    ~2GB de pesos, com perda de qualidade desprezivel para legendas curtas
    e classificacao Sim/Nao. Da bastante margem ate numa RTX 3060 6GB
    (laptop), o pior caso considerado.
  - 7B em bfloat16 exigiria ~14GB (OOM mesmo numa 4060); 7B em 4-bit (~5GB)
    poderia ser viavel, mas nao foi testado."""

import time
from pathlib import Path
from typing import List, Optional, Union

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

from .hf_utils import load_offline_first

_MAX_IMAGE_SIDE = 768

# Cache local dos pesos JA quantizados em NF4 (mesmo checkpoint e mesma
# BitsAndBytesConfig do model_name default abaixo -- so persistido pos-
# quantizacao). A quantizacao bnb 4-bit roda na CPU e custa ~25s a cada
# `from_pretrained`; salvando o resultado uma vez (ver `_load_or_quantize`),
# as proximas execucoes carregam os pesos ja quantizados em poucos
# segundos, em vez de requantizar do zero a cada busca.
_DEFAULT_QUANT_CACHE_DIR = Path(__file__).resolve().parents[5] / "04_dados" / "models" / "qwen25vl3b-nf4"


def _resize_for_vlm(image: Image.Image, max_side: int = _MAX_IMAGE_SIDE) -> Image.Image:
    """Limita o lado maior do frame antes do Qwen2.5-VL.

    Sem isso, o qwen_vl_utils aceita imagens de altissima resolucao (ex.:
    2688x1520 de videos 4K) e gera milhares de tokens visuais por frame
    (ate 16384), o que faz a VRAM da RTX 4060 (8GB) estourar durante o
    generate() e o driver da NVIDIA cair para "shared GPU memory" no
    Windows — isso trava a maquina inteira, nao so o processo Python."""
    width, height = image.size
    scale = max_side / max(width, height)
    if scale >= 1:
        return image
    return image.resize((round(width * scale), round(height * scale)))


class Qwen2VLDescriber:
    """VLM de legendagem condicionada a query, baseado no Qwen2.5-VL-3B.

    Ciclo de vida: load() / warmup() / unload() (chamados pelo ModelManager).
    A busca usa describe_for_query_batch(); describe() serve ao warmup e a
    legendas genericas (ver scripts/benchmark_optimizations.py).
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
        quantize: bool = True,
        quant_cache_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self._model_name = model_name
        # device_map="auto" manda o accelerate inspecionar a memoria livre de
        # todos os devices (GPU+CPU) e planejar onde colocar cada camada,
        # mesmo havendo só uma GPU -- esse planejamento custa alguns segundos
        # de carregamento sem trazer beneficio quando o modelo (3B em 4-bit,
        # ~2GB) sempre cabe inteiro na GPU. Apontar o device explicitamente
        # pula essa etapa.
        self._device_str = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self._dtype = dtype
        self._quantize = quantize
        self._quant_cache_dir = Path(quant_cache_dir) if quant_cache_dir else _DEFAULT_QUANT_CACHE_DIR
        self.model: Optional[Qwen2_5_VLForConditionalGeneration] = None
        self.processor = None
        self.load()

    # ------------------------------------------------------------------
    # Ciclo de vida
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Carrega pesos do Qwen2.5-VL para o device configurado. Idempotente."""
        if self.model is not None:
            return

        cache_dir = self._quant_cache_dir
        if self._quantize and (cache_dir / "config.json").exists():
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                str(cache_dir), device_map=self._device_str, attn_implementation="sdpa"
            ).eval()
            self.processor = AutoProcessor.from_pretrained(str(cache_dir))
        else:
            quantization_config = None
            if self._quantize:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=self._dtype,
                    bnb_4bit_use_double_quant=True,
                )
            self.model = load_offline_first(
                Qwen2_5_VLForConditionalGeneration.from_pretrained,
                self._model_name,
                torch_dtype=self._dtype,
                device_map=self._device_str,
                quantization_config=quantization_config,
                attn_implementation="sdpa",
            ).eval()
            self.processor = load_offline_first(AutoProcessor.from_pretrained, self._model_name)
            if self._quantize:
                # 1a execucao: persiste os pesos JA quantizados em
                # `cache_dir`, para que as proximas instancias entrem no
                # ramo acima e pulem a (re)quantizacao na CPU (~25s).
                print(f"[Qwen2VLDescriber] salvando cache 4-bit em '{cache_dir}' (so na 1a execucao)...")
                cache_dir.mkdir(parents=True, exist_ok=True)
                self.model.save_pretrained(str(cache_dir))
                self.processor.save_pretrained(str(cache_dir))

        self.device = self.model.device

    def warmup(self) -> None:
        """Inferencia ficticia com 1 token para inicializar kernels CUDA do Qwen."""
        dummy = Image.new("RGB", (64, 64), (128, 128, 128))
        self.describe(dummy, max_new_tokens=1)

    def unload(self) -> None:
        """Libera o modelo da VRAM."""
        self.model = None
        self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Legendagem
    # ------------------------------------------------------------------

    @staticmethod
    def _build_messages(image: Image.Image, prompt: str) -> List[dict]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def _prepare_inputs(self, image: Image.Image, prompt: str):
        image = _resize_for_vlm(image)
        messages = self._build_messages(image, prompt)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return inputs.to(self.device)

    @torch.no_grad()
    def describe(
        self,
        image: Image.Image,
        prompt: str = "Descreva esta imagem em uma frase, focando em pessoas, roupas e objetos visiveis.",
        max_new_tokens: int = 64,
    ) -> str:
        inputs = self._prepare_inputs(image, prompt)
        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        trimmed = output_ids[:, inputs["input_ids"].shape[1] :]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    @staticmethod
    def _query_prompt(query: str) -> str:
        """Prompt da legenda condicionada a query.

        Gera 1 frase curta com especificidade comparavel a da query (em vez
        de uma legenda generica e longa), o que torna a similaridade
        texto-texto com a query mais discriminativa.

        O prompt evita a forma "Esta imagem mostra X?": essa pergunta direta
        induz confirmacao em objetos visualmente parecidos mas diferentes do
        pedido (ex.: confundiu rack de teto com escada, mesmo em resolucao
        plena) -- o mesmo vies que fez a abordagem VQA Sim/Nao ser
        abandonada. Em vez disso, pede para nomear o tipo exato do objeto."""
        return (
            f"Estou procurando por: '{query}'. Sem assumir que isso esta na "
            "imagem, observe com atencao e descreva em 1 frase curta o que "
            "realmente aparece, nomeando o tipo exato dos objetos (ex.: "
            "moto ou bicicleta, van ou caminhonete, escada ou rack de "
            "teto). Se o que aparece for apenas parecido mas diferente do "
            "que foi pedido, diga o que e de fato, em vez de confirmar."
        )

    @torch.no_grad()
    def describe_for_query_batch(
        self,
        images: List[Image.Image],
        query: str,
        max_new_tokens: int = 40,
        profiler=None,
    ) -> List[str]:
        """Legendas condicionadas a `query` (ver `_query_prompt`) para todas
        as `images` numa unica chamada a `generate()`.

        `profiler` (Optional[ProfilingContext]): registra qwen_preprocess e
        qwen_infer separadamente, e incrementa qwen_calls.
        """
        if not images:
            return []
        prompt = self._query_prompt(query)
        resized = [_resize_for_vlm(image) for image in images]
        messages_batch = [self._build_messages(image, prompt) for image in resized]
        texts = [
            self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in messages_batch
        ]
        image_inputs, video_inputs = process_vision_info(messages_batch)
        self.processor.tokenizer.padding_side = "left"

        t0 = time.perf_counter()
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        if profiler:
            profiler.add_time("qwen_preprocess", time.perf_counter() - t0, len(images))

        t1 = time.perf_counter()
        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        if profiler:
            profiler.add_time("qwen_infer", time.perf_counter() - t1, len(images))
            profiler.count("qwen_calls", len(images))

        trimmed = output_ids[:, inputs["input_ids"].shape[1] :]
        return [c.strip() for c in self.processor.batch_decode(trimmed, skip_special_tokens=True)]
