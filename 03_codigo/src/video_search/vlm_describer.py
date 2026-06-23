"""Wrapper para Qwen2.5-VL-3B-Instruct (Apache 2.0): descricao de frames e
verificacao de atributos (VQA) com grau de confianca.

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

from typing import List, Optional, Sequence

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration


_MAX_IMAGE_SIDE = 768


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
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
        quantize: bool = True,
    ) -> None:
        quantization_config = None
        if quantize:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
        self.model = (
            Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map="auto",
                quantization_config=quantization_config,
            )
            .eval()
        )
        self.device = self.model.device
        self.processor = AutoProcessor.from_pretrained(model_name)

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

    @torch.no_grad()
    def describe_for_query(
        self,
        image: Image.Image,
        query: str,
        max_new_tokens: int = 40,
    ) -> str:
        """Legenda curta e focada na consulta: descreve em 1 frase os
        elementos da imagem relacionados a `query`. Mantem especificidade
        comparavel a da query (em vez de uma legenda generica e longa), o que
        torna a similaridade texto-texto com a query mais discriminativa.

        O prompt evita a forma "Esta imagem mostra X?": essa pergunta direta
        induz confirmacao em objetos visualmente parecidos mas diferentes do
        pedido (ex.: confundiu rack de teto com escada, mesmo em resolucao
        plena) -- viés parecido ao que fez `match_confidence` ser abandonado.
        Em vez disso, pede para nomear o tipo exato do objeto visto."""
        prompt = (
            f"Estou procurando por: '{query}'. Sem assumir que isso esta na "
            "imagem, observe com atencao e descreva em 1 frase curta o que "
            "realmente aparece, nomeando o tipo exato dos objetos (ex.: "
            "moto ou bicicleta, van ou caminhonete, escada ou rack de "
            "teto). Se o que aparece for apenas parecido mas diferente do "
            "que foi pedido, diga o que e de fato, em vez de confirmar."
        )
        return self.describe(image, prompt=prompt, max_new_tokens=max_new_tokens)

    @torch.no_grad()
    def match_confidence(self, image: Image.Image, description: str) -> float:
        """Confianca (0-1) de que `description` se aplica a imagem, baseada na
        probabilidade relativa do primeiro token gerado ser 'Sim' vs 'Nao'
        em resposta a uma pergunta de sim/nao."""
        question = (
            f'A seguinte descricao se aplica a esta imagem? "{description}". '
            "Responda apenas com Sim ou Nao."
        )
        inputs = self._prepare_inputs(image, question)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1,
            output_scores=True,
            return_dict_in_generate=True,
        )
        # log_softmax em float32 sobre o vocabulario completo evita o
        # underflow de softmax(logits) quando os logits brutos tem
        # magnitude grande (ex.: ~2000) e a lacuna Sim/Nao excede a faixa
        # representavel em bf16/fp32 apos exponenciar.
        log_probs = torch.log_softmax(outputs.scores[0][0].float(), dim=-1)

        yes_ids = self._token_ids(["Sim", " Sim", "sim", " sim", "Yes", " Yes", "yes", " yes"])
        no_ids = self._token_ids(["Nao", " Nao", "Não", " Não", "nao", " nao", "não", " não", "No", " No", "no", " no"])

        yes_logprob = torch.logsumexp(log_probs[yes_ids], dim=0)
        no_logprob = torch.logsumexp(log_probs[no_ids], dim=0)

        probs = torch.softmax(torch.stack([yes_logprob, no_logprob]), dim=0)
        return float(probs[0].item())

    def _token_ids(self, words: Sequence[str]) -> List[int]:
        ids = set()
        for word in words:
            encoded = self.processor.tokenizer.encode(word, add_special_tokens=False)
            if len(encoded) == 1:
                ids.add(encoded[0])
        return list(ids)
