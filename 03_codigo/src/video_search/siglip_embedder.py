"""SigLIP embedder: retrieval multilingue com melhor qualidade que CLIP.

Modelo  : google/siglip-base-patch16-256-multilingual (Apache 2.0)
Licenca : Apache 2.0
Dim     : 768 (vs 512 do CLIP ViT-B/32)
Entrada : 256x256 px (vs 224x224 do CLIP)

Vantagens sobre CLIP/M-CLIP:
  - Treinado com sigmoid loss (SigLIP) em vez de softmax (CLIP), o que melhora
    precisao em retrieval — o modelo aprende a pontuar cada par independentemente
    em vez de competir dentro de um batch.
  - Suporte multiligue nativo (PT, EN, e outros) sem traducao.
  - Scores de similaridade tipicamente mais altos e mais discriminativos.

Nota de implementacao: usamos vision_model/text_model + projecoes diretamente
em vez de get_image_features/get_text_features, pois no transformers 5.x esses
metodos retornam dataclasses em vez de tensors quando carregados via AutoModel.
"""

from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from transformers import SiglipModel, AutoProcessor

from .hf_utils import load_offline_first


class SigLIPEmbedder:
    MODEL_NAME = "google/siglip-base-patch16-256-multilingual"
    EMBEDDING_DIM = 768
    TEXT_MAX_LENGTH = 64

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = (
            load_offline_first(
                SiglipModel.from_pretrained, self.MODEL_NAME, torch_dtype=torch.bfloat16
            )
            .to(self.device)
            .eval()
        )
        self.processor = load_offline_first(AutoProcessor.from_pretrained, self.MODEL_NAME)

    def _encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # SigLIP nao tem camadas de projecao separadas — pooler_output ja e o
        # embedding no espaco conjunto imagem/texto.
        return self.model.vision_model(pixel_values=pixel_values).pooler_output

    def _encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model.text_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output

    def encode_text(self, texts: List[str]) -> np.ndarray:
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding="max_length",
            max_length=self.TEXT_MAX_LENGTH,
            truncation=True,
        ).to(self.device)
        input_ids = inputs["input_ids"]
        # SiglipTokenizer com padding=max_length pode omitir attention_mask
        # (todos os tokens no mesmo comprimento nao precisam de mascara).
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
        with torch.no_grad():
            features = self._encode_text(input_ids, attention_mask)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().float().numpy().astype(np.float32)

    def encode_images(self, images: List[Image.Image]) -> np.ndarray:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            features = self._encode_image(inputs["pixel_values"])
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().float().numpy().astype(np.float32)
