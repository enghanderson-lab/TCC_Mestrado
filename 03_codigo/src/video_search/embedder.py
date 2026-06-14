"""Wrapper para gerar embeddings de imagens e texto com OpenCLIP (licença MIT)."""

from typing import List, Optional, Sequence

import numpy as np
import open_clip
import torch
from PIL import Image


class ClipEmbedder:
    """Gera embeddings normalizados de imagens e texto no mesmo espaço vetorial."""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: Optional[str] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        batch = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        features = self.model.encode_image(batch)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> np.ndarray:
        tokens = self.tokenizer(texts).to(self.device)
        features = self.model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()
