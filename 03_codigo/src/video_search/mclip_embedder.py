"""M-CLIP embedder: retrieval multilingue nativo sem traducao.

- Texto  : sentence-transformers/clip-ViT-B-32-multilingual-v1
           (XLM-RoBERTa fine-tuned para o espaco do CLIP, Apache 2.0)
- Imagem : OpenAI CLIP ViT-B/32 via open_clip (espaco compativel com o texto)

Motivacao: CLIP (laion2b) foi treinado majoritariamente em ingles, o que
degrada queries em portugues. Este embedder processa PT diretamente.
"""

from typing import List, Optional

import numpy as np
import open_clip
import torch
from PIL import Image


class MCLIPEmbedder:
    TEXT_MODEL = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
    IMAGE_MODEL = "ViT-B-32"
    IMAGE_PRETRAINED = "openai"
    EMBEDDING_DIM = 512

    def __init__(self, device: Optional[str] = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.text_model = SentenceTransformer(self.TEXT_MODEL, device=self.device)

        self.vision_model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.IMAGE_MODEL, pretrained=self.IMAGE_PRETRAINED
        )
        self.vision_model = self.vision_model.to(self.device).eval()

    def encode_text(self, texts: List[str]) -> np.ndarray:
        return self.text_model.encode(texts, normalize_embeddings=True).astype(np.float32)

    def encode_images(self, images: List[Image.Image]) -> np.ndarray:
        tensors = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        with torch.no_grad():
            embeddings = self.vision_model.encode_image(tensors)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings.cpu().numpy().astype(np.float32)
