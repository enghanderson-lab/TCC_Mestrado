"""Teste de sanidade: carrega o ClipEmbedder e verifica similaridade
imagem-texto em imagens sinteticas (sem depender de video real)."""

import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_search.embedder import ClipEmbedder  # noqa: E402


def make_solid_image(color, size=(224, 224)):
    return Image.new("RGB", size, color=color)


def main():
    embedder = ClipEmbedder()
    print(f"Dispositivo: {embedder.device}")

    images = [
        make_solid_image((220, 20, 20)),   # vermelho
        make_solid_image((20, 20, 220)),   # azul
        make_solid_image((255, 255, 255)),  # branco
    ]
    labels = ["vermelho", "azul", "branco"]

    queries = [
        "uma imagem vermelha",
        "uma imagem azul",
        "a white image",
    ]

    image_embeddings = embedder.encode_images(images)
    text_embeddings = embedder.encode_text(queries)

    similarity = image_embeddings @ text_embeddings.T

    print("\nMatriz de similaridade (linhas=imagens, colunas=queries):")
    print("           " + "  ".join(f"{q:>18s}" for q in queries))
    for label, row in zip(labels, similarity):
        print(f"{label:>10s} " + "  ".join(f"{v:18.4f}" for v in row))


if __name__ == "__main__":
    main()
