"""Teste de sanidade do pipeline v2: legenda (Qwen2-VL) + similaridade
textual (CLIP) como score de confianca, comparado ao score CLIP imagem-texto.

Gera 4 cenas sinteticas, descreve cada uma com o Qwen2-VL e calcula, para a
mesma query, tanto o score CLIP imagem-texto (ranking) quanto o novo score
de confianca via similaridade texto-texto (query x legenda)."""

import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_search.embedder import ClipEmbedder  # noqa: E402
from video_search.vlm_describer import Qwen2VLDescriber  # noqa: E402


def make_scene_image(shirt_color=None, cap_color=None, size=(384, 384)):
    """Imagem sintetica com figura humanoide (cabeca, bone, camisa, calca)
    sobre fundo cinza. Se `shirt_color` for None, gera apenas o fundo (cena
    sem pessoa)."""
    img = Image.new("RGB", size, color=(120, 120, 120))
    draw = ImageDraw.Draw(img)

    if shirt_color is None:
        # Cena sem pessoa: apenas um objeto generico (moveis) no fundo.
        draw.rectangle((100, 250, 300, 380), fill=(110, 70, 40))
        return img

    # cabeca
    draw.ellipse((150, 60, 230, 140), fill=(230, 200, 170))
    # bone
    draw.rectangle((150, 50, 230, 80), fill=cap_color)
    # corpo - camisa
    draw.rectangle((130, 140, 250, 300), fill=shirt_color)
    # calca azul
    draw.rectangle((140, 300, 240, 380), fill=(30, 60, 150))

    return img


WHITE = (245, 245, 245)
BLACK = (20, 20, 20)
RED = (200, 20, 20)
BLUE = (40, 70, 180)


def main():
    embedder = ClipEmbedder()
    describer = Qwen2VLDescriber()
    print(f"CLIP device: {embedder.device} | Qwen2-VL device: {describer.device}")

    query = "homem de camisa branca e bone vermelho"
    query_embedding = embedder.encode_text([query])[0]

    scenes = {
        "branca+vermelho (match)": make_scene_image(WHITE, RED),
        "preta+azul (mismatch)": make_scene_image(BLACK, BLUE),
        "sem pessoa (mismatch)": make_scene_image(None, None),
        "branca+azul (parcial)": make_scene_image(WHITE, BLUE),
    }

    print(f"\nQuery: '{query}'\n")
    for name, image in scenes.items():
        image_embedding = embedder.encode_images([image])[0]
        clip_score = float(query_embedding @ image_embedding)

        caption = describer.describe(image)
        caption_embedding = embedder.encode_text([caption])[0]
        text_confidence = float(query_embedding @ caption_embedding)

        print(f"=== {name} ===")
        print(f"  legenda: {caption}")
        print(f"  clip_score (img-texto)      = {clip_score:.4f}")
        print(f"  confianca_legenda (txt-txt) = {text_confidence:.4f}")
        print()


if __name__ == "__main__":
    main()
