"""Teste de sanidade: carrega o Qwen2-VL-2B-Instruct e verifica
descricao de imagem + confianca de correspondencia (VQA sim/nao)."""

import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_search.models.vlm_describer import Qwen2VLDescriber  # noqa: E402


def make_scene_image(size=(384, 384)):
    """Imagem sintetica simples: figura humanoide com camisa branca e
    'bone' vermelho sobre fundo cinza, para testar reconhecimento de
    atributos compostos."""
    img = Image.new("RGB", size, color=(120, 120, 120))
    draw = ImageDraw.Draw(img)

    # cabeca
    draw.ellipse((150, 60, 230, 140), fill=(230, 200, 170))
    # bone vermelho
    draw.rectangle((150, 50, 230, 80), fill=(200, 20, 20))
    # corpo - camisa branca
    draw.rectangle((130, 140, 250, 300), fill=(245, 245, 245))
    # calca azul
    draw.rectangle((140, 300, 240, 380), fill=(30, 60, 150))

    return img


def main():
    describer = Qwen2VLDescriber()
    print(f"Dispositivo: {describer.device}")

    image = make_scene_image()

    print("\n--- describe() ---")
    description = describer.describe(image)
    print(description)

    print("\n--- match_confidence() ---")
    candidates = [
        "homem de camisa branca e bone vermelho",
        "homem de camisa preta e bone azul",
        "uma pessoa usando oculos de sol",
    ]
    for text in candidates:
        score = describer.match_confidence(image, text)
        print(f"  {score:6.1%}  '{text}'")


if __name__ == "__main__":
    main()
