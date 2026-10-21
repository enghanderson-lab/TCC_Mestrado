"""Recortes sobrepostos de um frame, para ampliar alvos pequenos.

O frame vai ao VLM reduzido para no máximo 768 px (limite de VRAM, ver
adapters/models/qwen.py): num 1080p, uma pessoa a 10 m da câmera vira um
borrão de ~40 px. Dividir o frame numa grade de recortes sobrepostos e
mandar ao VLM o recorte mais parecido com a consulta dá ~2× de resolução
efetiva sobre o alvo, sem aumentar a imagem inteira.
"""

from dataclasses import dataclass
from typing import List, Tuple

Box = Tuple[int, int, int, int]  # (esquerda, topo, direita, base), como em PIL.Image.crop


@dataclass(frozen=True)
class ZoomConfig:
    grid: int = 3            # recortes por lado (3 → 9 recortes)
    tile_fraction: float = 0.5  # lado do recorte ÷ lado do frame; 0,5 com grade 3 = 50% de sobreposição


def tile_boxes(width: int, height: int, config: ZoomConfig = ZoomConfig()) -> List[Box]:
    """Caixas da grade `grid × grid`, cada uma com `tile_fraction` do frame,
    distribuídas uniformemente de borda a borda (a sobreposição garante que
    um alvo na fronteira entre duas células caiba inteiro em algum recorte)."""
    tw = max(1, round(width * config.tile_fraction))
    th = max(1, round(height * config.tile_fraction))
    n = max(1, config.grid)

    def starts(full: int, tile: int) -> List[int]:
        if n == 1 or tile >= full:
            return [0]
        step = (full - tile) / (n - 1)
        return [round(i * step) for i in range(n)]

    return [(x, y, x + tw, y + th) for y in starts(height, th) for x in starts(width, tw)]
