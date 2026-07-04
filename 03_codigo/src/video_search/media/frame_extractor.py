"""Extração de frames de um vídeo a intervalos regulares."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence, Union

import cv2
from PIL import Image


@dataclass
class Frame:
    index: int
    timestamp_sec: float
    image: Image.Image


def extract_frame_at(video_path: Union[str, Path], timestamp_sec: float) -> Image.Image:
    """Acesso aleatorio: retorna o frame mais proximo do timestamp informado."""
    return extract_frames_at(video_path, [timestamp_sec])[0]


def extract_frames_at(video_path: Union[str, Path], timestamps_sec: Sequence[float]) -> List[Image.Image]:
    """Acesso aleatorio a varios timestamps com um unico VideoCapture aberto.

    Reabrir o arquivo (cv2.VideoCapture) por timestamp é caro (parse do
    container + init do codec a cada chamada); abrir uma vez e seekar
    `len(timestamps_sec)` vezes no mesmo capture evita esse custo repetido
    quando ha varios resultados do mesmo video (ex.: legendar o top-K de uma
    busca). Os timestamps sao processados em ordem crescente (seeks
    monotonicos tendem a ser mais baratos que saltar para tras e para
    frente), mas o retorno preserva a ordem de entrada."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Não foi possível abrir o vídeo: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        images: List[Image.Image] = [None] * len(timestamps_sec)  # type: ignore[list-item]
        for i in sorted(range(len(timestamps_sec)), key=lambda i: timestamps_sec[i]):
            timestamp_sec = timestamps_sec[i]
            cap.set(cv2.CAP_PROP_POS_FRAMES, round(timestamp_sec * fps))
            ok, bgr = cap.read()
            if not ok:
                raise ValueError(f"Não foi possível ler o frame em t={timestamp_sec:.2f}s de {video_path}")
            images[i] = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        return images
    finally:
        cap.release()


def extract_frames(video_path: Union[str, Path], interval_sec: float = 2.0) -> Iterator[Frame]:
    """Itera sobre frames do vídeo, amostrados a cada `interval_sec` segundos."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Não foi possível abrir o vídeo: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_step = max(1, round(fps * interval_sec))

    frame_no = 0
    sample_index = 0
    try:
        while True:
            ret, bgr = cap.read()
            if not ret:
                break
            if frame_no % frame_step == 0:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb)
                timestamp_sec = frame_no / fps
                yield Frame(index=sample_index, timestamp_sec=timestamp_sec, image=image)
                sample_index += 1
            frame_no += 1
    finally:
        cap.release()
