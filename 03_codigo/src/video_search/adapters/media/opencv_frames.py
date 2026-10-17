"""FrameSource via OpenCV: amostragem a intervalos regulares e acesso
aleatório por timestamp."""

from typing import Iterator, List, Sequence

import cv2
from PIL import Image

from ...domain.entities import Frame


def _open(video_path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Não foi possível abrir o vídeo: {video_path}")
    return cap


class OpenCVFrameSource:
    def iter_frames(self, video_path: str, interval_sec: float = 2.0) -> Iterator[Frame]:
        """Itera sobre frames do vídeo, amostrados a cada `interval_sec` segundos."""
        cap = _open(video_path)
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
                    image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                    yield Frame(index=sample_index, timestamp_sec=frame_no / fps, image=image)
                    sample_index += 1
                frame_no += 1
        finally:
            cap.release()

    def frames_at(self, video_path: str, timestamps_sec: Sequence[float]) -> List[Image.Image]:
        """Acesso aleatório a vários timestamps com um único VideoCapture aberto.

        Reabrir o arquivo (cv2.VideoCapture) por timestamp é caro (parse do
        container + init do codec a cada chamada); abrir uma vez e seekar
        `len(timestamps_sec)` vezes no mesmo capture evita esse custo repetido
        quando há vários resultados do mesmo vídeo (ex.: legendar o top-K de
        uma busca). Os timestamps são processados em ordem crescente (seeks
        monotônicos tendem a ser mais baratos que saltar para trás e para
        frente), mas o retorno preserva a ordem de entrada."""
        cap = _open(video_path)
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

    def estimate_samples(self, video_path: str, interval_sec: float) -> int:
        """Nº aproximado de frames que `iter_frames` vai produzir (para a
        barra de progresso), a partir da duração do vídeo."""
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        cap.release()
        duration = n_frames / fps if fps > 0 else 0.0
        return max(1, int(duration / interval_sec))
