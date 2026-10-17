"""Caso de uso: consulta e remoção de câmeras indexadas."""

from typing import List, Optional, Sequence

from ..domain.entities import Camera
from ..domain.ports import CameraRepository, IndexRepository


class CameraService:
    def __init__(self, cameras: CameraRepository, indexes: IndexRepository) -> None:
        self._cameras = cameras
        self._indexes = indexes

    def list(self) -> List[Camera]:
        return self._cameras.list_all()

    def get(self, camera_id: str) -> Optional[Camera]:
        return self._cameras.get(camera_id)

    def select(self, camera_ids: Sequence[str]) -> List[Camera]:
        """Câmeras com os ids pedidos; lista vazia de ids = todas."""
        cameras = self._cameras.list_all()
        if not camera_ids:
            return cameras
        wanted = set(camera_ids)
        return [c for c in cameras if c.camera_id in wanted]

    def delete(self, camera_id: str) -> bool:
        """Remove o registro e o índice em disco; False se não existir."""
        camera = self._cameras.get(camera_id)
        if camera is None:
            return False
        self._indexes.delete(camera.output_dir)
        self._cameras.delete(camera_id)
        return True
