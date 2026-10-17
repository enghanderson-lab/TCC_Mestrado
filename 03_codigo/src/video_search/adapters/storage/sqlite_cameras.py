"""CameraRepository em SQLite (tabela `cameras`)."""

from typing import List, Optional

from ...domain.entities import Camera
from .sqlite import SQLiteDatabase


class SQLiteCameraRepository:
    def __init__(self, db: SQLiteDatabase) -> None:
        self._db = db
        with self._db.write() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cameras (
                    camera_id  TEXT PRIMARY KEY,
                    video_path TEXT NOT NULL,
                    output_dir TEXT NOT NULL,
                    frame_count INTEGER DEFAULT 0,
                    indexed_at  TEXT,
                    status      TEXT DEFAULT 'indexed'
                )
            """)

    def upsert(self, camera: Camera) -> None:
        with self._db.write() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO cameras
                    (camera_id, video_path, output_dir, frame_count, indexed_at, status)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (camera.camera_id, camera.video_path, camera.output_dir,
                 camera.frame_count, camera.indexed_at, camera.status),
            )

    def get(self, camera_id: str) -> Optional[Camera]:
        row = self._db.conn().execute(
            "SELECT * FROM cameras WHERE camera_id = ?", (camera_id,)
        ).fetchone()
        return Camera(**dict(row)) if row else None

    def list_all(self) -> List[Camera]:
        rows = self._db.conn().execute(
            "SELECT * FROM cameras ORDER BY indexed_at DESC"
        ).fetchall()
        return [Camera(**dict(r)) for r in rows]

    def delete(self, camera_id: str) -> bool:
        with self._db.write() as conn:
            cur = conn.execute("DELETE FROM cameras WHERE camera_id = ?", (camera_id,))
        return cur.rowcount > 0
