import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class CameraRecord:
    camera_id: str
    video_path: str
    output_dir: str
    frame_count: int = 0
    indexed_at: Optional[str] = None
    status: str = "indexed"


class CameraDB:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
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

    @contextmanager
    def _connect(self):
        with self._lock:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()

    def upsert(self, rec: CameraRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO cameras
                    (camera_id, video_path, output_dir, frame_count, indexed_at, status)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (rec.camera_id, rec.video_path, rec.output_dir,
                 rec.frame_count, rec.indexed_at, rec.status),
            )

    def get(self, camera_id: str) -> Optional[CameraRecord]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM cameras WHERE camera_id = ?", (camera_id,)
            ).fetchone()
            return CameraRecord(**dict(row)) if row else None

    def list_all(self) -> List[CameraRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM cameras ORDER BY indexed_at DESC"
            ).fetchall()
            return [CameraRecord(**dict(r)) for r in rows]

    def delete(self, camera_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM cameras WHERE camera_id = ?", (camera_id,)
            )
            return cur.rowcount > 0
