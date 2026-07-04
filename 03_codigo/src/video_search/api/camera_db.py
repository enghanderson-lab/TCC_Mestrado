import sqlite3
import threading
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
    """Registro SQLite de câmeras indexadas.

    Conexão persistente por thread (WAL) via threading.local(), mesmo padrão
    de `indexing/embedding_cache.py`: lock explícito só nas escritas
    (upsert/delete); leituras usam a conexão da própria thread sem bloquear
    entre si.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._write_lock = threading.Lock()
        self._local = threading.local()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    def _init_db(self) -> None:
        with self._write_lock:
            conn = self._conn()
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
            conn.commit()

    def upsert(self, rec: CameraRecord) -> None:
        with self._write_lock:
            conn = self._conn()
            conn.execute(
                """
                INSERT OR REPLACE INTO cameras
                    (camera_id, video_path, output_dir, frame_count, indexed_at, status)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (rec.camera_id, rec.video_path, rec.output_dir,
                 rec.frame_count, rec.indexed_at, rec.status),
            )
            conn.commit()

    def get(self, camera_id: str) -> Optional[CameraRecord]:
        row = self._conn().execute(
            "SELECT * FROM cameras WHERE camera_id = ?", (camera_id,)
        ).fetchone()
        return CameraRecord(**dict(row)) if row else None

    def list_all(self) -> List[CameraRecord]:
        rows = self._conn().execute(
            "SELECT * FROM cameras ORDER BY indexed_at DESC"
        ).fetchall()
        return [CameraRecord(**dict(r)) for r in rows]

    def delete(self, camera_id: str) -> bool:
        with self._write_lock:
            conn = self._conn()
            cur = conn.execute(
                "DELETE FROM cameras WHERE camera_id = ?", (camera_id,)
            )
            conn.commit()
            return cur.rowcount > 0

    def close(self) -> None:
        if hasattr(self._local, "conn"):
            self._local.conn.close()
            del self._local.conn
