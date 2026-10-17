"""JobRepository em SQLite (tabela `jobs`, no mesmo banco das câmeras).

Persistir os jobs faz o status sobreviver a um reinício da API; o
`cancel_event` é estado de execução e não é persistido.
"""

from typing import Optional

from ...domain.entities import IndexJob, JobStatus
from .sqlite import SQLiteDatabase

_COLUMNS = (
    "job_id", "camera_id", "status", "frames_indexed", "frames_estimated",
    "error_msg", "started_at", "finished_at",
)


class SQLiteJobRepository:
    def __init__(self, db: SQLiteDatabase) -> None:
        self._db = db
        with self._db.write() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id           TEXT PRIMARY KEY,
                    camera_id        TEXT NOT NULL,
                    status           TEXT NOT NULL,
                    frames_indexed   INTEGER DEFAULT 0,
                    frames_estimated INTEGER DEFAULT 0,
                    error_msg        TEXT,
                    started_at       REAL NOT NULL,
                    finished_at      REAL
                )
            """)

    def save(self, job: IndexJob) -> None:
        with self._db.write() as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO jobs ({', '.join(_COLUMNS)}) "
                f"VALUES ({', '.join('?' * len(_COLUMNS))})",
                tuple(getattr(job, c) for c in _COLUMNS),
            )

    def get(self, job_id: str) -> Optional[IndexJob]:
        row = self._db.conn().execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return IndexJob(**dict(row)) if row else None

    def mark_interrupted(self) -> int:
        with self._db.write() as conn:
            cur = conn.execute(
                "UPDATE jobs SET status = ?, error_msg = ?, "
                "finished_at = COALESCE(finished_at, started_at) WHERE status = ?",
                (JobStatus.ERROR, "interrompido: a API foi reiniciada durante a indexação",
                 JobStatus.RUNNING),
            )
        return cur.rowcount
