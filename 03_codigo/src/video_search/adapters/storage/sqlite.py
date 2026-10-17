"""Conexão SQLite compartilhada pelos repositórios (câmeras, jobs).

Conexão persistente por thread (WAL) via threading.local(); lock explícito
só nas escritas — leituras usam a conexão da própria thread sem bloquear
entre si.
"""

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Union


class SQLiteDatabase:
    def __init__(self, path: Union[str, Path]) -> None:
        self._path = str(path)
        self._write_lock = threading.Lock()
        self._local = threading.local()

    def conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            conn = sqlite3.connect(self._path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    @contextmanager
    def write(self) -> Iterator[sqlite3.Connection]:
        """Executa escritas serializadas e faz commit ao final."""
        with self._write_lock:
            conn = self.conn()
            yield conn
            conn.commit()
