"""Cache persistente de embeddings por hash de imagem.

Frames já vistos nunca geram nova inferência SigLIP. O cache sobrevive entre
execuções (SQLite) e é thread-safe (lock explícito + conexões por thread).

Schema SQLite:
    image_hash TEXT PRIMARY KEY
    embedding  BLOB   — np.ndarray float32 serializado em bytes
    dim        INT    — dimensão do vetor (768 para SigLIP)
    created_at TEXT

Uso típico no SigLIP worker:
    cache = EmbeddingCache(path)
    miss_images, miss_idx = [], []
    for i, (item, h) in enumerate(zip(batch, hashes)):
        emb = cache.get(h)
        if emb is not None:
            all_embs[i] = emb
        else:
            miss_idx.append(i)
            miss_images.append(item.image)
    if miss_images:
        new_embs = embedder.encode_images(miss_images)
        for i, emb in zip(miss_idx, new_embs):
            all_embs[i] = emb
            cache.put(hashes[i], emb)
"""

import hashlib
import sqlite3
import threading
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
from PIL import Image

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS embeddings (
    image_hash TEXT PRIMARY KEY,
    embedding  BLOB NOT NULL,
    dim        INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
)
"""
_GET_SQL = "SELECT embedding, dim FROM embeddings WHERE image_hash = ?"
_PUT_SQL = "INSERT OR IGNORE INTO embeddings (image_hash, embedding, dim) VALUES (?, ?, ?)"


class EmbeddingCache:
    """Cache SQLite de embeddings por hash SHA-256 da imagem.

    Conexões por thread: SQLite não é thread-safe no modo padrão, mas
    `check_same_thread=False` + lock explícito nas escritas é seguro.
    """

    def __init__(self, db_path: Union[str, Path]) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()
        self._local = threading.local()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._local.conn.execute("PRAGMA journal_mode=WAL")
        return self._local.conn

    def _init_db(self) -> None:
        with self._write_lock:
            conn = self._conn()
            conn.execute(_CREATE_SQL)
            conn.commit()

    # ------------------------------------------------------------------
    # Interface pública
    # ------------------------------------------------------------------

    @staticmethod
    def hash_image(image: Image.Image) -> str:
        """SHA-256 do buffer bruto da imagem (modo + tamanho + pixels)."""
        h = hashlib.sha256()
        h.update(image.mode.encode())
        h.update(b"%dx%d" % image.size)
        h.update(image.tobytes())
        return h.hexdigest()

    def get(self, image_hash: str) -> Optional[np.ndarray]:
        """Retorna o embedding cached ou None se não encontrado."""
        row = self._conn().execute(_GET_SQL, (image_hash,)).fetchone()
        if row is None:
            return None
        blob, dim = row
        return np.frombuffer(blob, dtype=np.float32).copy().reshape(dim)

    def put(self, image_hash: str, embedding: np.ndarray) -> None:
        """Armazena embedding; ignora silenciosamente se já existir (INSERT OR IGNORE)."""
        blob = embedding.astype(np.float32).tobytes()
        dim = int(embedding.shape[-1])
        with self._write_lock:
            conn = self._conn()
            conn.execute(_PUT_SQL, (image_hash, blob, dim))
            conn.commit()

    def get_batch(self, hashes: list) -> Dict[str, Optional[np.ndarray]]:
        """Busca múltiplos hashes em uma única query."""
        if not hashes:
            return {}
        placeholders = ",".join("?" * len(hashes))
        rows = self._conn().execute(
            f"SELECT image_hash, embedding, dim FROM embeddings WHERE image_hash IN ({placeholders})",
            hashes,
        ).fetchall()
        result: Dict[str, Optional[np.ndarray]] = {h: None for h in hashes}
        for image_hash, blob, dim in rows:
            result[image_hash] = np.frombuffer(blob, dtype=np.float32).copy().reshape(dim)
        return result

    def close(self) -> None:
        if hasattr(self._local, "conn"):
            self._local.conn.close()
            del self._local.conn
