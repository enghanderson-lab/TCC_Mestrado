"""IndexRepository em disco: uma pasta por câmera, FAISS por padrão.

`load()` detecta o formato pelo conteúdo da pasta (faiss.index → FAISS;
senão numpy legado) e mantém o índice em memória, invalidado pelo mtime de
`records.json` (reescrito a cada indexação). Assim a API não relê do disco o
índice de todas as câmeras a cada busca; índices carregados são
somente-leitura.
"""

import shutil
import threading
from pathlib import Path
from typing import Dict, Tuple

from .faiss_index import FAISS_AVAILABLE, FaissIndex
from .numpy_index import NumpyIndex


class FileIndexRepository:
    def __init__(self, use_faiss: bool = True) -> None:
        self._use_faiss = use_faiss and FAISS_AVAILABLE
        self._cache: Dict[str, Tuple[float, object]] = {}
        self._lock = threading.Lock()

    def new(self):
        return FaissIndex() if self._use_faiss else NumpyIndex()

    def load(self, location: str):
        key = str(Path(location).resolve())
        mtime = (Path(key) / "records.json").stat().st_mtime  # FileNotFoundError se removido
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None and cached[0] == mtime:
                return cached[1]
        index = FaissIndex.load(key) if (Path(key) / "faiss.index").exists() else NumpyIndex.load(key)
        with self._lock:
            self._cache[key] = (mtime, index)
        return index

    def delete(self, location: str) -> None:
        key = str(Path(location).resolve())
        with self._lock:
            self._cache.pop(key, None)
        if Path(key).exists():
            shutil.rmtree(key)
