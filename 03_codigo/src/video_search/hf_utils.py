"""Utilitario compartilhado para acelerar carregamento de modelos
Hugging Face que ja estao em cache local (SigLIP, Qwen2.5-VL,
sentence-transformer)."""

from typing import Callable, TypeVar

T = TypeVar("T")


def load_offline_first(loader: Callable[..., T], *args, **kwargs) -> T:
    """Tenta carregar com `local_files_only=True` primeiro -- isso pula o
    round-trip de rede que o `from_pretrained` faz por padrao para checar
    se ha uma revisao mais nova do repo no Hub, mesmo quando os arquivos ja
    estao em cache (custa alguns segundos por modelo, multiplicado por
    SigLIP + Qwen2.5-VL + sentence-transformer numa busca). Se os arquivos
    ainda nao estiverem em cache (1a execucao), cai para o modo online
    normal, que baixa o necessario."""
    try:
        return loader(*args, local_files_only=True, **kwargs)
    except OSError:
        return loader(*args, **kwargs)
