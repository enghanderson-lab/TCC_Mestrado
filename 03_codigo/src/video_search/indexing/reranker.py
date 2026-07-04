"""Reranker leve baseado em Maximal Marginal Relevance (MMR).

MMR balanceia relevância (score SigLIP) e diversidade (distância de cosseno
entre candidatos já selecionados). Isso elimina resultados redundantes
do Top-100 antes de passar para o Qwen — o VLM só processa frames
genuinamente diferentes.

Referência: Carbonell & Goldstein (1998), "The use of MMR, diversity-based
reranking for reordering documents and producing summaries".

Algoritmo:
    selected = []
    remaining = candidatos Top-100

    enquanto |selected| < k:
        para cada r em remaining:
            mmr(r) = λ · sim(query, r) - (1-λ) · max_{s ∈ selected} sim(r, s)
        selecionado = argmax_{r ∈ remaining} mmr(r)
        move r de remaining para selected

Complexidade: O(N·k) em produto interno — trivial para N=100, k≤20.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .embedding_store import FrameRecord


@dataclass
class RerankerConfig:
    lambda_: float = 0.7    # 1.0 = puramente relevância; 0.0 = puramente diversidade
    enabled: bool = True


class MmrReranker:
    """Reranker de diversidade por MMR.

    Recebe candidatos (score, FrameRecord, embedding) vindos do FAISS e
    seleciona os k mais relevantes e diversos sem nenhum modelo adicional.
    """

    def __init__(self, config: RerankerConfig = RerankerConfig()) -> None:
        self._lambda = config.lambda_
        self._enabled = config.enabled

    def rerank(
        self,
        query_embedding: np.ndarray,
        candidates: List[Tuple[float, FrameRecord, np.ndarray]],
        k: int,
    ) -> List[Tuple[float, FrameRecord]]:
        """Seleciona os `k` candidatos mais relevantes e diversos.

        Se desabilitado ou k >= len(candidates), retorna os candidatos
        originais em ordem de score SigLIP.

        Args:
            query_embedding: embedding L2-normalizado da query (768-dim).
            candidates:      lista de (score_siglip, record, embedding).
            k:               número de candidatos a selecionar.

        Returns:
            Lista de (score_siglip_original, record) dos k selecionados,
            em ordem de seleção MMR (não de score).
        """
        if not candidates:
            return []

        k = min(k, len(candidates))

        if not self._enabled or len(candidates) <= k:
            return [(s, r) for s, r, _ in candidates[:k]]

        scores = np.array([c[0] for c in candidates], dtype=np.float32)
        embeddings = np.stack([c[2] for c in candidates]).astype(np.float32)

        selected_idx: List[int] = []
        remaining_idx = list(range(len(candidates)))

        for _ in range(k):
            if not remaining_idx:
                break

            if not selected_idx:
                # Primeiro: candidato com maior score SigLIP
                best = max(remaining_idx, key=lambda i: scores[i])
            else:
                sel_embs = embeddings[selected_idx]  # (|selected|, dim)
                # Similaridade de cosseno candidato × cada já selecionado
                # embeddings[remaining] @ sel_embs.T → (|remaining|, |selected|)
                rem = np.array(remaining_idx)
                redundancy = (embeddings[rem] @ sel_embs.T).max(axis=1)  # (|remaining|,)
                relevance = scores[rem]
                mmr_scores = self._lambda * relevance - (1.0 - self._lambda) * redundancy
                best = remaining_idx[int(mmr_scores.argmax())]

            selected_idx.append(best)
            remaining_idx.remove(best)

        return [(candidates[i][0], candidates[i][1]) for i in selected_idx]
