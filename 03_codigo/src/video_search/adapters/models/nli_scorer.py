"""ConfidenceScorer por inferência textual (NLI): a legenda implica a query?

Substitui a similaridade de cosseno texto-texto (sentence-transformer), que
ignorava negação: "Não há uma mulher de rosa sobre uma moto" repete as
palavras da query e pontuava 92,8%, mais que o frame com o alvo.

Modelo: MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 (MIT),
mDeBERTa-v3 treinado em XNLI + multilingual-NLI-26lang (inclui português).
Confiança = P(entailment | premissa = legenda, hipótese = "A imagem mostra
<query>."); neutro (legenda não fala do alvo) e contradição ("não há X")
ficam ambos com confiança baixa.
"""

import re
from typing import List, Sequence

import torch

NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

# Prefixos de comando que o operador costuma digitar e que não descrevem a cena.
_COMMAND_PREFIX = re.compile(
    r"^\s*(procure|procurar|busque|buscar|encontre|encontrar|ache|achar|mostre|mostrar)\s+(por\s+)?",
    re.IGNORECASE,
)


def hypothesis_for(query: str) -> str:
    """Transforma a query numa afirmação sobre a imagem:
    "procure por uma van branca" → "A imagem mostra uma van branca."."""
    text = _COMMAND_PREFIX.sub("", query).strip().rstrip(".!?")
    if text:
        text = text[0].lower() + text[1:]
    return f"A imagem mostra {text}."


class NliConfidenceScorer:
    def __init__(self, model, tokenizer, device: str = "cpu", max_length: int = 256) -> None:
        # O checkpoint é float16 e o transformers 5.x o carrega assim; na CPU
        # não há kernels float16 rápidos e 5 legendas levavam ~22 s (float32:
        # ~0,25 s, mesma probabilidade). Na GPU, float16 é o mais rápido.
        dtype = torch.float32 if str(device).startswith("cpu") else model.dtype
        self._model = model.to(device=device, dtype=dtype).eval()
        self._tokenizer = tokenizer
        self._device = device
        self._max_length = max_length
        labels = {v.lower(): int(k) for k, v in model.config.id2label.items()}
        self._entailment = labels["entailment"]

    @torch.no_grad()
    def score(self, query: str, captions: Sequence[str]) -> List[float]:
        if not captions:
            return []
        hypothesis = hypothesis_for(query)
        inputs = self._tokenizer(
            list(captions), [hypothesis] * len(captions),
            return_tensors="pt", padding=True, truncation=True, max_length=self._max_length,
        ).to(self._device)
        probs = self._model(**inputs).logits.float().softmax(dim=-1)
        return [float(p) for p in probs[:, self._entailment]]
