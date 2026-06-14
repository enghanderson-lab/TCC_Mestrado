"""Calibracao do score de confianca: estima o piso (pares nao relacionados)
e o teto (parafrases) da similaridade de texto-texto do CLIP, para
reescalonar confianca_legenda para uma faixa 0-100% mais interpretavel."""

import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_search.embedder import ClipEmbedder  # noqa: E402

# Pares "match": mesma cena/conceito descrito com palavras diferentes
PARAPHRASE_PAIRS = [
    ("homem de camisa branca e bone vermelho", "um homem vestindo camisa branca e usando um bone vermelho"),
    ("carro estacionado na rua", "um veiculo parado ao lado da via"),
    ("ponte sobre o rio", "uma passarela atravessando o rio"),
    ("estrada com linha amarela no meio", "via asfaltada com uma faixa amarela central"),
    ("montanhas cobertas de arvores", "colinas verdes cheias de vegetacao"),
    ("mulher de jaqueta preta andando na calcada", "uma pessoa com casaco escuro caminhando na rua"),
]

# Pares "mismatch": conceitos nao relacionados
RANDOM_PAIRS = [
    ("homem de camisa branca e bone vermelho", "uma geladeira branca na cozinha"),
    ("carro estacionado na rua", "um gato dormindo no sofa"),
    ("ponte sobre o rio", "comida sendo preparada em uma panela"),
    ("estrada com linha amarela no meio", "uma pessoa tocando violao"),
    ("montanhas cobertas de arvores", "um computador sobre a mesa"),
    ("mulher de jaqueta preta andando na calcada", "um prato de macarrao com queijo"),
]


def avg_sim_clip(embedder, pairs):
    sims = []
    for a, b in pairs:
        ea = embedder.encode_text([a])[0]
        eb = embedder.encode_text([b])[0]
        sims.append(float(ea @ eb))
    return sims


def avg_sim_st(model, pairs):
    sims = []
    for a, b in pairs:
        emb = model.encode([a, b], normalize_embeddings=True)
        sims.append(float(np.dot(emb[0], emb[1])))
    return sims


def report(name, match_sims, mismatch_sims, pairs_match, pairs_mismatch):
    print(f"\n=== {name} ===")
    print("Pares 'match' (parafrase):")
    for (a, b), s in zip(pairs_match, match_sims):
        print(f"  {s:.4f}  '{a}' <-> '{b}'")
    print(f"  media={sum(match_sims)/len(match_sims):.4f}  min={min(match_sims):.4f}")

    print("Pares 'mismatch' (nao relacionados):")
    for (a, b), s in zip(pairs_mismatch, mismatch_sims):
        print(f"  {s:.4f}  '{a}' <-> '{b}'")
    print(f"  media={sum(mismatch_sims)/len(mismatch_sims):.4f}  max={max(mismatch_sims):.4f}")

    floor = max(mismatch_sims)
    ceil_ = sum(match_sims) / len(match_sims)
    print(f"  FLOOR sugerido (max mismatch) = {floor:.4f}")
    print(f"  CEIL sugerido (media match)   = {ceil_:.4f}")
    print(f"  separacao (ceil-floor)        = {ceil_-floor:.4f}")


def main():
    embedder = ClipEmbedder()
    clip_match = avg_sim_clip(embedder, PARAPHRASE_PAIRS)
    clip_mismatch = avg_sim_clip(embedder, RANDOM_PAIRS)
    report("CLIP text-text", clip_match, clip_mismatch, PARAPHRASE_PAIRS, RANDOM_PAIRS)

    st_model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        device=embedder.device,
    )
    st_match = avg_sim_st(st_model, PARAPHRASE_PAIRS)
    st_mismatch = avg_sim_st(st_model, RANDOM_PAIRS)
    report("paraphrase-multilingual-mpnet-base-v2", st_match, st_mismatch, PARAPHRASE_PAIRS, RANDOM_PAIRS)


if __name__ == "__main__":
    main()
