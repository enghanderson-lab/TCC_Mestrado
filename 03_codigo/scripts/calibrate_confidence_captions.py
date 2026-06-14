"""Calibracao do score de confianca para o caso real: query curta vs legenda
longa gerada pelo Qwen2-VL (estilo multi-frase, cheia de detalhes do cenario).

A calibracao anterior (calibrate_confidence.py) usou pares parafrase-vs-
parafrase de tamanho/especificidade similares e obteve excelente separacao
(match~0.85, mismatch~0.10). Mas na busca real comparamos uma query curta
("carro estacionado na rua") com uma legenda longa cheia de detalhes
irrelevantes, o que comprime a similaridade para a faixa ~0.45-0.66 mesmo
quando o conceito esta presente. Este script calibra FLOOR/CEIL nesse regime
assimetrico, para reescalonar confianca_legenda de modo que matches reais
fiquem >80%.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

# Pares (query curta, legenda longa) onde o conceito da query ESTA presente
# na legenda - estilo real de saida do Qwen2-VL.
CAPTION_MATCH_PAIRS = [
    (
        "carro estacionado na rua",
        "A imagem mostra uma rua tranquila com um carro cinza parado na esquina, "
        "ao lado de um edificio de apartamentos brancos. Ao fundo, ha uma casa de "
        "madeira com uma varanda branca e arvores altas. No ceu, temos nuvens.",
    ),
    (
        "ponte sobre o rio",
        "Na imagem, vemos uma ponte de metal atravessando um rio largo. As margens "
        "sao cobertas de vegetacao densa e algumas casas aparecem ao fundo, sob um "
        "ceu parcialmente nublado.",
    ),
    (
        "estrada com linha amarela no meio",
        "A foto mostra uma estrada asfaltada com uma linha amarela central, "
        "cercada por campos verdes. Ha postes de eletricidade ao longo da via e "
        "montanhas distantes no horizonte.",
    ),
    (
        "montanhas cobertas de arvores",
        "A paisagem exibe uma cadeia de montanhas densamente cobertas por arvores "
        "verdes. No primeiro plano, ha uma estrada de terra e um pequeno riacho "
        "que corta o vale.",
    ),
    (
        "mulher de jaqueta preta andando na calcada",
        "A imagem mostra uma mulher vestindo uma jaqueta preta caminhando por uma "
        "calcada urbana. Ao redor, ha predios baixos, algumas arvores e um ceu "
        "cinzento de fundo.",
    ),
    (
        "homem de camisa branca e bone vermelho",
        "Na cena, um homem usando camisa branca e bone vermelho aparece proximo a "
        "uma cerca de madeira. Atras dele, ha um campo aberto e algumas nuvens no "
        "ceu azul.",
    ),
]

# Pares (query curta, legenda longa) onde o conceito da query NAO aparece -
# legenda descreve outra cena, estilo real de saida do Qwen2-VL.
CAPTION_MISMATCH_PAIRS = [
    (
        "carro estacionado na rua",
        "A imagem mostra uma cena urbana com uma cerca de madeira branca ao lado "
        "de uma rua. Na frente da cerca, ha uma placa com a inscricao 'Street'. No "
        "fundo, ha um ceu nublado com algumas arvores e uma casa de madeira.",
    ),
    (
        "ponte sobre o rio",
        "A foto mostra uma cozinha com uma geladeira branca e armarios de madeira. "
        "Sobre a bancada, ha alguns utensilios e uma janela deixa entrar luz "
        "natural.",
    ),
    (
        "estrada com linha amarela no meio",
        "Na imagem, uma pessoa toca violao sentada em um sofa de uma sala "
        "decorada com quadros na parede e uma estante de livros ao fundo.",
    ),
    (
        "montanhas cobertas de arvores",
        "A cena mostra um computador sobre uma mesa de escritorio, com uma "
        "cadeira preta e uma janela com cortinas claras ao fundo.",
    ),
    (
        "mulher de jaqueta preta andando na calcada",
        "A imagem mostra um prato de macarrao com queijo sobre uma mesa de "
        "madeira, acompanhado de um guardanapo e um copo de suco ao lado.",
    ),
    (
        "homem de camisa branca e bone vermelho",
        "Na foto, um gato cinza dorme enrolado sobre um sofa marrom, em uma sala "
        "iluminada por uma janela com cortinas brancas.",
    ),
]


def avg_sim(model, pairs):
    sims = []
    for query, caption in pairs:
        emb = model.encode([query, caption], normalize_embeddings=True)
        sims.append(float(np.dot(emb[0], emb[1])))
    return sims


def main():
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

    match_sims = avg_sim(model, CAPTION_MATCH_PAIRS)
    mismatch_sims = avg_sim(model, CAPTION_MISMATCH_PAIRS)

    print("Pares 'match' (query curta <-> legenda longa, conceito presente):")
    for (q, c), s in zip(CAPTION_MATCH_PAIRS, match_sims):
        print(f"  {s:.4f}  '{q}' <-> '{c[:60]}...'")
    print(f"  media={sum(match_sims)/len(match_sims):.4f}  min={min(match_sims):.4f}")

    print("\nPares 'mismatch' (query curta <-> legenda longa, conceito ausente):")
    for (q, c), s in zip(CAPTION_MISMATCH_PAIRS, mismatch_sims):
        print(f"  {s:.4f}  '{q}' <-> '{c[:60]}...'")
    print(f"  media={sum(mismatch_sims)/len(mismatch_sims):.4f}  max={max(mismatch_sims):.4f}")

    floor = max(mismatch_sims)
    ceil_ = sum(match_sims) / len(match_sims)
    print(f"\nFLOOR sugerido (max mismatch) = {floor:.4f}")
    print(f"CEIL sugerido (media match)   = {ceil_:.4f}")
    print(f"separacao (ceil-floor)        = {ceil_-floor:.4f}")


if __name__ == "__main__":
    main()
