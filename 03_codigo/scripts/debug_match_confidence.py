"""Depuracao do match_confidence: inspeciona tokens Sim/Nao, logits brutos
e os tokens mais provaveis gerados pelo Qwen2-VL para cada candidato."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_search.models.vlm_describer import Qwen2VLDescriber  # noqa: E402
from smoke_test_qwen2vl import make_scene_image  # noqa: E402


def main():
    describer = Qwen2VLDescriber()

    yes_words = ["Sim", " Sim", "sim", " sim", "Yes", " Yes", "yes", " yes"]
    no_words = ["Nao", " Nao", "Não", " Não", "nao", " nao", "não", " não", "No", " No", "no", " no"]

    print("Tokens 'sim' candidatos:")
    for w in yes_words:
        ids = describer.processor.tokenizer.encode(w, add_special_tokens=False)
        print(f"  {w!r:8s} -> {ids}")

    print("Tokens 'nao' candidatos:")
    for w in no_words:
        ids = describer.processor.tokenizer.encode(w, add_special_tokens=False)
        print(f"  {w!r:8s} -> {ids}")

    yes_ids = describer._token_ids(yes_words)
    no_ids = describer._token_ids(no_words)
    print(f"\nyes_ids (len==1): {yes_ids}")
    print(f"no_ids  (len==1): {no_ids}")

    image = make_scene_image()
    candidates = [
        "homem de camisa branca e bone vermelho",
        "homem de camisa preta e bone azul",
        "uma pessoa usando oculos de sol",
    ]

    for text in candidates:
        question = (
            f'A seguinte descricao se aplica a esta imagem? "{text}". '
            "Responda apenas com Sim ou Nao."
        )
        inputs = describer._prepare_inputs(image, question)
        with torch.no_grad():
            outputs = describer.model.generate(
                **inputs, max_new_tokens=1, output_scores=True, return_dict_in_generate=True
            )
        log_probs = torch.log_softmax(outputs.scores[0][0].float(), dim=-1)
        probs_full = log_probs.exp()

        top_probs, top_ids = torch.topk(probs_full, 5)
        print(f"\n=== '{text}' ===")
        print("Top-5 tokens gerados:")
        for p, tid in zip(top_probs.tolist(), top_ids.tolist()):
            tok = describer.processor.tokenizer.decode([tid])
            print(f"  {p:12.8%}  id={tid:6d}  {tok!r}")

        yes_logprob = torch.logsumexp(log_probs[yes_ids], dim=0)
        no_logprob = torch.logsumexp(log_probs[no_ids], dim=0)
        print(f"P(Sim) absoluta = {yes_logprob.exp().item():.8f}")
        print(f"P(Nao) absoluta = {no_logprob.exp().item():.8f}")
        probs = torch.softmax(torch.stack([yes_logprob, no_logprob]), dim=0)
        print(f"P(Sim | Sim ou Nao) = {probs[0].item():.8f}")


if __name__ == "__main__":
    main()
