"""Benchmark de comparacao entre Florence-2-large e Qwen2.5-VL-3B para a
tarefa de legendagem condicionada a query no pipeline de busca em video.

Uso rapido (frame do experimento de validacao, t=95.32s):
    python scripts/benchmark_vlm_florence2.py caminho/video.mp4

Com timestamp e query customizados:
    python scripts/benchmark_vlm_florence2.py caminho/video.mp4 ^
        --timestamp 95.32 ^
        --query "uma mulher com cabelos escuros, tenis brancos e mochila" ^
        --query-en "a woman with dark hair, white sneakers and a backpack"

Dependencia adicional (nao esta em requirements.txt):
    pip install timm  # requerido pela Florence-2 (encoder de visao)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentence_transformers import SentenceTransformer  # noqa: E402

from video_search.frame_extractor import extract_frames_at  # noqa: E402
from video_search.hf_utils import load_offline_first  # noqa: E402
from video_search.vlm_describer import Qwen2VLDescriber  # noqa: E402

_DEFAULT_QUERY_PT = (
    "uma mulher com cabelos escuros, tenis brancos e usando uma mochila nas costas"
)
_DEFAULT_QUERY_EN = (
    "a woman with dark hair, white sneakers and carrying a backpack on her back"
)
_CONFIDENCE_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
_BATCH_SIZE = 12


def _vram_mb() -> float:
    return torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0


def _peak_vram_mb() -> float:
    return torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0


def _hr(title: str = "") -> None:
    print()
    w = 60
    if title:
        pad = (w - len(title) - 2) // 2
        print("=" * pad + f" {title} " + "=" * (w - pad - len(title) - 2))
    else:
        print("=" * w)


# ---------------------------------------------------------------------------
# Florence-2 helpers
# ---------------------------------------------------------------------------

def _load_florence(device: str):
    from transformers import AutoModelForCausalLM, AutoProcessor

    t0 = time.perf_counter()
    model = load_offline_first(
        AutoModelForCausalLM.from_pretrained,
        "microsoft/Florence-2-large",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    ).to(device).eval()
    processor = load_offline_first(
        AutoProcessor.from_pretrained,
        "microsoft/Florence-2-large",
        trust_remote_code=True,
    )
    return model, processor, time.perf_counter() - t0


def _to_device(inputs: dict, device: str) -> dict:
    return {
        k: v.to(device, dtype=torch.float16) if (torch.is_floating_point(v)) else v.to(device)
        for k, v in inputs.items()
    }


def _pad_square(image):
    """Pad com borda preta para tornar a imagem quadrada (exigido pelo DaViT da Florence-2)."""
    from PIL import Image as _Image
    w, h = image.size
    s = max(w, h)
    out = _Image.new("RGB", (s, s), (0, 0, 0))
    out.paste(image, ((s - w) // 2, (s - h) // 2))
    return out


@torch.no_grad()
def _florence_single(model, processor, image, task: str, text: str = "") -> tuple:
    prompt = task + text
    raw_inputs = processor(text=prompt, images=_pad_square(image), return_tensors="pt")
    inputs = _to_device(dict(raw_inputs), str(model.device))
    t0 = time.perf_counter()
    gen_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=60,
        do_sample=False,
        num_beams=3,
    )
    dt = time.perf_counter() - t0
    raw_text = processor.batch_decode(gen_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        raw_text, task=task, image_size=(image.width, image.height)
    )
    caption = parsed.get(task, raw_text)
    return caption, dt


@torch.no_grad()
def _florence_batch(model, processor, images, task: str, text: str = "") -> tuple:
    prompts = [task + text] * len(images)
    raw_inputs = processor(text=prompts, images=[_pad_square(img) for img in images], return_tensors="pt", padding=True)
    inputs = _to_device(dict(raw_inputs), str(model.device))
    t0 = time.perf_counter()
    gen_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=40,
        do_sample=False,
        num_beams=1,
    )
    dt = time.perf_counter() - t0
    captions = []
    for gid in gen_ids:
        raw_text = processor.decode(gid, skip_special_tokens=False)
        parsed = processor.post_process_generation(
            raw_text, task=task, image_size=(images[0].width, images[0].height)
        )
        captions.append(parsed.get(task, raw_text))
    return captions, dt


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _confidence_scores(query: str, captions: list) -> list:
    text_model = load_offline_first(SentenceTransformer, _CONFIDENCE_MODEL, device="cpu")
    embs = text_model.encode([query] + captions, normalize_embeddings=True)
    q, caps = embs[0], embs[1:]
    return [float(np.dot(q, c)) for c in caps]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="Caminho do video (mp4)")
    parser.add_argument("--timestamp", type=float, default=95.32,
                        help="Timestamp do frame de teste (s) [default: 95.32]")
    parser.add_argument("--query", default=_DEFAULT_QUERY_PT,
                        help="Query em portugues (para Qwen e para score)")
    parser.add_argument("--query-en", default=_DEFAULT_QUERY_EN, dest="query_en",
                        help="Traducao da query para ingles (para Florence VQA)")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"Device : {device}")
    print(f"Query PT: {args.query}")
    print(f"Query EN: {args.query_en}")
    print(f"Batch   : {_BATCH_SIZE} imagens")

    print(f"\nExtraindo frame em t={args.timestamp:.2f}s ...")
    (frame,) = extract_frames_at(args.video, [args.timestamp])
    print(f"Frame: {frame.size[0]}x{frame.size[1]} px")
    test_batch = [frame] * _BATCH_SIZE

    # ===================================================================
    _hr("1  FLORENCE-2-large")
    # ===================================================================
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    f_model, f_proc, f_load_t = _load_florence(device)
    f_vram_load = _vram_mb()
    print(f"Carga          : {f_load_t:.2f}s")
    print(f"VRAM apos carga: {f_vram_load:.0f} MB")

    # -- 1a.  Legenda generica (<MORE_DETAILED_CAPTION>) -----------------
    print("\n[1a] <MORE_DETAILED_CAPTION>  (generica, sem query)")
    f_cap_generic, t = _florence_single(f_model, f_proc, frame, "<MORE_DETAILED_CAPTION>")
    print(f"     Tempo : {t:.2f}s")
    print(f"     Saida : {f_cap_generic}")

    # -- 1b.  VQA com query em PT ----------------------------------------
    print("\n[1b] <VQA> + query em PT")
    f_cap_vqa_pt, t = _florence_single(f_model, f_proc, frame, "<VQA>", args.query)
    print(f"     Tempo : {t:.2f}s")
    print(f"     Saida : {f_cap_vqa_pt}")

    # -- 1c.  VQA com query em EN ----------------------------------------
    print("\n[1c] <VQA> + query em EN")
    f_cap_vqa_en, t = _florence_single(f_model, f_proc, frame, "<VQA>", args.query_en)
    print(f"     Tempo : {t:.2f}s")
    print(f"     Saida : {f_cap_vqa_en}")

    # -- 1d.  Batch -------------------------------------------------------
    print(f"\n[1d] Batch {_BATCH_SIZE}x <VQA> + query EN")
    f_batch_caps, f_batch_t = _florence_batch(f_model, f_proc, test_batch, "<VQA>", args.query_en)
    print(f"     Tempo total: {f_batch_t:.2f}s  ({f_batch_t/_BATCH_SIZE:.2f}s/img)")
    print(f"     Saida [0]  : {f_batch_caps[0]}")

    f_vram_peak = _peak_vram_mb()

    del f_model, f_proc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # ===================================================================
    _hr("2  QWEN2.5-VL-3B-Instruct  (NF4 cached)")
    # ===================================================================
    t0 = time.perf_counter()
    qwen = Qwen2VLDescriber()
    q_load_t = time.perf_counter() - t0
    q_vram_load = _vram_mb()
    print(f"Carga          : {q_load_t:.2f}s")
    print(f"VRAM apos carga: {q_vram_load:.0f} MB")

    # -- 2a.  Legenda generica -------------------------------------------
    print("\n[2a] describe()  (generica, sem query)")
    t0 = time.perf_counter()
    q_cap_generic = qwen.describe(frame)
    t = time.perf_counter() - t0
    print(f"     Tempo : {t:.2f}s")
    print(f"     Saida : {q_cap_generic}")

    # -- 2b.  Query-condicionada single ----------------------------------
    print("\n[2b] describe_for_query() + query PT")
    t0 = time.perf_counter()
    q_cap_query = qwen.describe_for_query(frame, args.query)
    t = time.perf_counter() - t0
    print(f"     Tempo : {t:.2f}s")
    print(f"     Saida : {q_cap_query}")

    # -- 2c.  Batch -------------------------------------------------------
    print(f"\n[2c] Batch {_BATCH_SIZE}x describe_for_query_batch() + query PT")
    t0 = time.perf_counter()
    q_batch_caps = qwen.describe_for_query_batch(test_batch, args.query)
    q_batch_t = time.perf_counter() - t0
    print(f"     Tempo total: {q_batch_t:.2f}s  ({q_batch_t/_BATCH_SIZE:.2f}s/img)")
    print(f"     Saida [0]  : {q_batch_caps[0]}")

    q_vram_peak = _peak_vram_mb()

    # ===================================================================
    _hr("3  SCORES DE CONFIANCA  (vs query PT)")
    # ===================================================================
    print("(SentenceTransformer paraphrase-multilingual-mpnet-base-v2)\n")
    labels = [
        "Florence  <MORE_DETAILED_CAPTION>  generica",
        "Florence  <VQA> + query PT",
        "Florence  <VQA> + query EN",
        "Florence  batch<VQA>+EN  [0]",
        "Qwen      describe()     generica",
        "Qwen      describe_for_query()  PT",
        "Qwen      batch descr_for_query [0]",
    ]
    captions = [
        f_cap_generic,
        f_cap_vqa_pt,
        f_cap_vqa_en,
        f_batch_caps[0],
        q_cap_generic,
        q_cap_query,
        q_batch_caps[0],
    ]
    scores = _confidence_scores(args.query, captions)
    for label, score, cap in zip(labels, scores, captions):
        print(f"  {score:.3f}  {label}")
        print(f"         >> {cap}")
        print()

    # ===================================================================
    _hr("4  RESUMO")
    # ===================================================================
    row = "  {:<40} {:>16} {:>14}"
    sep = "  " + "-" * 72
    print(row.format("Metrica", "Florence-2-large", "Qwen2.5-VL-3B"))
    print(sep)
    print(row.format("Carga (s)", f"{f_load_t:.1f}s", f"{q_load_t:.1f}s"))
    print(row.format("VRAM apos carga (MB)", f"{f_vram_load:.0f}", f"{q_vram_load:.0f}"))
    print(row.format("VRAM pico durante generate (MB)", f"{f_vram_peak:.0f}", f"{q_vram_peak:.0f}"))
    print(row.format(f"Batch {_BATCH_SIZE}x (s total)", f"{f_batch_t:.1f}s", f"{q_batch_t:.1f}s"))
    print(row.format(f"Batch {_BATCH_SIZE}x (s/img)", f"{f_batch_t/_BATCH_SIZE:.2f}s", f"{q_batch_t/_BATCH_SIZE:.2f}s"))
    print(row.format("Conf. legenda condicionada (melhor)", f"{scores[2]:.3f} (EN)", f"{scores[5]:.3f} (PT)"))
    print()
    print("Nota: Florence usa <VQA> para condicionar em query (nao instruction-following")
    print("      livre). Qwen usa prompt PT nativo sem etapa de traducao.")


if __name__ == "__main__":
    main()
