"""CLI: indexar vídeos e buscar frames por descrição em linguagem natural.

A indexação gera apenas embeddings SigLIP (rápido, todos os frames). A
legenda (Qwen2.5-VL) e a confiança textual são calculadas sob demanda,
apenas para os top-K resultados de uma busca.

Adapter de entrada: traduz argumentos para os serviços montados em
`video_search.bootstrap.Container`. Os modelos são carregados uma única vez
por processo.

Exemplos:
    python -m video_search index video.mp4 --output index/video --interval 2
    python -m video_search search "homem de camisa branca" --index index/video
    python -m video_search search "van escolar" --index index/cam1 index/cam2

Profiling:
    python -m video_search index video.mp4 --profiling-dir profiling/
    python -m video_search search "..." --profiling-dir profiling/

Avaliação da confiança (ver application/evaluation.py):
    python -m video_search evaluate ..\\04_dados\\anotacoes\\consultas.jsonl --index-root index
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path

from ..adapters.models.model_config import ModelConfig
from ..application.evaluation import EvaluationConfig, Estimate, evaluate, load_annotations
from ..application.search_service import SearchMode
from ..bootstrap import Container
from ..config import MultiIndexConfig
from ..domain.entities import Camera, VideoSource
from ..profiling import profiling_session


def cmd_index(args: argparse.Namespace, container: Container) -> None:
    def on_progress(_camera_id: str, frame_count: int) -> None:
        print(f"Processados {frame_count} frames...")

    source = VideoSource(camera_id=Path(args.video).stem, video_path=args.video, output_dir=args.output)
    with profiling_session("indexing", args.profiling_dir) as profiler:
        result = container.indexing.index_video(
            source,
            interval=args.interval,
            batch_size=args.batch_size,
            limit=args.limit,
            on_progress=on_progress,
            profiler=profiler,
        )
    print(
        f"Indice salvo em '{result.output_dir}' ({result.frames_accepted} frames, "
        f"intervalo={args.interval}s)"
    )


def cmd_index_multi(args: argparse.Namespace, container: Container) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(threadName)s %(name)s %(levelname)s %(message)s",
    )
    sources = [
        VideoSource(
            camera_id=Path(video).stem,
            video_path=video,
            output_dir=str(Path(args.output) / Path(video).stem),
        )
        for video in args.videos
    ]
    with profiling_session("indexing", args.profiling_dir) as profiler:
        results = container.indexing.index_cameras(
            sources,
            config=MultiIndexConfig.load(args.config),
            interval=args.interval,
            profiler=profiler,
        )
    for r in results:
        print(
            f"[{r.camera_id}] {r.frames_accepted}/{r.frames_read} frames aceitos -> '{r.output_dir}' "
            f"(motion={r.frames_dropped_motion}, similarity={r.frames_dropped_similarity})"
        )


def cmd_search(args: argparse.Namespace, container: Container) -> None:
    cameras = []
    for index_dir in args.index:
        meta_path = Path(index_dir) / "metadata.json"
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                print(f"[Indice {index_dir}: modelo={json.load(f).get('model_name', 'siglip')}]")
        cameras.append(Camera(camera_id=Path(index_dir).name, video_path="", output_dir=index_dir))

    mode = SearchMode(args.mode)
    print(
        f"[Modo: {mode.value}  top_k={args.top_k}  retrieval_k={args.retrieval_k}  "
        f"reranker_lambda={args.reranker_lambda}]"
    )

    with profiling_session("search", args.profiling_dir) as profiler:
        results = container.search.search(
            args.query,
            cameras,
            mode=mode,
            top_k=args.top_k,
            retrieval_k=args.retrieval_k,
            mmr_lambda=args.reranker_lambda,
            profiler=profiler,
        )
        if profiler:
            profiler.count("frames_indexed", len(results))

    if not results:
        print("Indice vazio.")
        return

    print(f"Resultados para: '{args.query}'")
    for r in results:
        minutes, seconds = divmod(r.timestamp_sec, 60)
        line = (
            f"  t={int(minutes):02d}:{seconds:05.2f}  frame={r.frame_index:6d}  "
            f"retrieval_score={r.score:.4f}"
        )
        if r.confidence is not None:
            line += f"  confianca={r.confidence * 100:.1f}%  legenda='{r.caption}'"
        line += f"  video={r.video}"
        if len(cameras) > 1:
            line += f"  camera={r.camera_id}"
        print(line)


def _fmt(est: Estimate, pct: bool = True) -> str:
    if est.value is None:
        return "indefinida"
    f = (lambda x: f"{x * 100:.1f}%") if pct else (lambda x: f"{x:.3f}")
    ci = f"  (IC 95%: {f(est.ci95[0])} a {f(est.ci95[1])})" if est.ci95 else ""
    return f(est.value) + ci


def cmd_evaluate(args: argparse.Namespace, container: Container) -> None:
    from ..adapters.evaluation_report import write_report

    annotations = load_annotations(args.annotations)
    n_pos = sum(a.positive for a in annotations)
    print(f"[{len(annotations)} consultas: {n_pos} positivas, {len(annotations) - n_pos} negativas]")
    config = EvaluationConfig(
        top_k=args.top_k, tolerance_sec=args.tolerance, target_tpr=args.target,
        tau=args.tau, folds=args.folds, n_boot=args.bootstrap, seed=args.seed,
        retrieval_k=args.retrieval_k, mmr_lambda=args.reranker_lambda,
    )
    report = evaluate(
        container.search, annotations, Path(args.index_root), config,
        on_query=lambda i, n, a: print(f"  [{i}/{n}] {a.query_id}: {a.query}"),
    )
    out = write_report(report, args.output)

    mt = report.metrics
    tau = f"{mt.tau:.4f}" if math.isfinite(mt.tau) else "indefinido"
    print()
    print(f"1. Detecção confirmada (top-{args.top_k}, confiança ≥ τ): {_fmt(mt.detection)}")
    print(f"2. Taxa de falso alarme (consultas negativas):        {_fmt(mt.false_alarm)}")
    print(f"3. AUROC da confiança (por resultado):               {_fmt(mt.auroc, pct=False)}")
    print(f"   τ = {tau}  [{mt.tau_source}; alvo de detecção {args.target * 100:.0f}%]")
    if not mt.target_reached:
        print(f"   ATENÇÃO: o alvo de {args.target * 100:.0f}% de detecção não é atingível nem com τ mínimo —")
        print("   há consultas positivas sem nenhum resultado correto no top-K.")
    if mt.recall_at_k is not None:
        print(f"   Diagnóstico: alvo no top-{args.top_k} (qualquer confiança) = {mt.recall_at_k * 100:.1f}%; "
              f"alvo entre os candidatos do SigLIP = {mt.retrieval_recall * 100:.1f}%")
    if mt.n_results_without_confidence:
        print(f"   {mt.n_results_without_confidence} resultado(s) sem legenda (vídeo ilegível) ficaram fora da AUROC.")
    print(f"Relatório salvo em '{out}' (metricas.json, consultas.csv, roc.png)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Busca semantica em video via SigLIP + Qwen2.5-VL")
    parser.add_argument(
        "--device",
        default="auto",
        help="Device dos modelos: 'auto' (default), 'cuda:0', 'cpu'",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Pula o warmup dos modelos apos o carregamento",
    )
    parser.add_argument(
        "--profiling-dir",
        default=None,
        metavar="DIR",
        help=(
            "Pasta onde salvar o relatório de profiling (.txt) e o gráfico (.png) "
            "ao final da execução. Omitir desativa a geração de arquivos."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Extrai frames e gera embeddings")
    p_index.add_argument("video", help="Caminho do arquivo de video")
    p_index.add_argument("--output", default="index", help="Pasta onde salvar o indice")
    p_index.add_argument("--interval", type=float, default=2.0, help="Intervalo entre frames (s)")
    p_index.add_argument("--batch-size", type=int, default=16, help="Tamanho do lote para inferencia")
    p_index.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limita o numero de frames processados (0 = sem limite; util para testes)",
    )
    p_index.set_defaults(func=cmd_index)

    p_multi = sub.add_parser(
        "index-multi",
        help="Indexa multiplos videos/cameras concorrentemente (motion detection + batch cross-camera + filtro de similaridade)",
    )
    p_multi.add_argument("videos", nargs="+", help="Caminhos dos videos (um por camera)")
    p_multi.add_argument("--output", default="index", help="Pasta base; cada camera grava em <output>/<camera_id>")
    p_multi.add_argument(
        "--config",
        default=None,
        help="Caminho do YAML de configuracao (default: parametros padrao, ver configs/multi_index.yaml)",
    )
    p_multi.add_argument("--interval", type=float, default=2.0, help="Intervalo entre frames (s)")
    p_multi.set_defaults(func=cmd_index_multi)

    p_search = sub.add_parser("search", help="Busca frames por descricao em texto")
    p_search.add_argument("query", help="Descricao em linguagem natural")
    p_search.add_argument(
        "--index",
        nargs="+",
        default=["index"],
        help="Pasta(s) de indice geradas por 'index'/'index-multi'; varias = busca com ranking global",
    )
    p_search.add_argument("--top-k", type=int, default=5, help="Numero de resultados retornados")
    p_search.add_argument(
        "--mode",
        choices=[m.value for m in SearchMode],
        default=SearchMode.DETAILED.value,
        help=(
            "fast: SigLIP apenas (latência mínima). "
            "detailed: SigLIP → MMR → Qwen (máxima qualidade). "
            "Default: detailed"
        ),
    )
    p_search.add_argument(
        "--retrieval-k",
        type=int,
        default=100,
        help="Candidatos FAISS antes do reranker MMR (modo detailed). Default: 100",
    )
    p_search.add_argument(
        "--reranker-lambda",
        type=float,
        default=0.7,
        help=(
            "Peso MMR entre relevância e diversidade (modo detailed). "
            "1.0=relevância pura, 0.0=diversidade pura. Default: 0.7"
        ),
    )
    p_search.set_defaults(func=cmd_search)

    p_eval = sub.add_parser(
        "evaluate",
        help="Avalia a confiança contra consultas anotadas: detecção confirmada, falso alarme e AUROC",
    )
    p_eval.add_argument("annotations", help="Arquivo JSONL de consultas anotadas (ver application/evaluation.py)")
    p_eval.add_argument("--index-root", default="index", help="Pasta com um índice por câmera (<index-root>/<camera>)")
    p_eval.add_argument("--output", default="avaliacao", help="Pasta do relatório (metricas.json, consultas.csv, roc.png)")
    p_eval.add_argument("--top-k", type=int, default=5, help="Resultados considerados por consulta. Default: 5")
    p_eval.add_argument("--tolerance", type=float, default=2.0, help="Tolerância temporal em segundos ao redor do trecho anotado. Default: 2")
    p_eval.add_argument("--target", type=float, default=0.9, help="Detecção mínima exigida para escolher τ (0-1). Default: 0.9")
    p_eval.add_argument("--tau", type=float, default=None, help="Fixa o limiar de confiança em vez de calibrá-lo por validação cruzada")
    p_eval.add_argument("--folds", type=int, default=5, help="Folds da validação cruzada que escolhe τ. Default: 5")
    p_eval.add_argument("--bootstrap", type=int, default=2000, help="Reamostras para os IC 95%%. Default: 2000")
    p_eval.add_argument("--seed", type=int, default=0, help="Semente (resultados reprodutíveis). Default: 0")
    p_eval.add_argument("--retrieval-k", type=int, default=None, help="Candidatos FAISS antes do MMR. Default: max(4·top-k, 20)")
    p_eval.add_argument("--reranker-lambda", type=float, default=0.7, help="λ do MMR. Default: 0.7")
    p_eval.set_defaults(func=cmd_evaluate)

    return parser


def main() -> None:
    # No Windows, stdout nao-interativo (pipe/redirecionamento) cai para a
    # codepage do sistema em vez de UTF-8, e acentos (ç, õ, ã) saem como "?".
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = build_parser()
    args = parser.parse_args()

    model_config = ModelConfig(device=args.device, enable_warmup=not args.no_warmup)
    with Container(model_config=model_config) as container:
        args.func(args, container)


if __name__ == "__main__":
    main()
