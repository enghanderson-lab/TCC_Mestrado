"""Microsservico HTTP (FastAPI) para o pipeline de busca semantica em video.

Expoe `index`/`search` (ver `pipeline.py`) como endpoints REST, para
integracao com o VMS (Qt). Videos de entrada sao referenciados por caminho em
disco acessivel ao processo do servico — nao ha upload via multipart (videos
de horas de duracao nao cabem num upload HTTP razoavel).

Indices sao identificados por `name` e resolvidos dentro de `INDEX_ROOT`
(padrao: `04_dados/index`, configuravel via `VIDEO_SEARCH_INDEX_ROOT`) — isso
evita expor caminhos arbitrarios do filesystem por essa rota.

Rodar com:
    uvicorn video_search.api:app --host 0.0.0.0 --port 8000
"""

import json
import os
import re
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

from .pipeline import run_index, run_search

INDEX_ROOT = Path(
    os.environ.get(
        "VIDEO_SEARCH_INDEX_ROOT",
        str(Path(__file__).resolve().parents[3] / "04_dados" / "index"),
    )
)

_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")

app = FastAPI(title="video_search", description=__doc__)

# Jobs de indexacao em memoria (sem persistencia entre restarts do processo —
# suficiente para v1; ver plano/README para evolucao futura se necessario).
_jobs: Dict[str, dict] = {}


def _validate_name(name: str) -> None:
    if not _NAME_RE.match(name):
        raise HTTPException(
            status_code=400,
            detail="Nome invalido (use apenas letras, numeros, '_' ou '-')",
        )


class IndexRequest(BaseModel):
    video_path: str = Field(..., description="Caminho do video, acessivel ao processo do servico")
    name: str = Field(..., description="Nome do indice (define a subpasta em INDEX_ROOT)")
    interval: float = Field(1.0, gt=0, description="Intervalo entre frames amostrados (s)")
    batch_size: int = Field(16, gt=0, description="Tamanho do lote para inferencia")
    limit: int = Field(0, ge=0, description="Limita o numero de frames processados (0 = sem limite)")


class IndexJobStatus(BaseModel):
    job_id: str
    status: str
    frame_count: int
    output_dir: str
    error: Optional[str] = None


class SearchRequest(BaseModel):
    query: str = Field(..., description="Descricao em linguagem natural (PT)")
    index: str = Field(..., description="Nome do indice (gerado por /index)")
    top_k: int = Field(12, gt=0)
    with_caption: bool = Field(True, description="Gera legenda/confianca via Qwen2.5-VL para o reranking")


class SearchResultModel(BaseModel):
    frame_index: int
    timestamp_sec: float
    video: str
    retrieval_score: float
    confidence: Optional[float] = None
    caption: Optional[str] = None


class IndexInfo(BaseModel):
    name: str
    frame_count: int
    model_name: str


def _job_response(job_id: str) -> IndexJobStatus:
    job = _jobs[job_id]
    return IndexJobStatus(job_id=job_id, **job)


def _run_index_job(job_id: str, video_path: str, output_dir: str, interval: float, batch_size: int, limit: int) -> None:
    _jobs[job_id]["status"] = "running"
    try:
        def on_progress(frame_count: int) -> None:
            _jobs[job_id]["frame_count"] = frame_count

        summary = run_index(
            video_path, output_dir, interval=interval, batch_size=batch_size, limit=limit, on_progress=on_progress
        )
        _jobs[job_id]["frame_count"] = summary.frame_count
        _jobs[job_id]["status"] = "done"
    except Exception as exc:  # job em background: captura para reportar via /index/jobs/{id}
        _jobs[job_id]["status"] = "error"
        _jobs[job_id]["error"] = str(exc)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/indexes", response_model=List[IndexInfo])
def list_indexes() -> List[IndexInfo]:
    if not INDEX_ROOT.exists():
        return []
    infos = []
    for entry in sorted(INDEX_ROOT.iterdir()):
        records_path = entry / "records.json"
        if not entry.is_dir() or not records_path.exists():
            continue
        try:
            with open(records_path, encoding="utf-8") as f:
                frame_count = len(json.load(f))
            model_name = "siglip"
            meta_path = entry / "metadata.json"
            if meta_path.exists():
                with open(meta_path, encoding="utf-8") as f:
                    model_name = json.load(f).get("model_name", "siglip")
            infos.append(IndexInfo(name=entry.name, frame_count=frame_count, model_name=model_name))
        except (json.JSONDecodeError, OSError):
            continue
    return infos


@app.post("/index", response_model=IndexJobStatus)
def create_index_job(req: IndexRequest, background_tasks: BackgroundTasks) -> IndexJobStatus:
    _validate_name(req.name)
    if not Path(req.video_path).is_file():
        raise HTTPException(status_code=404, detail=f"Video nao encontrado: {req.video_path}")

    output_dir = INDEX_ROOT / req.name
    job_id = uuid.uuid4().hex
    _jobs[job_id] = {"status": "queued", "frame_count": 0, "output_dir": str(output_dir), "error": None}

    background_tasks.add_task(
        _run_index_job, job_id, req.video_path, str(output_dir), req.interval, req.batch_size, req.limit
    )
    return _job_response(job_id)


@app.get("/index/jobs/{job_id}", response_model=IndexJobStatus)
def get_index_job(job_id: str) -> IndexJobStatus:
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job nao encontrado")
    return _job_response(job_id)


@app.post("/search", response_model=List[SearchResultModel])
def search(req: SearchRequest) -> List[SearchResultModel]:
    _validate_name(req.index)
    index_dir = INDEX_ROOT / req.index
    if not (index_dir / "records.json").exists():
        raise HTTPException(status_code=404, detail=f"Indice nao encontrado: {req.index}")

    results = run_search(req.query, str(index_dir), top_k=req.top_k, with_caption=req.with_caption)
    return [SearchResultModel(**vars(r)) for r in results]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
