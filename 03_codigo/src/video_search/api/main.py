"""FastAPI application factory para o Video Search Engine.

Inicialização:
    uvicorn video_search.api.main:app --host 0.0.0.0 --port 8000

Variáveis de ambiente:
    VS_API_KEY   Chave de autenticação (default: "dev-key")
    VS_DB_PATH   Caminho do SQLite de câmeras (default: cameras.db)
    VS_CORS_ORIGINS  Origens CORS separadas por vírgula (default: nenhuma —
                     fail-safe; configure explicitamente com a origem do
                     frontend antes de qualquer deploy real. Ver API.md.)
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import cameras, frames, health, index, search

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Carregando modelos...")
    from video_search.models.model_manager import get_model_manager
    mm = get_model_manager()
    mm.load_models()
    logger.info("Modelos prontos.")
    yield
    logger.info("Encerrando API.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Video Search API",
        version="1.0.0",
        description="Busca semântica em vídeo via SigLIP + Qwen2.5-VL",
        lifespan=lifespan,
    )

    # CORS — allow_credentials=True combinado com allow_origins=["*"] é
    # rejeitado pelo navegador por especificação (CORS não permite wildcard
    # de origem junto com credentials), então isso falhava silenciosamente
    # em produção se VS_CORS_ORIGINS não fosse configurado. Default agora é
    # fail-safe: nenhuma origem liberada até ser configurado explicitamente.
    raw_origins = os.getenv("VS_CORS_ORIGINS", "").strip()
    origins = [o.strip() for o in raw_origins.split(",") if o.strip()]

    if not origins:
        logger.warning(
            "VS_CORS_ORIGINS não configurado — nenhuma origem CORS liberada "
            "(fail-safe). Configure com a origem do frontend (ex.: "
            "https://seu-projeto.lovable.app) antes de qualquer deploy real. "
            "Ver API.md."
        )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    prefix = "/api/v1"
    app.include_router(health.router, prefix=prefix)
    app.include_router(cameras.router, prefix=prefix)
    app.include_router(index.router, prefix=prefix)
    app.include_router(search.router, prefix=prefix)
    app.include_router(frames.router, prefix=prefix)

    return app


app = create_app()
