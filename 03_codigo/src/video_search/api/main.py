"""FastAPI application factory para o Video Search Engine.

Inicialização:
    uvicorn video_search.api.main:app --host 0.0.0.0 --port 8000

Variáveis de ambiente:
    VS_API_KEY   Chave de autenticação (default: "dev-key")
    VS_DB_PATH   Caminho do SQLite de câmeras (default: cameras.db)
    VS_CORS_ORIGINS  Origens CORS separadas por vírgula (default: *)
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

    # CORS — em prod, restringir VS_CORS_ORIGINS à origem do Lovable
    raw_origins = os.getenv("VS_CORS_ORIGINS", "*")
    origins = [o.strip() for o in raw_origins.split(",")]

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
