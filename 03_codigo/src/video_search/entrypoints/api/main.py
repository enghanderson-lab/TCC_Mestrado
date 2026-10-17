"""FastAPI application factory para o Video Search Engine.

Inicialização:
    uvicorn video_search.entrypoints.api.main:app --host 0.0.0.0 --port 8000

Variáveis de ambiente (ver `video_search.config.Settings` e API.md):
    VS_API_KEY       Chave de autenticação (default: "dev-key")
    VS_DB_PATH       SQLite de câmeras e jobs (default: cameras.db)
    VS_INDEX_DIR     Pasta base dos índices (default: index)
    VS_UPLOAD_DIR    Pasta dos vídeos enviados via upload (default: uploads)
    VS_CORS_ORIGINS  Origens CORS separadas por vírgula (default: nenhuma —
                     fail-safe; configure explicitamente com a origem do
                     frontend antes de qualquer deploy real.)
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ...bootstrap import Container
from ...config import Settings
from .routers import cameras, clips, frames, health, index, search

logger = logging.getLogger(__name__)


def create_app(container: Optional[Container] = None, load_models: bool = True) -> FastAPI:
    """`container`: injetado em testes; None = cria um a partir do ambiente.
    `load_models=False` pula o carregamento dos modelos no startup."""
    container = container or Container(Settings.from_env())

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if load_models:
            logger.info("Carregando modelos...")
            container.models.load_models()
            container.jobs  # instancia o serviço: marca jobs interrompidos no último shutdown
            logger.info("Modelos prontos.")
        yield
        logger.info("Encerrando API.")
        container.close()

    app = FastAPI(
        title="Video Search API",
        version="1.0.0",
        description="Busca semântica em vídeo via SigLIP + Qwen2.5-VL",
        lifespan=lifespan,
    )
    app.state.container = container

    # CORS — allow_credentials=True combinado com allow_origins=["*"] é
    # rejeitado pelo navegador por especificação, então o default é
    # fail-safe: nenhuma origem liberada até ser configurado explicitamente.
    origins = container.settings.cors_origins
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
    for module in (health, cameras, index, search, frames, clips):
        app.include_router(module.router, prefix=prefix)
    return app


app = create_app()
