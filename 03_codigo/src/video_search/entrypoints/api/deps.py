"""Dependências FastAPI: resolvem os serviços a partir do Container guardado
em `app.state.container` (criado em main.py). Testes podem substituí-las via
`app.dependency_overrides`."""

from fastapi import Depends, Request

from ...application.camera_service import CameraService
from ...application.index_jobs import IndexJobService
from ...application.search_service import SearchService
from ...bootstrap import Container


def get_container(request: Request) -> Container:
    return request.app.state.container


def get_search_service(c: Container = Depends(get_container)) -> SearchService:
    return c.search


def get_job_service(c: Container = Depends(get_container)) -> IndexJobService:
    return c.jobs


def get_camera_service(c: Container = Depends(get_container)) -> CameraService:
    return c.cameras
