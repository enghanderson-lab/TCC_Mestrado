from fastapi import HTTPException, Request, Security
from fastapi.security.api_key import APIKeyHeader

_KEY_HEADER = APIKeyHeader(name="X-Api-Key", auto_error=False)


def is_valid_key(app, api_key: str) -> bool:
    return api_key == app.state.container.settings.api_key


def require_api_key(request: Request, api_key: str = Security(_KEY_HEADER)) -> str:
    if not is_valid_key(request.app, api_key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key
