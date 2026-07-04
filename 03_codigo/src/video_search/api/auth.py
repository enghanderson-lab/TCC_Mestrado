import os
from fastapi import HTTPException, Query, Security
from fastapi.security.api_key import APIKeyHeader

_KEY_HEADER = APIKeyHeader(name="X-Api-Key", auto_error=False)

_API_KEY: str = os.getenv("VS_API_KEY", "dev-key")


def require_api_key(api_key: str = Security(_KEY_HEADER)) -> str:
    if api_key != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key


def check_ws_key(api_key: str = Query(default="")) -> str:
    """Query-param auth for WebSocket (browsers can't set headers in WS)."""
    if api_key != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key
