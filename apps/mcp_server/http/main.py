from __future__ import annotations

from typing import Any

try:  # pragma: no cover - optional dependency guard
    from fastapi import FastAPI
except ModuleNotFoundError:  # pragma: no cover - fallback when not installed
    FastAPI = None  # type: ignore[assignment]

from apps.mcp_server.service.mcp_service import McpService

from .routes import build_router

__all__ = ["create_app"]


def create_app(
    service: McpService,
    *,
    enable_openapi: bool = False,
    deterministic_ids: bool = False,
) -> Any:
    """Return a FastAPI application exposing the MCP HTTP endpoints."""

    if FastAPI is None:
        msg = "fastapi is required to build the MCP HTTP application"
        raise RuntimeError(msg)

    docs_url = "/docs" if enable_openapi else None
    openapi_url = "/openapi.json" if enable_openapi else None
    app = FastAPI(title="RAGX MCP Server", docs_url=docs_url, openapi_url=openapi_url)
    app.include_router(build_router(service, deterministic_ids=deterministic_ids))
    return app
