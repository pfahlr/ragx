from __future__ import annotations

from fastapi import FastAPI

from apps.mcp_server.service.mcp_service import McpService

from .routes import build_router

__all__ = ["create_app"]


def create_app(service: McpService, *, enable_openapi: bool = False) -> FastAPI:
    """Return a FastAPI application exposing the MCP HTTP endpoints."""

    docs_url = "/docs" if enable_openapi else None
    openapi_url = "/openapi.json" if enable_openapi else None
    app = FastAPI(title="RAGX MCP Server", docs_url=docs_url, openapi_url=openapi_url)
    app.include_router(build_router(service))
    return app
