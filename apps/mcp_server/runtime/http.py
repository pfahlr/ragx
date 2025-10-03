"""HTTP transport for the MCP server."""

from __future__ import annotations

from fastapi import FastAPI, status
from fastapi.responses import JSONResponse

from apps.mcp_server.models import Envelope
from apps.mcp_server.runtime.service import McpService

__all__ = ["create_http_app"]


def _serialise(envelope: Envelope) -> JSONResponse:
    return JSONResponse(envelope.to_serialisable(), status_code=status.HTTP_200_OK)


def create_http_app(service: McpService) -> FastAPI:
    """Create a FastAPI application exposing the MCP HTTP surface."""

    app = FastAPI(title="RAGX MCP Server", version="0.1.0")

    @app.get("/mcp/discover")
    def discover() -> JSONResponse:
        envelope = service.discover()
        return _serialise(envelope)

    @app.get("/mcp/prompt/{domain}/{name}/{major}")
    def get_prompt(domain: str, name: str, major: int) -> JSONResponse:
        envelope = service.get_prompt(domain=domain, name=name, major=major)
        return _serialise(envelope)

    @app.post("/mcp/tool/{tool_name}")
    def invoke_tool(tool_name: str, body: dict[str, object] | None = None) -> JSONResponse:
        arguments = {}
        if body and isinstance(body, dict):
            arguments = dict(body.get("arguments", {}))
        envelope = service.invoke_tool(tool_name=tool_name, arguments=arguments)
        return _serialise(envelope)

    return app
