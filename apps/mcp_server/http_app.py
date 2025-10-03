from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException

from apps.mcp_server.service import McpService

__all__ = ["create_http_app"]


def create_http_app(service: McpService) -> FastAPI:
    app = FastAPI()

    @app.get("/mcp/discover")
    def discover() -> dict[str, Any]:
        envelope = service.discover(transport="http")
        return envelope.model_dump()

    @app.get("/mcp/prompt/{domain}/{name}/{major}")
    def get_prompt(domain: str, name: str, major: int) -> dict[str, Any]:
        envelope = service.get_prompt(domain, name, major, transport="http")
        return envelope.model_dump()

    @app.post("/mcp/tool/{tool_name}")
    def invoke_tool(tool_name: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        if payload is None:
            raise HTTPException(status_code=400, detail="Missing JSON payload")
        envelope = service.invoke_tool(tool_name, payload, transport="http")
        return envelope.model_dump()

    return app
