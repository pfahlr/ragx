"""HTTP transport for the MCP server bootstrap."""

from __future__ import annotations

from fastapi import FastAPI, Request

from .observability import log_event
from .service import McpService


def create_app(service: McpService) -> FastAPI:
    app = FastAPI(title="RAGX MCP Server", version="0.0.1")

    @app.get("/mcp/discover")
    def discover(request: Request) -> dict[str, object]:
        envelope = service.discover()
        trace_id = envelope.ensure_trace_id()
        log_event(
            trace_id=trace_id,
            transport="http",
            status="ok" if envelope.ok else "error",
            route=str(request.url.path),
        )
        return envelope.to_dict()

    @app.get("/mcp/prompt/{domain}/{name}/{major}")
    def get_prompt(domain: str, name: str, major: int, request: Request) -> dict[str, object]:
        envelope = service.get_prompt(domain, name, major)
        trace_id = envelope.ensure_trace_id()
        log_event(
            trace_id=trace_id,
            transport="http",
            status="ok" if envelope.ok else "error",
            route=str(request.url.path),
        )
        return envelope.to_dict()

    @app.post("/mcp/tool/{tool_name}")
    async def invoke_tool(tool_name: str, request: Request) -> dict[str, object]:
        payload = await request.json()
        envelope = service.invoke_tool(tool_name, payload)
        trace_id = envelope.ensure_trace_id()
        log_event(
            trace_id=trace_id,
            transport="http",
            status="ok" if envelope.ok else "error",
            route=str(request.url.path),
            tool=tool_name,
        )
        return envelope.to_dict()

    return app
