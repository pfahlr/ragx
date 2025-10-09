from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from apps.mcp_server.service.errors import CanonicalError
from apps.mcp_server.service.mcp_service import McpService, RequestContext

__all__ = ["build_router"]


class ToolInvocationPayload(BaseModel):
    arguments: dict[str, Any] = {}


def build_router(service: McpService, *, deterministic_ids: bool = False) -> APIRouter:
    router = APIRouter()

    def _context(route: str, method: str) -> RequestContext:
        return RequestContext(
            transport="http",
            route=route,
            method=method,
            deterministic_ids=deterministic_ids,
        )

    def _render(envelope) -> JSONResponse:
        payload = envelope.model_dump(by_alias=True)
        status = 200
        if not envelope.ok and envelope.error is not None:
            try:
                status = CanonicalError.to_http_status(envelope.error.code)
            except KeyError:
                status = 400
        return JSONResponse(content=payload, status_code=status)

    @router.get("/mcp/discover")
    def discover() -> JSONResponse:
        context = _context("discover", "mcp.discover")
        envelope = service.discover(context)
        return _render(envelope)

    @router.get("/mcp/prompt/{prompt_id}")
    def get_prompt(prompt_id: str) -> JSONResponse:
        context = _context("prompt", "mcp.prompt.get")
        envelope = service.get_prompt(prompt_id, context)
        return _render(envelope)

    @router.post("/mcp/tool/{tool_id}")
    def invoke_tool(tool_id: str, payload: ToolInvocationPayload) -> JSONResponse:
        context = _context("tool", "mcp.tool.invoke")
        envelope = service.invoke_tool(
            tool_id=tool_id,
            arguments=payload.arguments,
            context=context,
        )
        return _render(envelope)

    @router.get("/healthz")
    def health() -> dict[str, Any]:
        context = _context("health", "mcp.health")
        return service.health(context)

    return router
