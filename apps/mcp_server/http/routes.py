from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from apps.mcp_server.service.mcp_service import McpService, RequestContext

__all__ = ["build_router"]


class ToolInvocationPayload(BaseModel):
    arguments: dict[str, Any] = {}


def build_router(service: McpService) -> APIRouter:
    router = APIRouter()

    @router.get("/mcp/discover")
    def discover() -> dict[str, Any]:
        context = RequestContext(transport="http", route="discover", method="mcp.discover")
        envelope = service.discover(context)
        return envelope.model_dump(by_alias=True)

    @router.get("/mcp/prompt/{prompt_id}")
    def get_prompt(prompt_id: str) -> dict[str, Any]:
        context = RequestContext(transport="http", route="prompt", method="mcp.prompt.get")
        envelope = service.get_prompt(prompt_id, context)
        return envelope.model_dump(by_alias=True)

    @router.post("/mcp/tool/{tool_id}")
    def invoke_tool(tool_id: str, payload: ToolInvocationPayload) -> dict[str, Any]:
        context = RequestContext(transport="http", route="tool", method="mcp.tool.invoke")
        envelope = service.invoke_tool(tool_id=tool_id, arguments=payload.arguments, context=context)
        return envelope.model_dump(by_alias=True)

    @router.get("/healthz")
    def health() -> dict[str, Any]:
        context = RequestContext(transport="http", route="health", method="mcp.health")
        return service.health(context)

    return router
