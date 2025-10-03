from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Response, status
from pydantic import BaseModel

from apps.mcp_server.service.mcp_service import McpService, RequestContext

__all__ = ["build_router"]


class ToolInvocationPayload(BaseModel):
    arguments: dict[str, Any] = {}


_ERROR_STATUS: dict[str, int] = {
    "INVALID_INPUT": status.HTTP_400_BAD_REQUEST,
    "NOT_FOUND": status.HTTP_404_NOT_FOUND,
    "RATE_LIMIT": status.HTTP_429_TOO_MANY_REQUESTS,
    "TIMEOUT": status.HTTP_504_GATEWAY_TIMEOUT,
    "UNAVAILABLE": status.HTTP_503_SERVICE_UNAVAILABLE,
    "INTERNAL": status.HTTP_500_INTERNAL_SERVER_ERROR,
    "NONDETERMINISTIC": status.HTTP_409_CONFLICT,
    "UNSUPPORTED": status.HTTP_501_NOT_IMPLEMENTED,
}


def _serialise_envelope(response: Response, envelope_payload: dict[str, Any]) -> dict[str, Any]:
    if not envelope_payload.get("ok", True):
        error = envelope_payload.get("error") or {}
        code = error.get("code")
        if code is not None:
            response.status_code = _ERROR_STATUS.get(code, status.HTTP_500_INTERNAL_SERVER_ERROR)
    return envelope_payload


def build_router(service: McpService) -> APIRouter:
    router = APIRouter()

    @router.get("/mcp/discover")
    def discover(response: Response) -> dict[str, Any]:
        context = RequestContext(transport="http", route="discover", method="mcp.discover")
        envelope = service.discover(context)
        payload = envelope.model_dump(by_alias=True)
        return _serialise_envelope(response, payload)

    @router.get("/mcp/prompt/{prompt_id}")
    def get_prompt(prompt_id: str, response: Response) -> dict[str, Any]:
        context = RequestContext(transport="http", route="prompt", method="mcp.prompt.get")
        envelope = service.get_prompt(prompt_id, context)
        payload = envelope.model_dump(by_alias=True)
        return _serialise_envelope(response, payload)

    @router.post("/mcp/tool/{tool_id}")
    def invoke_tool(
        tool_id: str, payload: ToolInvocationPayload, response: Response
    ) -> dict[str, Any]:
        context = RequestContext(transport="http", route="tool", method="mcp.tool.invoke")
        envelope = service.invoke_tool(
            tool_id=tool_id,
            arguments=payload.arguments,
            context=context,
        )
        payload_dict = envelope.model_dump(by_alias=True)
        return _serialise_envelope(response, payload_dict)

    @router.get("/healthz")
    def health() -> dict[str, Any]:
        context = RequestContext(transport="http", route="health", method="mcp.health")
        return service.health(context)

    return router
