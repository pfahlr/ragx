from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any

from fastapi import Body, FastAPI

from apps.mcp_server.service import McpService


def _extract_arguments(payload: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not payload:
        return {}
    for key in ("arguments", "inputs"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            return value
    return payload if isinstance(payload, Mapping) else {}


Payload = Annotated[dict[str, Any] | None, Body()]


def create_app(service: McpService) -> FastAPI:
    app = FastAPI()

    @app.get("/mcp/discover")
    async def discover() -> Any:
        envelope = service.discover(transport="http")
        return envelope.to_dict()

    @app.get("/mcp/prompt/{domain}/{name}/{major}")
    async def get_prompt(domain: str, name: str, major: str) -> Any:
        envelope = service.get_prompt(domain, name, major, transport="http")
        return envelope.to_dict()

    @app.post("/mcp/tool/{tool_name}")
    async def invoke_tool(
        tool_name: str,
        payload: Payload = None,
    ) -> Any:
        arguments = _extract_arguments(payload)
        envelope = service.invoke_tool(tool_name, arguments, transport="http")
        return envelope.to_dict()

    return app
