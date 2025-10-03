from __future__ import annotations

from typing import Any

from apps.mcp_server.models import Envelope

__all__ = ["McpService"]


class McpService:
    """In-memory placeholder service for MCP transports."""

    def __init__(self, *, service_name: str = "ragx.mcp", version: str = "0.0.0") -> None:
        self._service_name = service_name
        self._version = version

    def discover(self, *, transport: str) -> Envelope:
        data = {
            "service": self._service_name,
            "version": self._version,
            "tools": [],
            "prompts": [],
        }
        return Envelope.success(data=data, transport=transport)

    def get_prompt(self, domain: str, name: str, major: int, *, transport: str) -> Envelope:
        data = {
            "domain": domain,
            "name": name,
            "major": major,
            "content": None,
            "message": "Prompt registry not yet initialised.",
        }
        return Envelope.success(data=data, transport=transport)

    def invoke_tool(self, tool_name: str, payload: dict[str, Any], *, transport: str) -> Envelope:
        data = {
            "tool": tool_name,
            "input": payload,
            "message": "Tool execution not yet implemented.",
        }
        return Envelope.success(data=data, transport=transport)
