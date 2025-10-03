"""Core MCP service skeleton returning placeholder envelopes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .envelope import Envelope


@dataclass(slots=True)
class McpService:
    """Placeholder MCP service exposing discovery, prompts, and tools."""

    service_name: str = "ragx.mcp"
    version: str = "0.0.1"

    def discover(self) -> Envelope:
        data = {
            "kind": "discovery",
            "service": self.service_name,
            "version": self.version,
            "toolpacks": [],
            "prompts": [],
        }
        return Envelope.success(data=data)

    def get_prompt(self, domain: str, name: str, major: int) -> Envelope:
        data = {
            "kind": "prompt",
            "domain": domain,
            "name": name,
            "major": major,
            "content": None,
            "status": "placeholder",
        }
        return Envelope.success(data=data)

    def invoke_tool(self, tool_name: str, payload: Mapping[str, Any]) -> Envelope:
        data = {
            "kind": "tool",
            "tool_name": tool_name,
            "status": "not_implemented",
            "echo": dict(payload),
        }
        return Envelope.success(data=data)
