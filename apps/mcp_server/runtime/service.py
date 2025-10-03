"""Service layer powering the MCP server transports."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from apps.mcp_server.models import Envelope

__all__ = ["McpService"]


class McpService:
    """Implements the core MCP RPCs with placeholder payloads."""

    def __init__(
        self,
        *,
        server_name: str = "ragx.mcp",
        server_version: str = "0.1.0",
        logger: logging.Logger | None = None,
    ) -> None:
        self._server_name = server_name
        self._server_version = server_version
        self._start_time = datetime.now(tz=UTC)
        self._logger = logger or logging.getLogger("ragx.mcp.server")
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(message)s")
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
        self._logger.setLevel(logging.INFO)

    def discover(self) -> Envelope:
        """Return placeholder discovery metadata."""

        uptime = int((datetime.now(tz=UTC) - self._start_time).total_seconds())
        payload = {
            "server": {"name": self._server_name, "version": self._server_version},
            "tools": [],
            "prompts": [],
            "health": {"status": "ok", "uptimeSec": uptime},
        }
        envelope = Envelope.success(
            data=payload,
            tool="mcp.discover",
            trace_id=self._new_trace_id(),
        )
        self._emit_log("discover", envelope)
        return envelope

    def get_prompt(self, domain: str, name: str, major: int | str) -> Envelope:
        """Return a placeholder prompt payload."""

        prompt_id = f"{domain}/{name}"
        payload = {
            "prompt": {
                "id": prompt_id,
                "version": {"major": int(major), "description": "placeholder"},
                "body": "",
                "spec": {
                    "ref": None,
                    "notes": "prompt registry not implemented",
                },
            }
        }
        envelope = Envelope.success(
            data=payload,
            tool="mcp.prompt.get",
            trace_id=self._new_trace_id(),
        )
        self._emit_log("prompt", envelope, extra={"prompt_id": prompt_id})
        return envelope

    def invoke_tool(self, tool_name: str, arguments: Mapping[str, Any] | None = None) -> Envelope:
        """Return a placeholder tool invocation response."""

        payload = {
            "echo": {
                "tool": tool_name,
                "arguments": dict(arguments or {}),
                "message": "tool execution not yet implemented",
            }
        }
        envelope = Envelope.success(
            data=payload,
            tool="mcp.tool.invoke",
            trace_id=self._new_trace_id(),
        )
        self._emit_log("tool", envelope, extra={"tool": tool_name})
        return envelope

    def _emit_log(
        self,
        event: str,
        envelope: Envelope,
        *,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        record = {
            "event": event,
            "traceId": envelope.meta.traceId,
            "tool": envelope.meta.tool,
            "ok": envelope.ok,
        }
        if extra:
            record.update(dict(extra))
        self._logger.info(json.dumps(record, sort_keys=True))

    @staticmethod
    def _new_trace_id() -> str:
        return str(uuid4())
