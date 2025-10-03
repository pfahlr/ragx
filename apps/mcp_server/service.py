from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from apps.mcp_server.models import Envelope

LOGGER = logging.getLogger(__name__)


class McpService:
    """Bootstrap implementation returning placeholder envelopes."""

    def __init__(self, *, logger: logging.Logger | None = None) -> None:
        self._logger = logger or LOGGER

    def discover(self, *, transport: str) -> Envelope:
        trace_id = self._new_trace_id()
        data = {
            "version": "0.1.0",
            "tools": [],
            "prompts": [],
        }
        envelope = Envelope.success(data=data, trace_id=trace_id, transport=transport)
        self._log("mcp.discover", transport=transport, trace_id=trace_id, status="ok")
        return envelope

    def get_prompt(
        self,
        domain: str,
        name: str,
        major: str,
        *,
        transport: str,
    ) -> Envelope:
        trace_id = self._new_trace_id()
        prompt = {
            "domain": domain,
            "name": name,
            "major": str(major),
            "content": (
                "Placeholder prompt content. Configure prompts under apps/mcp_server/prompts/."
            ),
        }
        envelope = Envelope.success(data=prompt, trace_id=trace_id, transport=transport)
        self._log("mcp.prompt.get", transport=transport, trace_id=trace_id, status="ok")
        return envelope

    def invoke_tool(
        self,
        tool: str,
        arguments: Mapping[str, Any] | None,
        *,
        transport: str,
    ) -> Envelope:
        trace_id = self._new_trace_id()
        payload = {
            "tool": tool,
            "inputs": dict(arguments or {}),
            "status": "placeholder",
        }
        envelope = Envelope.success(data=payload, trace_id=trace_id, transport=transport)
        self._log("mcp.tool.invoke", transport=transport, trace_id=trace_id, status="ok")
        return envelope

    def _new_trace_id(self) -> str:
        return str(uuid4())

    def _log(self, event: str, *, transport: str, trace_id: str, status: str) -> None:
        record = {
            "event": event,
            "traceId": trace_id,
            "transport": transport,
            "status": status,
        }
        self._logger.info(json.dumps(record, sort_keys=True))
