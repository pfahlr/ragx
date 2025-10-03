"""STDIO JSON-RPC transport for the MCP server bootstrap."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from typing import IO, Any

from .envelope import Envelope
from .observability import log_event
from .service import McpService


class McpStdioTransport:
    """Minimal STDIO JSON-RPC handler dispatching to :class:`McpService`."""

    def __init__(
        self,
        service: McpService,
        *,
        reader: IO[str] | None = None,
        writer: IO[str] | None = None,
    ) -> None:
        self._service = service
        self._reader = reader or sys.stdin
        self._writer = writer or sys.stdout

    def handle_once(self) -> bool:
        line = self._reader.readline()
        if line == "":
            return False
        line = line.strip()
        if not line:
            return False

        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            envelope = Envelope.failure([
                {"code": "invalid_json", "message": "Failed to decode JSON-RPC request."}
            ])
            self._write_response(message_id=None, envelope=envelope)
            return True

        method = message.get("method")
        params = message.get("params") or {}
        message_id = message.get("id")

        envelope = self._dispatch(method, params)
        trace_id = envelope.ensure_trace_id()
        log_event(
            trace_id=trace_id,
            transport="stdio",
            status="ok" if envelope.ok else "error",
            method=method,
        )
        self._write_response(message_id=message_id, envelope=envelope)
        return True

    def serve_forever(self) -> None:
        while self.handle_once():
            continue

    def _dispatch(self, method: str | None, params: Mapping[str, Any]) -> Envelope:
        if method == "mcp.discover":
            return self._service.discover()
        if method == "mcp.prompt.get":
            domain = str(params.get("domain", ""))
            name = str(params.get("name", ""))
            major = int(params.get("major", 0))
            return self._service.get_prompt(domain, name, major)
        if method == "mcp.tool.invoke":
            tool_name = str(params.get("tool"))
            payload = params.get("input")
            if tool_name is None or payload is None:
                return Envelope.failure(
                    [{"code": "invalid_params", "message": "tool and input are required."}]
                )
            if not isinstance(payload, Mapping):
                return Envelope.failure(
                    [{"code": "invalid_params", "message": "input must be an object."}]
                )
            return self._service.invoke_tool(tool_name, payload)
        return Envelope.failure(
            [{"code": "method_not_found", "message": f"Unknown method: {method}"}]
        )

    def _write_response(self, *, message_id: Any, envelope: Envelope) -> None:
        response = {
            "jsonrpc": "2.0",
            "id": message_id,
            "result": envelope.to_dict(),
        }
        self._writer.write(json.dumps(response) + "\n")
        self._writer.flush()
