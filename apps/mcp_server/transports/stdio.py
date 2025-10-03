from __future__ import annotations

import json
import logging
import sys
from collections.abc import Mapping
from typing import Any

from apps.mcp_server.models import Envelope
from apps.mcp_server.service import McpService

LOGGER = logging.getLogger(__name__)


class McpStdioServer:
    """Minimal JSON-RPC handler for STDIO transport."""

    def __init__(
        self,
        service: McpService,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        self._service = service
        self._logger = logger or LOGGER

    def serve_forever(self) -> None:
        """Consume JSON-RPC messages from stdin and write responses to stdout."""

        for raw_line in sys.stdin:
            response = self.process_line(raw_line)
            if response is None:
                continue
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()

    def process_line(self, raw_line: str) -> dict[str, Any] | None:
        raw_line = raw_line.strip()
        if not raw_line:
            return None
        try:
            message = json.loads(raw_line)
        except json.JSONDecodeError:
            return self._error_response(None, code=-32700, message="Parse error")
        if not isinstance(message, Mapping):
            return self._error_response(None, code=-32600, message="Invalid request")
        if "id" not in message:
            # Notification, no response expected.
            return None
        return self.handle_message(message)

    def handle_message(self, message: Mapping[str, Any]) -> dict[str, Any]:
        request_id = message.get("id")
        method = message.get("method")
        params = message.get("params") if isinstance(message.get("params"), Mapping) else {}
        try:
            result = self._dispatch(method, params)
        except Exception as exc:  # pragma: no cover - defensive guard
            self._logger.exception("Unhandled MCP STDIO error")
            return self._error_response(request_id, code=-32603, message=str(exc))
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result.to_dict(),
        }

    def _dispatch(self, method: Any, params: Mapping[str, Any]) -> Envelope:
        if method == "mcp.discover":
            return self._service.discover(transport="stdio")
        if method == "mcp.prompt.get":
            domain = str(params.get("domain", ""))
            name = str(params.get("name", ""))
            major = str(params.get("major", ""))
            return self._service.get_prompt(domain, name, major, transport="stdio")
        if method == "mcp.tool.invoke":
            tool = params.get("tool")
            if not isinstance(tool, str) or not tool:
                raise ValueError("tool parameter must be provided")
            arguments = params.get("arguments")
            if not isinstance(arguments, Mapping):
                arguments = {}
            return self._service.invoke_tool(tool, arguments, transport="stdio")
        raise ValueError(f"Unsupported MCP method: {method}")

    @staticmethod
    def _error_response(request_id: Any, *, code: int, message: str) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        }
