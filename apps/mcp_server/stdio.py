from __future__ import annotations

import json
from typing import TextIO

from apps.mcp_server.service import McpService

__all__ = ["StdIoServer"]


class StdIoServer:
    """Minimal JSON-RPC STDIO server for MCP."""

    def __init__(
        self,
        service: McpService,
        *,
        input_stream: TextIO | None = None,
        output_stream: TextIO | None = None,
    ) -> None:
        self._service = service
        self._input = input_stream
        self._output = output_stream

    def handle_message(self, raw: str) -> str:
        request = json.loads(raw)
        method = request.get("method")
        request_id = request.get("id")
        params = request.get("params") or {}

        if method == "mcp.discover":
            envelope = self._service.discover(transport="stdio")
        elif method == "mcp.prompt.get":
            envelope = self._service.get_prompt(
                params.get("domain", ""),
                params.get("name", ""),
                int(params.get("major", 0)),
                transport="stdio",
            )
        elif method == "mcp.tool.invoke":
            tool_name = params.get("tool") or params.get("name") or "unknown"
            payload = params.get("arguments") or params.get("input") or {}
            if isinstance(payload, dict):
                payload_mapping = payload
            else:
                payload_mapping = dict(payload)
            envelope = self._service.invoke_tool(str(tool_name), payload_mapping, transport="stdio")
        else:
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"Unknown method: {method}",
                },
            }
            return json.dumps(response)

        response = {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": envelope.model_dump(),
        }
        return json.dumps(response)

    def serve_forever(self) -> None:
        import sys

        input_stream = self._input or sys.stdin
        output_stream = self._output or sys.stdout

        for line in input_stream:
            line = line.strip()
            if not line:
                continue
            response = self.handle_message(line)
            output_stream.write(response + "\n")
            output_stream.flush()
