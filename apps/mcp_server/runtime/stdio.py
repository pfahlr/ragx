"""STDIO JSON-RPC transport for the MCP server."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any, TextIO

from apps.mcp_server.runtime.service import McpService

__all__ = ["JsonRpcRequest", "JsonRpcResponse", "McpStdIoServer"]


@dataclass(slots=True)
class JsonRpcRequest:
    """Incoming JSON-RPC request."""

    id: str | int
    method: str
    params: dict[str, Any]


@dataclass(slots=True)
class JsonRpcResponse:
    """Outgoing JSON-RPC response."""

    id: str | int
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    jsonrpc: str = "2.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "jsonrpc": self.jsonrpc,
            "id": self.id,
            "result": self.result,
            "error": self.error,
        }


class McpStdIoServer:
    """Minimal STDIO server handling a subset of MCP JSON-RPC calls."""

    def __init__(
        self,
        *,
        service: McpService,
        input_stream: TextIO | None = None,
        output_stream: TextIO | None = None,
    ) -> None:
        self._service = service
        self._input = input_stream or sys.stdin
        self._output = output_stream or sys.stdout

    def handle_request(self, request: JsonRpcRequest) -> JsonRpcResponse:
        """Handle a structured JSON-RPC request."""

        try:
            if request.method == "mcp.discover":
                envelope = self._service.discover()
            elif request.method == "mcp.prompt.get":
                envelope = self._service.get_prompt(
                    domain=request.params["domain"],
                    name=request.params["name"],
                    major=request.params.get("major", 1),
                )
            elif request.method == "mcp.tool.invoke":
                envelope = self._service.invoke_tool(
                    tool_name=request.params["tool"],
                    arguments=request.params.get("arguments", {}),
                )
            else:
                return JsonRpcResponse(
                    id=request.id,
                    error={"code": -32601, "message": "Method not found"},
                )
        except KeyError as exc:
            return JsonRpcResponse(
                id=request.id,
                error={"code": -32602, "message": f"Missing parameter: {exc.args[0]}"},
            )

        return JsonRpcResponse(id=request.id, result=envelope.to_serialisable())

    def handle_line(self, payload: str) -> JsonRpcResponse:
        """Parse a JSON line and return the corresponding response."""

        message = json.loads(payload)
        request = JsonRpcRequest(
            id=message["id"],
            method=message["method"],
            params=dict(message.get("params", {})),
        )
        return self.handle_request(request)

    def dumps_response(self, response: JsonRpcResponse) -> str:
        """Serialise a response to a JSON string."""

        return json.dumps(response.to_dict(), separators=(",", ":"))

    def serve_forever(self) -> None:  # pragma: no cover - requires IO integration tests
        """Run the STDIO loop until EOF."""

        for raw in self._input:
            line = raw.strip()
            if not line:
                continue
            response = self.handle_line(line)
            self._output.write(self.dumps_response(response) + "\n")
            self._output.flush()
