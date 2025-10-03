from __future__ import annotations

import asyncio
import json
from typing import Any, Mapping

from apps.mcp_server.service.mcp_service import McpService, RequestContext

__all__ = ["JsonRpcStdioServer"]


class JsonRpcStdioServer:
    """Minimal JSON-RPC 2.0 server over STDIO."""

    def __init__(self, service: McpService, *, deterministic_ids: bool = False) -> None:
        self._service = service
        self._deterministic_ids = deterministic_ids

    async def handle_request(self, message: Mapping[str, Any]) -> dict[str, Any]:
        method = str(message.get("method", ""))
        request_id = message.get("id")
        params = message.get("params") or {}
        if method == "mcp.discover":
            context = RequestContext(
                transport="stdio",
                route="discover",
                method="mcp.discover",
                deterministic_ids=self._deterministic_ids,
            )
            envelope = self._service.discover(context)
            result = envelope.model_dump(by_alias=True)
            response: dict[str, Any] = {"jsonrpc": "2.0", "result": result}
        elif method == "mcp.prompt.get":
            prompt_id = str(params.get("promptId", ""))
            context = RequestContext(
                transport="stdio",
                route="prompt",
                method="mcp.prompt.get",
                deterministic_ids=self._deterministic_ids,
            )
            envelope = self._service.get_prompt(prompt_id, context)
            result = envelope.model_dump(by_alias=True)
            response = {"jsonrpc": "2.0", "result": result}
        elif method == "mcp.tool.invoke":
            tool_id = str(params.get("toolId", ""))
            arguments = params.get("arguments") or {}
            context = RequestContext(
                transport="stdio",
                route="tool",
                method="mcp.tool.invoke",
                deterministic_ids=self._deterministic_ids,
            )
            envelope = self._service.invoke_tool(tool_id=tool_id, arguments=arguments, context=context)
            result = envelope.model_dump(by_alias=True)
            response = {"jsonrpc": "2.0", "result": result}
        else:
            response = {
                "jsonrpc": "2.0",
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }
        if request_id is not None:
            response["id"] = request_id
        return response

    async def handle_notification(self, message: Mapping[str, Any]) -> None:
        # Currently only $/cancel is recognised; all notifications are ignored.
        return None

    async def serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, *, once: bool = False) -> None:
        while True:
            line = await reader.readline()
            if not line:
                break
            payload = line.strip()
            if not payload:
                continue
            try:
                message = json.loads(payload)
            except json.JSONDecodeError:
                error = {
                    "jsonrpc": "2.0",
                    "error": {"code": -32700, "message": "Invalid JSON"},
                }
                writer.write((json.dumps(error) + "\n").encode("utf-8"))
                await writer.drain()
                continue
            if "id" in message:
                response = await self.handle_request(message)
                writer.write((json.dumps(response) + "\n").encode("utf-8"))
                await writer.drain()
            else:
                await self.handle_notification(message)
            if once:
                break
        writer.close()
        await writer.wait_closed()
