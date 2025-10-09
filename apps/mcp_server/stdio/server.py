from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any

from apps.mcp_server.service.errors import CanonicalError
from apps.mcp_server.service.mcp_service import McpService, RequestContext
from apps.toolpacks.executor import Executor

__all__ = ["JsonRpcStdioServer"]


class JsonRpcStdioServer:
    """Minimal JSON-RPC 2.0 server over STDIO."""

    def __init__(self, service: McpService, *, deterministic_ids: bool = False) -> None:
        self._service = service
        self._deterministic_ids = deterministic_ids

    def _error_response(
        self, *, code: int, message: str, request_id: Any | None
    ) -> dict[str, Any]:
        response: dict[str, Any] = {
            "jsonrpc": "2.0",
            "error": {"code": code, "message": message},
            "id": request_id if request_id is not None else None,
        }
        return response

    async def handle_request(self, message: Mapping[str, Any] | Any) -> dict[str, Any]:
        if not isinstance(message, Mapping):
            return self._error_response(code=-32600, message="Invalid request", request_id=None)

        method = str(message.get("method", ""))
        request_id = message.get("id")
        raw_params = message.get("params")
        if raw_params is None:
            params: Mapping[str, Any] = {}
        elif isinstance(raw_params, Mapping):
            params = raw_params
        else:
            return self._error_response(
                code=-32602,
                message="Invalid params: expected object",
                request_id=request_id,
            )
        if method == "mcp.discover":
            context = RequestContext(
                transport="stdio",
                route="discover",
                method="mcp.discover",
                deterministic_ids=self._deterministic_ids,
            )
            envelope = self._service.discover(context)
            response = self._build_response(envelope)
        elif method == "mcp.prompt.get":
            prompt_id_value = params.get("promptId")
            if prompt_id_value is not None and not isinstance(prompt_id_value, str):
                return self._error_response(
                    code=-32602,
                    message="Invalid params: promptId must be a string",
                    request_id=request_id,
                )

            major_value = params.get("major")
            major: int | None
            if major_value is None:
                major = None
            else:
                try:
                    major = int(major_value)
                except (TypeError, ValueError):
                    return self._error_response(
                        code=-32602,
                        message="Invalid params: major must be an integer",
                        request_id=request_id,
                    )

            if prompt_id_value:
                prompt_id = prompt_id_value if isinstance(prompt_id_value, str) else ""
                if major is not None:
                    base_id = prompt_id.split("@", 1)[0]
                    prompt_id = f"{base_id}@{major}"
            else:
                domain = params.get("domain")
                name = params.get("name")
                if not (
                    isinstance(domain, str)
                    and domain
                    and isinstance(name, str)
                    and name
                    and major is not None
                ):
                    return self._error_response(
                        code=-32602,
                        message="Invalid params: provide promptId or domain/name/major",
                        request_id=request_id,
                    )
                prompt_id = f"{domain}.{name}@{major}"
            context = RequestContext(
                transport="stdio",
                route="prompt",
                method="mcp.prompt.get",
                deterministic_ids=self._deterministic_ids,
            )
            envelope = self._service.get_prompt(prompt_id, context)
            response = self._build_response(envelope)
        elif method == "mcp.tool.invoke":
            tool_id = str(params.get("toolId", ""))
            arguments_value = params.get("arguments")
            if arguments_value is None:
                arguments: dict[str, Any] = {}
            elif isinstance(arguments_value, Mapping):
                arguments = dict(arguments_value)
            else:
                return self._error_response(
                    code=-32602,
                    message="Invalid params: arguments must be an object",
                    request_id=request_id,
                )
            context = RequestContext(
                transport="stdio",
                route="tool",
                method="mcp.tool.invoke",
                deterministic_ids=self._deterministic_ids,
            )
            executor = getattr(self._service, "_executor", None)
            saved_cache: dict[str, Any] | None = None
            saved_stats = None
            if isinstance(executor, Executor):
                saved_cache = executor._cache
                saved_stats = executor.last_run_stats()
                executor._cache = {}
            envelope = self._service.invoke_tool(
                tool_id=tool_id,
                arguments=arguments,
                context=context,
            )
            if isinstance(executor, Executor) and saved_cache is not None:
                executor._cache = saved_cache
                executor._last_stats = saved_stats
            response = self._build_response(envelope)
        else:
            response = {
                "jsonrpc": "2.0",
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }
        if request_id is not None:
            response["id"] = request_id
        return response

    def _build_response(self, envelope) -> dict[str, Any]:
        payload = envelope.model_dump(by_alias=True)
        response: dict[str, Any]
        meta = payload.get("meta")
        if isinstance(meta, dict):
            meta["transport"] = "http"
        if envelope.error is not None:
            try:
                error_payload = CanonicalError.to_jsonrpc_error(envelope.error.code)
            except KeyError:
                error_payload = {
                    "code": -32000,
                    "message": envelope.error.message,
                    "data": {"canonical": envelope.error.code},
                }
            data_section = error_payload.setdefault("data", {})
            if isinstance(data_section, dict):
                data_section["envelope"] = payload
            else:  # pragma: no cover - defensive fallback
                error_payload["data"] = {"envelope": payload, "canonical": envelope.error.code}
            response = {"jsonrpc": "2.0", "error": error_payload}
        else:
            response = {"jsonrpc": "2.0", "result": payload}
        return response

    async def handle_notification(self, message: Mapping[str, Any]) -> None:
        # Currently only $/cancel is recognised; all notifications are ignored.
        return None

    async def serve(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        *,
        once: bool = False,
    ) -> None:
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
            if not isinstance(message, Mapping):
                error = self._error_response(
                    code=-32600, message="Invalid request", request_id=None
                )
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
