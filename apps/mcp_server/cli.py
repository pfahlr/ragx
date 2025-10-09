from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, cast

try:  # pragma: no cover - optional dependency guard
    import uvicorn
except ModuleNotFoundError:  # pragma: no cover - fallback when not installed
    uvicorn = None

from apps.mcp_server.http import create_app
from apps.mcp_server.service.mcp_service import McpService, RequestContext
from apps.mcp_server.stdio import JsonRpcStdioServer

_DEFAULT_TOOLPACKS = Path("apps/mcp_server/toolpacks")
_DEFAULT_PROMPTS = Path("apps/mcp_server/prompts")
_DEFAULT_SCHEMAS = Path("apps/mcp_server/schemas/mcp")

_UVICORN_LOG_LEVELS = {
    "DEBUG": "debug",
    "INFO": "info",
    "WARN": "warning",
    "ERROR": "error",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RAGX MCP server")
    parser.add_argument("--http", action="store_true", help="Enable HTTP transport")
    parser.add_argument("--stdio", action="store_true", help="Enable STDIO JSON-RPC transport")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP host")
    parser.add_argument("--port", type=int, default=3333, help="HTTP port")
    parser.add_argument(
        "--max-connections",
        type=int,
        default=256,
        help="Maximum concurrent HTTP connections",
    )
    parser.add_argument(
        "--shutdown-grace",
        type=int,
        default=10,
        help="Grace period in seconds for shutdown",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARN", "ERROR"],
        default="INFO",
        help="Logging level",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a deterministic single-request cycle and exit",
    )
    parser.add_argument(
        "--deterministic-ids",
        action="store_true",
        help="Use deterministic UUID5 request identifiers",
    )
    parser.add_argument(
        "--log-dir",
        default="runs",
        help="Directory for structured logs",
    )
    parser.add_argument(
        "--max-input-bytes",
        type=int,
        default=None,
        help="Override global maximum input payload size in bytes",
    )
    parser.add_argument(
        "--max-output-bytes",
        type=int,
        default=None,
        help="Override global maximum output payload size in bytes",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=None,
        help="Override global execution timeout in milliseconds",
    )
    return parser.parse_args(argv)


async def _stdio_loop(server: JsonRpcStdioServer, *, once: bool) -> None:
    while True:
        line = await asyncio.to_thread(sys.stdin.readline)
        if not line:
            break
        payload = line.strip()
        if not payload:
            continue
        try:
            message = json.loads(payload)
        except json.JSONDecodeError:
            await asyncio.to_thread(sys.stdout.write, json.dumps({
                "jsonrpc": "2.0",
                "error": {"code": -32700, "message": "Invalid JSON"},
            }) + "\n")
            await asyncio.to_thread(sys.stdout.flush)
            continue
        if "id" in message:
            response = await server.handle_request(message)
            await asyncio.to_thread(sys.stdout.write, json.dumps(response) + "\n")
            await asyncio.to_thread(sys.stdout.flush)
        else:
            await server.handle_notification(message)
        if once:
            break


async def _run_server(args: argparse.Namespace) -> None:
    toolpacks_dir = _DEFAULT_TOOLPACKS
    prompts_dir = _DEFAULT_PROMPTS
    schema_dir = _DEFAULT_SCHEMAS
    log_dir = Path(args.log_dir)

    service = McpService.create(
        toolpacks_dir=toolpacks_dir,
        prompts_dir=prompts_dir,
        schema_dir=schema_dir,
        log_dir=log_dir,
        deterministic_logs=args.deterministic_ids,
        max_input_bytes=args.max_input_bytes,
        max_output_bytes=args.max_output_bytes,
        timeout_ms=args.timeout_ms,
    )

    if args.once:
        await _run_once(service, deterministic_ids=args.deterministic_ids)
        return

    if not args.http and not args.stdio:
        args.http = True

    tasks: list[asyncio.Task[Any]] = []
    http_server: Any | None = None
    http_task: asyncio.Task[Any] | None = None
    if args.http:
        if uvicorn is None:
            msg = "uvicorn is required for HTTP transport but is not installed"
            raise RuntimeError(msg)
        config = uvicorn.Config(
            create_app(service, deterministic_ids=args.deterministic_ids),
            host=args.host,
            port=args.port,
            log_level=_UVICORN_LOG_LEVELS[args.log_level],
            access_log=False,
            limit_concurrency=args.max_connections,
            timeout_graceful_shutdown=args.shutdown_grace,
        )
        http_server = uvicorn.Server(config)
        if not args.stdio:
            assert http_server is not None
            await http_server.serve()
            return
        assert http_server is not None
        http_task = asyncio.create_task(http_server.serve())
        tasks.append(http_task)

    if args.stdio:
        stdio_server = JsonRpcStdioServer(service, deterministic_ids=args.deterministic_ids)
        tasks.append(asyncio.create_task(_stdio_loop(stdio_server, once=False)))

    if tasks:
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        finally:
            for task in tasks:
                if task is http_task and http_server is not None:
                    cast(Any, http_server).should_exit = True
                if not task.done():
                    task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task


async def _run_once(service: McpService, *, deterministic_ids: bool) -> None:
    discover_context = RequestContext(
        transport="http",
        route="discover",
        method="mcp.discover",
        deterministic_ids=deterministic_ids,
    )
    service.discover(discover_context)

    prompt_context = RequestContext(
        transport="http",
        route="prompt",
        method="mcp.prompt.get",
        deterministic_ids=deterministic_ids,
    )
    service.get_prompt("core.generic.bootstrap@1", prompt_context)

    tool_context = RequestContext(
        transport="http",
        route="tool",
        method="mcp.tool.invoke",
        deterministic_ids=deterministic_ids,
    )
    fixture_path = Path("tests/fixtures/mcp/docs/sample_article.md")
    service.invoke_tool(
        tool_id="mcp.tool:docs.load.fetch",
        arguments={"path": str(fixture_path)},
        context=tool_context,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    try:
        asyncio.run(_run_server(args))
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
