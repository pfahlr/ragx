"""Command line interface for the MCP server."""

from __future__ import annotations

import argparse
import threading
from collections.abc import Sequence

import uvicorn

from apps.mcp_server.runtime import McpService, McpStdIoServer, create_http_app

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mcp-server", description="Run the RAGX MCP server")
    parser.add_argument("--http", action="store_true", help="Run HTTP transport.")
    parser.add_argument("--stdio", action="store_true", help="Run STDIO JSON-RPC transport.")
    parser.add_argument("--host", default="127.0.0.1", help="Host for HTTP transport.")
    parser.add_argument("--port", type=int, default=3333, help="Port for HTTP transport.")
    return parser


def _run_http(service: McpService, *, host: str, port: int) -> None:
    app = create_http_app(service)
    uvicorn.run(app, host=host, port=port, log_level="info")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.http and not args.stdio:
        args.http = True

    service = McpService()

    http_thread: threading.Thread | None = None
    if args.http:
        http_thread = threading.Thread(
            target=_run_http,
            args=(service,),
            kwargs={"host": args.host, "port": args.port},
            daemon=args.stdio,
        )
        http_thread.start()

    if args.stdio:
        server = McpStdIoServer(service=service)
        server.serve_forever()

    if http_thread and not args.stdio:
        http_thread.join()

    return 0


if __name__ == "__main__":  # pragma: no cover - manual execution path
    raise SystemExit(main())
