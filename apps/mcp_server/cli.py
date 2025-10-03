"""Command line entry point for the MCP server bootstrap."""

from __future__ import annotations

import argparse
import threading
from collections.abc import Sequence

import uvicorn

from .http import create_app
from .service import McpService
from .stdio import McpStdioTransport


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mcp-server")
    parser.add_argument("--http", action="store_true", help="Run HTTP transport.")
    parser.add_argument("--stdio", action="store_true", help="Run STDIO JSON-RPC transport.")
    parser.add_argument("--host", default="127.0.0.1", help="Host for HTTP transport.")
    parser.add_argument("--port", type=int, default=3333, help="Port for HTTP transport.")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.http and not args.stdio:
        parser.error("At least one transport must be enabled via --http or --stdio.")
    return args


def _run_stdio(service: McpService) -> None:
    transport = McpStdioTransport(service)
    transport.serve_forever()


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    service = McpService()

    stdio_thread: threading.Thread | None = None
    if args.stdio:
        stdio_thread = threading.Thread(target=_run_stdio, args=(service,), daemon=True)
        stdio_thread.start()

    if args.http:
        app = create_app(service)
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    elif stdio_thread:
        stdio_thread.join()


if __name__ == "__main__":
    main()
