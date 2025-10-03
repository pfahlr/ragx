from __future__ import annotations

import argparse
import logging
import threading
from collections.abc import Sequence

import uvicorn

from apps.mcp_server.app import create_app
from apps.mcp_server.service import McpService
from apps.mcp_server.transports import McpStdioServer

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mcp-server", description="RAGX MCP server")
    parser.add_argument("--http", action="store_true", help="Run HTTP transport.")
    parser.add_argument("--stdio", action="store_true", help="Run STDIO JSON-RPC transport.")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind host.")
    parser.add_argument("--port", type=int, default=3333, help="HTTP bind port.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.http and not args.stdio:
        args.http = True

    service = McpService()

    stdio_thread: threading.Thread | None = None
    if args.stdio:
        stdio_server = McpStdioServer(service)
        stdio_thread = threading.Thread(target=stdio_server.serve_forever, daemon=True)
        stdio_thread.start()
        LOGGER.info("STDIO transport started")

    if args.http:
        app = create_app(service)
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    else:
        # If only STDIO was requested, block until interrupted.
        if stdio_thread is not None:
            stdio_thread.join()

    return 0


if __name__ == "__main__":  # pragma: no cover - manual execution entry point
    raise SystemExit(main())
