from __future__ import annotations

import argparse
from typing import Sequence

from apps.mcp_server.http_app import create_http_app
from apps.mcp_server.service import McpService
from apps.mcp_server.stdio import StdIoServer

try:  # pragma: no cover - uvicorn optional during tests
    import uvicorn
except Exception:  # pragma: no cover
    uvicorn = None  # type: ignore[assignment]

__all__ = ["create_parser", "main"]


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mcp-server")
    parser.add_argument("--http", action="store_true", help="Run HTTP transport.")
    parser.add_argument("--stdio", action="store_true", help="Run STDIO JSON-RPC transport.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3333)
    parser.add_argument("--max-connections", type=int, default=256)
    parser.add_argument("--shutdown-grace", type=int, default=10)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    service = McpService()

    ran = False
    if args.http:
        if uvicorn is None:  # pragma: no cover - environment guard
            raise RuntimeError("uvicorn is required for --http")
        app = create_http_app(service)
        config = uvicorn.Config(
            app,
            host=args.host,
            port=args.port,
            limit_max_requests=args.max_connections,
            timeout_graceful_shutdown=args.shutdown_grace,
            log_config=None,
        )
        server = uvicorn.Server(config)
        server.run()
        ran = True

    if args.stdio:
        StdIoServer(service).serve_forever()
        ran = True

    if not ran:
        parser.error("At least one transport (--http or --stdio) must be selected.")

    return 0
