from __future__ import annotations

from apps.mcp_server.cli import create_parser


def test_cli_parser_supports_transports_flags() -> None:
    parser = create_parser()
    namespace = parser.parse_args(["--http", "--host", "0.0.0.0", "--port", "9000"])
    assert namespace.http is True
    assert namespace.stdio is False
    assert namespace.host == "0.0.0.0"
    assert namespace.port == 9000
