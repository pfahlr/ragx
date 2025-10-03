"""End-to-end tests for the MCP server bootstrap skeleton."""

from __future__ import annotations

import io
import json
import logging
from typing import Any

import pytest
from fastapi.testclient import TestClient

from apps.mcp_server.cli import parse_args
from apps.mcp_server.http import create_app
from apps.mcp_server.service import McpService
from apps.mcp_server.stdio import McpStdioTransport


def _extract_single_log(caplog: pytest.LogCaptureFixture) -> dict[str, Any]:
    assert caplog.records, "expected a log record to be emitted"
    raw = caplog.records[-1].message
    return json.loads(raw)


def test_http_discover_endpoint_returns_envelope(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="ragx.mcp_server")
    service = McpService()
    app = create_app(service)
    client = TestClient(app)

    response = client.get("/mcp/discover")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["data"]["kind"] == "discovery"
    trace_id = payload["meta"].get("trace_id")
    assert isinstance(trace_id, str) and trace_id

    log_payload = _extract_single_log(caplog)
    assert log_payload["transport"] == "http"
    assert log_payload["trace_id"] == trace_id
    assert log_payload["route"] == "/mcp/discover"
    assert log_payload["status"] == "ok"


def test_stdio_discover_round_trip(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="ragx.mcp_server")
    service = McpService()
    request = {"jsonrpc": "2.0", "id": "req-1", "method": "mcp.discover", "params": {}}

    stdin = io.StringIO(json.dumps(request) + "\n")
    stdout = io.StringIO()

    transport = McpStdioTransport(service, reader=stdin, writer=stdout)
    processed = transport.handle_once()

    assert processed is True
    stdout.seek(0)
    response = json.loads(stdout.readline())
    result = response["result"]
    assert result["ok"] is True
    assert result["data"]["kind"] == "discovery"
    trace_id = result["meta"].get("trace_id")
    assert isinstance(trace_id, str) and trace_id

    log_payload = _extract_single_log(caplog)
    assert log_payload["transport"] == "stdio"
    assert log_payload["trace_id"] == trace_id
    assert log_payload["method"] == "mcp.discover"
    assert log_payload["status"] == "ok"


def test_cli_flag_parsing_defaults() -> None:
    args = parse_args(["--http", "--port", "7777"])
    assert args.http is True
    assert args.stdio is False
    assert args.host == "127.0.0.1"
    assert args.port == 7777


def test_cli_requires_transport() -> None:
    with pytest.raises(SystemExit):
        parse_args([])

