from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from apps.mcp_server.http_app import create_http_app
from apps.mcp_server.service import McpService
from apps.mcp_server.stdio import StdIoServer


@pytest.fixture()
def service() -> McpService:
    return McpService()


@pytest.fixture()
def http_client(service: McpService) -> TestClient:
    app = create_http_app(service)
    return TestClient(app)


def assert_envelope_shape(payload: dict[str, Any], *, transport: str) -> None:
    assert payload["ok"] is True
    assert isinstance(payload.get("data"), dict)
    meta = payload.get("meta")
    assert isinstance(meta, dict)
    assert meta.get("transport") == transport
    trace_id = meta.get("trace_id")
    assert isinstance(trace_id, str) and len(trace_id) >= 8
    errors = payload.get("errors")
    assert isinstance(errors, list)


def test_http_discover_returns_envelope(http_client: TestClient) -> None:
    response = http_client.get("/mcp/discover")
    assert response.status_code == 200
    payload = response.json()
    assert_envelope_shape(payload, transport="http")
    assert payload["data"]["service"] == "ragx.mcp"


def test_stdio_discover_matches_http(service: McpService, http_client: TestClient) -> None:
    http_payload = http_client.get("/mcp/discover").json()

    server = StdIoServer(service)
    request = {
        "jsonrpc": "2.0",
        "id": "req-1",
        "method": "mcp.discover",
        "params": {},
    }
    response_raw = server.handle_message(json.dumps(request))
    response = json.loads(response_raw)
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == "req-1"

    stdio_payload = response["result"]
    assert_envelope_shape(stdio_payload, transport="stdio")
    assert stdio_payload["data"] == http_payload["data"]
