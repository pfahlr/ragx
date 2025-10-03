"""Acceptance tests for MCP server bootstrap implementation."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from apps.mcp_server.runtime.http import create_http_app
from apps.mcp_server.runtime.service import McpService
from apps.mcp_server.runtime.stdio import JsonRpcRequest, McpStdIoServer


@pytest.fixture()
def mcp_service() -> McpService:
    return McpService()


def assert_envelope_structure(payload: dict[str, Any], *, tool: str) -> None:
    """Helper asserting the shape of an envelope response."""

    assert payload["ok"] is True
    assert isinstance(payload["data"], dict)
    assert "meta" in payload and isinstance(payload["meta"], dict)
    meta = payload["meta"]
    assert meta["tool"] == tool
    assert isinstance(meta["traceId"], str) and meta["traceId"]
    assert meta["version"] == "0.1.0"
    assert isinstance(meta["durationMs"], int)
    assert isinstance(meta["warnings"], list)
    assert payload["errors"] == []


def test_http_discover_returns_envelope(mcp_service: McpService) -> None:
    app = create_http_app(mcp_service)
    client = TestClient(app)
    response = client.get("/mcp/discover")
    assert response.status_code == 200
    payload = response.json()
    assert_envelope_structure(payload, tool="mcp.discover")
    assert payload["data"]["server"]["name"] == "ragx.mcp"


def test_stdio_discover_matches_http_envelope(mcp_service: McpService) -> None:
    server = McpStdIoServer(service=mcp_service)
    request = JsonRpcRequest(id="1", method="mcp.discover", params={})
    response = server.handle_request(request)
    assert response.error is None
    assert response.result is not None

    http_payload = TestClient(create_http_app(mcp_service)).get("/mcp/discover").json()
    assert response.result["ok"] is http_payload["ok"] is True
    assert response.result["data"] == http_payload["data"]
    assert response.result["errors"] == http_payload["errors"] == []
    assert response.result["meta"]["tool"] == http_payload["meta"]["tool"]
    assert response.result["meta"]["version"] == http_payload["meta"]["version"]
    assert response.result["meta"]["warnings"] == http_payload["meta"]["warnings"]


def test_prompt_endpoint_placeholder(mcp_service: McpService) -> None:
    app = create_http_app(mcp_service)
    client = TestClient(app)
    response = client.get("/mcp/prompt/test/domain/1")
    assert response.status_code == 200
    payload = response.json()
    assert_envelope_structure(payload, tool="mcp.prompt.get")
    assert payload["data"]["prompt"]["id"] == "test/domain"


def test_tool_endpoint_placeholder(mcp_service: McpService) -> None:
    app = create_http_app(mcp_service)
    client = TestClient(app)
    body = {"arguments": {"query": "ping"}}
    response = client.post("/mcp/tool/echo", json=body)
    assert response.status_code == 200
    payload = response.json()
    assert_envelope_structure(payload, tool="mcp.tool.invoke")
    assert payload["data"]["echo"]["tool"] == "echo"


def test_stdio_tool_invocation_matches_http(mcp_service: McpService) -> None:
    server = McpStdIoServer(service=mcp_service)
    request = JsonRpcRequest(
        id="2",
        method="mcp.tool.invoke",
        params={"tool": "echo", "arguments": {"value": 42}},
    )
    response = server.handle_request(request)
    assert response.error is None
    assert response.result is not None

    http_payload = TestClient(create_http_app(mcp_service)).post(
        "/mcp/tool/echo", json={"arguments": {"value": 42}}
    ).json()
    assert response.result["data"] == http_payload["data"]
    assert response.result["errors"] == http_payload["errors"] == []
    assert response.result["meta"]["tool"] == http_payload["meta"]["tool"]


def test_stdio_round_trip_serialisation(mcp_service: McpService) -> None:
    server = McpStdIoServer(service=mcp_service)
    request = JsonRpcRequest(id="42", method="mcp.discover", params={})
    serialised = server.dumps_response(server.handle_request(request))
    parsed = json.loads(serialised)
    assert parsed["id"] == "42"
    assert parsed["jsonrpc"] == "2.0"
    assert parsed["result"]["meta"]["tool"] == "mcp.discover"
