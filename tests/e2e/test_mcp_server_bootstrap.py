from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from apps.mcp_server.app import create_app
from apps.mcp_server.service import McpService
from apps.mcp_server.transports.stdio import McpStdioServer


@pytest.fixture()
def service() -> McpService:
    return McpService()


@pytest.fixture()
def http_client(service: McpService) -> TestClient:
    app = create_app(service)
    return TestClient(app)


def _assert_envelope_structure(payload: dict[str, Any]) -> None:
    assert payload["ok"] is True
    assert "meta" in payload and isinstance(payload["meta"], dict)
    meta = payload["meta"]
    assert "traceId" in meta and isinstance(meta["traceId"], str) and meta["traceId"]
    assert "data" in payload
    assert "errors" in payload and isinstance(payload["errors"], list)


def test_http_discover_returns_envelope(http_client: TestClient) -> None:
    response = http_client.get("/mcp/discover")
    assert response.status_code == 200
    payload = response.json()
    _assert_envelope_structure(payload)
    assert payload["data"]["version"] == "0.1.0"
    assert payload["data"]["tools"] == []
    assert payload["data"]["prompts"] == []


def test_http_prompt_get_returns_placeholder(http_client: TestClient) -> None:
    response = http_client.get("/mcp/prompt/core/example/1")
    assert response.status_code == 200
    payload = response.json()
    _assert_envelope_structure(payload)
    prompt = payload["data"]
    assert prompt["domain"] == "core"
    assert prompt["name"] == "example"
    assert prompt["major"] == "1"
    assert isinstance(prompt["content"], str) and "placeholder" in prompt["content"].lower()


def test_http_tool_invoke_echoes_payload(http_client: TestClient) -> None:
    request_body = {"inputs": {"query": "hello"}}
    response = http_client.post("/mcp/tool/demo.tool", json=request_body)
    assert response.status_code == 200
    payload = response.json()
    _assert_envelope_structure(payload)
    data = payload["data"]
    assert data["tool"] == "demo.tool"
    assert data["inputs"] == request_body["inputs"]


def test_stdio_discover_returns_envelope(service: McpService) -> None:
    server = McpStdioServer(service)
    request = {"jsonrpc": "2.0", "id": "1", "method": "mcp.discover", "params": {}}
    response = server.handle_message(request)
    assert response["id"] == "1"
    assert response["jsonrpc"] == "2.0"
    result = response["result"]
    _assert_envelope_structure(result)
    assert result["data"]["version"] == "0.1.0"


def test_stdio_prompt_get_returns_envelope(service: McpService) -> None:
    server = McpStdioServer(service)
    request = {
        "jsonrpc": "2.0",
        "id": "2",
        "method": "mcp.prompt.get",
        "params": {"domain": "core", "name": "demo", "major": "1"},
    }
    response = server.handle_message(request)
    prompt = response["result"]["data"]
    assert prompt["domain"] == "core"
    assert prompt["name"] == "demo"
    assert prompt["major"] == "1"


def test_stdio_tool_invoke_returns_envelope(service: McpService) -> None:
    server = McpStdioServer(service)
    request = {
        "jsonrpc": "2.0",
        "id": "3",
        "method": "mcp.tool.invoke",
        "params": {"tool": "demo.tool", "arguments": {"foo": "bar"}},
    }
    response = server.handle_message(request)
    data = response["result"]["data"]
    assert data["tool"] == "demo.tool"
    assert data["inputs"] == {"foo": "bar"}
