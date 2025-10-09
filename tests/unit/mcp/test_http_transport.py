from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from apps.mcp_server.http.main import create_app
from apps.mcp_server.service.mcp_service import McpService

SCHEMA_DIR = Path("apps/mcp_server/schemas/mcp")
TOOLPACKS_DIR = Path("apps/mcp_server/toolpacks")
PROMPTS_DIR = Path("apps/mcp_server/prompts")


@pytest.fixture(autouse=True)
def _seed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGX_SEED", "42")


@pytest.fixture
def service(tmp_path: Path) -> McpService:
    return McpService.create(
        toolpacks_dir=TOOLPACKS_DIR,
        prompts_dir=PROMPTS_DIR,
        schema_dir=SCHEMA_DIR,
        log_dir=tmp_path / "logs",
        schema_version="0.1.0",
    )


@pytest.fixture
def client(service: McpService) -> TestClient:
    app = create_app(service)
    return TestClient(app)


@pytest.fixture
def deterministic_client(service: McpService) -> TestClient:
    app = create_app(service, deterministic_ids=True)
    return TestClient(app)


def test_http_discover_endpoint(client: TestClient) -> None:
    response = client.get("/mcp/discover")
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["data"]["tools"]


def test_http_prompt_endpoint(client: TestClient) -> None:
    response = client.get("/mcp/prompt/core.generic.bootstrap@1")
    assert response.status_code == 200
    payload = response.json()
    assert payload["data"]["id"] == "core.generic.bootstrap@1"


def test_http_prompt_not_found_returns_404(client: TestClient) -> None:
    response = client.get("/mcp/prompt/unknown.prompt@1")
    assert response.status_code == 404
    payload = response.json()
    assert payload["error"]["code"] == "NOT_FOUND"


def test_http_tool_endpoint(client: TestClient) -> None:
    fixture_path = Path("tests/fixtures/mcp/docs/sample_article.md")
    response = client.post(
        "/mcp/tool/mcp.tool:docs.load.fetch",
        json={"arguments": {"path": str(fixture_path)}},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["data"]["result"]["document"]["path"].endswith("sample_article.md")
    assert payload["meta"]["execution"]["inputBytes"] > 0
    assert payload["meta"]["execution"]["outputBytes"] > 0
    assert payload["meta"]["idempotency"]["cacheHit"] is False


def test_http_tool_not_found_returns_404(client: TestClient) -> None:
    response = client.post("/mcp/tool/mcp.tool:missing.tool", json={"arguments": {}})
    assert response.status_code == 404
    payload = response.json()
    assert payload["error"]["code"] == "NOT_FOUND"


def test_http_tool_invalid_payload_returns_400(client: TestClient) -> None:
    response = client.post(
        "/mcp/tool/mcp.tool:docs.load.fetch",
        json={"arguments": {"encoding": "utf-8"}},
    )
    assert response.status_code == 400
    payload = response.json()
    assert payload["error"]["code"] == "INVALID_INPUT"
    assert payload["meta"]["execution"]["inputBytes"] > 0
    assert payload["meta"]["execution"]["outputBytes"] == 0
    assert payload["meta"]["idempotency"]["cacheHit"] is False


def test_http_health_endpoint(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_http_deterministic_ids(deterministic_client: TestClient) -> None:
    first = deterministic_client.get("/mcp/discover")
    second = deterministic_client.get("/mcp/discover")

    assert first.status_code == 200
    assert second.status_code == 200

    first_meta = first.json()["meta"]
    second_meta = second.json()["meta"]

    assert first_meta["deterministic"] is True
    assert first_meta["execution"]["durationMs"] >= 0
    assert first_meta["requestId"] == second_meta["requestId"]
    assert first_meta["traceId"] == second_meta["traceId"]
    assert first_meta["spanId"] == second_meta["spanId"]
