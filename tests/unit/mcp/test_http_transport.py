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


def test_http_health_endpoint(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
