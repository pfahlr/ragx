from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("pydantic")

from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator

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
        log_dir=tmp_path / "runs",
        schema_version="0.1.0",
    )


def _validator(name: str) -> Draft202012Validator:
    schema_path = SCHEMA_DIR / name
    with schema_path.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    validator = Draft202012Validator(schema)
    validator.check_schema(schema)
    return validator


def test_http_endpoints_validate_against_schemas(service: McpService) -> None:
    client = TestClient(create_app(service))

    discover = client.get("/mcp/discover")
    assert discover.status_code == 200
    payload = discover.json()
    _validator("discover.response.schema.json").validate(payload["data"])

    prompt = client.get("/mcp/prompt/core.generic.bootstrap@1")
    assert prompt.status_code == 200
    prompt_payload = prompt.json()
    _validator("prompt.response.schema.json").validate(prompt_payload["data"])

    fixture_path = Path("tests/fixtures/mcp/docs/sample_article.md")
    tool = client.post(
        "/mcp/tool/mcp.tool:docs.load.fetch",
        json={"arguments": {"path": str(fixture_path)}},
    )
    assert tool.status_code == 200
    tool_payload = tool.json()
    _validator("tool.response.schema.json").validate(tool_payload["data"])

    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
