from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

pytest.importorskip("fastapi")
pytest.importorskip("pydantic")

from fastapi.testclient import TestClient

from apps.mcp_server.http.main import create_app
from apps.mcp_server.service.mcp_service import McpService

SCHEMA_DIR = Path("apps/mcp_server/schemas/mcp")
TOOLPACKS_DIR = Path("apps/mcp_server/toolpacks")
PROMPTS_DIR = Path("apps/mcp_server/prompts")


def _validator(name: str) -> Draft202012Validator:
    schema_path = SCHEMA_DIR / name
    with schema_path.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    validator = Draft202012Validator(schema)
    validator.check_schema(schema)
    return validator


@pytest.fixture
def service(tmp_path: Path) -> McpService:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    return McpService.create(
        toolpacks_dir=TOOLPACKS_DIR,
        prompts_dir=PROMPTS_DIR,
        schema_dir=SCHEMA_DIR,
        log_dir=runs_dir,
        schema_version="0.1.0",
    )


def test_invalid_tool_payload_returns_invalid_input(service: McpService) -> None:
    client = TestClient(create_app(service))
    response = client.post(
        "/mcp/tool/mcp.tool:docs.load.fetch",
        json={"arguments": {"encoding": "utf-8"}},
    )
    assert response.status_code == 200
    payload = response.json()
    _validator("envelope.schema.json").validate(payload)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "INVALID_INPUT"
