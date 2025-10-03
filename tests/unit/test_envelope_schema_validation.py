from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

pytest.importorskip("pydantic")

from apps.mcp_server.service.mcp_service import McpService, RequestContext

SCHEMA_DIR = Path("apps/mcp_server/schemas/mcp")
TOOLPACKS_DIR = Path("apps/mcp_server/toolpacks")
PROMPTS_DIR = Path("apps/mcp_server/prompts")


def _validator(name: str) -> Draft202012Validator:
    schema_path = SCHEMA_DIR / name
    if not schema_path.exists():
        pytest.fail(f"Schema not found: {schema_path}")
    with schema_path.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    validator = Draft202012Validator(schema)
    validator.check_schema(schema)
    return validator


@pytest.fixture
def service(tmp_path: Path) -> McpService:
    logs_dir = tmp_path / "runs"
    logs_dir.mkdir()
    return McpService.create(
        toolpacks_dir=TOOLPACKS_DIR,
        prompts_dir=PROMPTS_DIR,
        schema_dir=SCHEMA_DIR,
        log_dir=logs_dir,
        schema_version="0.1.0",
    )


def _context(route: str) -> RequestContext:
    return RequestContext(
        transport="http",
        route=route,
        method={
            "discover": "mcp.discover",
            "prompt": "mcp.prompt.get",
            "tool": "mcp.tool.invoke",
        }[route],
    )


def test_success_envelope_matches_schema(service: McpService) -> None:
    envelope = service.discover(_context("discover"))
    payload = envelope.model_dump(by_alias=True)
    _validator("envelope.schema.json").validate(payload)
    assert payload["ok"] is True


def test_error_envelope_uses_canonical_code(service: McpService) -> None:
    envelope = service.invoke_tool(
        tool_id="mcp.tool:docs.load.fetch",
        arguments={"encoding": "utf-8"},
        context=_context("tool"),
    )
    payload = envelope.model_dump(by_alias=True)
    _validator("envelope.schema.json").validate(payload)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "INVALID_INPUT"
