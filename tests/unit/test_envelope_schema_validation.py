from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

pytest.importorskip("pydantic")

from apps.mcp_server.service.envelope import Envelope, EnvelopeMeta
from apps.mcp_server.service.mcp_service import McpService, RequestContext

SCHEMA_DIR = Path("apps/mcp_server/schemas/mcp")
TOOLPACKS_DIR = Path("apps/mcp_server/toolpacks")
PROMPTS_DIR = Path("apps/mcp_server/prompts")


@pytest.fixture(autouse=True)
def _seed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGX_SEED", "41")


@pytest.fixture
def envelope_validator() -> Draft202012Validator:
    schema_path = SCHEMA_DIR / "envelope.schema.json"
    if not schema_path.exists():
        pytest.fail(f"Missing envelope schema: {schema_path}")
    with schema_path.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    validator = Draft202012Validator(schema)
    validator.check_schema(schema)
    return validator


@pytest.fixture
def service(tmp_path: Path) -> McpService:
    return McpService.create(
        toolpacks_dir=TOOLPACKS_DIR,
        prompts_dir=PROMPTS_DIR,
        schema_dir=SCHEMA_DIR,
        log_dir=tmp_path / "logs",
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
        deterministic_ids=True,
    )


def test_envelope_schema_rejects_negative_duration(envelope_validator: Draft202012Validator) -> None:
    envelope = Envelope.success(
        data={"status": "ok"},
        meta=EnvelopeMeta(
            request_id="req-123",
            trace_id="trace-456",
            span_id="span-789",
            schema_version="0.1.0",
            deterministic=True,
            transport="http",
            route="tool",
            method="mcp.tool.invoke",
            duration_ms=-1.0,
            status="ok",
            attempt=0,
            input_bytes=0,
            output_bytes=0,
        ),
    )
    payload = envelope.model_dump(by_alias=True)
    with pytest.raises(ValidationError):
        envelope_validator.validate(payload)


def test_tool_output_schema_violation_returns_internal_error(
    service: McpService, monkeypatch: pytest.MonkeyPatch
) -> None:
    module_path = "apps.toolpacks.python.core.docs.load_fetch"
    module = pytest.importorskip(module_path)

    def _invalid_run(payload: Mapping[str, Any]) -> dict[str, Any]:
        return {"metadata": {}, "stats": {"bytes": 0}}

    monkeypatch.setattr(module, "run", _invalid_run)
    # Ensure deterministic cache does not return a previously materialised value.
    service._executor._cache.clear()  # type: ignore[attr-defined]

    fixture_path = Path("tests/fixtures/mcp/docs/sample_article.md")
    context = _context("tool")
    envelope = service.invoke_tool(
        tool_id="mcp.tool:docs.load.fetch",
        arguments={"path": str(fixture_path)},
        context=context,
    )

    payload = envelope.model_dump(by_alias=True)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "INTERNAL"
    assert payload["error"]["details"]["stage"] == "output"
    assert "document" in payload["error"]["message"].lower()
    assert payload["meta"]["status"] == "error"
