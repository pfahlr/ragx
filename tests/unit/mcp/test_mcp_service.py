from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator, ValidationError

pytest.importorskip("pydantic")

from apps.mcp_server.service.mcp_service import McpService, RequestContext, ValidationMode
from apps.toolpacks.executor import ToolpackExecutionError

SCHEMA_DIR = Path("apps/mcp_server/schemas/mcp")
TOOLPACKS_DIR = Path("apps/mcp_server/toolpacks")
PROMPTS_DIR = Path("apps/mcp_server/prompts")


@pytest.fixture(autouse=True)
def _set_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGX_SEED", "42")


def _validator(name: str) -> Draft202012Validator:
    schema_path = SCHEMA_DIR / name
    if not schema_path.exists():
        pytest.fail(f"Missing schema: {schema_path}")
    with schema_path.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    validator = Draft202012Validator(schema)
    validator.check_schema(schema)
    return validator


class _NullLogger:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self._counter = 0

    def next_step_id(self) -> int:
        self._counter += 1
        return self._counter

    def emit(self, payload: dict[str, Any]) -> None:
        self.records.append(payload)


@pytest.fixture
def service(tmp_path: Path) -> McpService:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    return McpService.create(
        toolpacks_dir=TOOLPACKS_DIR,
        prompts_dir=PROMPTS_DIR,
        schema_dir=SCHEMA_DIR,
        log_dir=logs_dir,
        logger=_NullLogger(),
        schema_version="0.1.0",
    )


def _context(route: str, *, deterministic: bool = True) -> RequestContext:
    return RequestContext(
        transport="http",
        route=route,
        method={
            "discover": "mcp.discover",
            "prompt": "mcp.prompt.get",
            "tool": "mcp.tool.invoke",
            "health": "mcp.health",
        }[route],
        start_time=time.monotonic(),
        deterministic_ids=deterministic,
    )


def test_discover_returns_tools_and_prompts(service: McpService) -> None:
    context = _context("discover")
    envelope = service.discover(context)
    payload = envelope.model_dump(by_alias=True)
    _validator("discover.response.schema.json").validate(payload["data"])
    assert payload["ok"] is True
    tool_ids = [tool["id"] for tool in payload["data"]["tools"]]
    assert "mcp.tool:vector.query.search" in tool_ids
    prompt_ids = [prompt["id"] for prompt in payload["data"]["prompts"]]
    assert any(identifier.startswith("core.generic") for identifier in prompt_ids)


def test_get_prompt_returns_prompt_payload(service: McpService) -> None:
    context = _context("prompt")
    envelope = service.get_prompt("core.generic.bootstrap@1", context)
    payload = envelope.model_dump(by_alias=True)
    _validator("prompt.response.schema.json").validate(payload["data"])
    assert payload["ok"] is True
    assert payload["data"]["id"] == "core.generic.bootstrap@1"
    assert payload["data"]["messages"], "expected prompt to contain messages"


def test_get_prompt_unknown_returns_not_found(service: McpService) -> None:
    context = _context("prompt")
    envelope = service.get_prompt("missing.prompt@1", context)
    payload = envelope.model_dump(by_alias=True)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "NOT_FOUND"
    assert "not" in payload["error"]["message"].lower()


def test_invoke_tool_executes_and_validates(service: McpService, tmp_path: Path) -> None:
    fixture_path = Path("tests/fixtures/mcp/docs/sample_article.md")
    assert fixture_path.exists(), "fixture missing"
    context = _context("tool")
    envelope = service.invoke_tool(
        tool_id="mcp.tool:docs.load.fetch",
        arguments={"path": str(fixture_path)},
        context=context,
    )
    payload = envelope.model_dump(by_alias=True)
    _validator("tool.response.schema.json").validate(payload["data"])
    assert payload["ok"] is True
    assert payload["data"]["result"]["document"]["sha256"].isalnum()


def test_invoke_tool_invalid_payload_returns_error(service: McpService) -> None:
    context = _context("tool")
    envelope = service.invoke_tool(
        tool_id="mcp.tool:docs.load.fetch",
        arguments={"encoding": "utf-8"},
        context=context,
    )
    payload = envelope.model_dump(by_alias=True)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "INVALID_INPUT"
    assert "path" in payload["error"]["message"].lower()


def test_invoke_tool_unknown_returns_not_found(service: McpService) -> None:
    context = _context("tool")
    envelope = service.invoke_tool(
        tool_id="mcp.tool:missing.tool", arguments={}, context=context
    )
    payload = envelope.model_dump(by_alias=True)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "NOT_FOUND"
    assert "tool" in payload["error"]["message"].lower()


def test_invoke_tool_execution_error_returns_internal_error(
    service: McpService, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context("tool")

    def _raise(*_: Any, **__: Any) -> None:
        raise ToolpackExecutionError("boom")

    monkeypatch.setattr(service._executor, "run_toolpack", _raise)
    fixture_path = Path("tests/fixtures/mcp/docs/sample_article.md")
    envelope = service.invoke_tool(
        tool_id="mcp.tool:docs.load.fetch",
        arguments={"path": str(fixture_path)},
        context=context,
    )
    payload = envelope.model_dump(by_alias=True)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "INTERNAL_ERROR"
    assert "boom" in payload["error"]["message"].lower()


def test_enforce_validation_mode_raises_on_invalid_envelope(
    service: McpService,
) -> None:
    """Enforce mode should raise when envelope validation fails."""

    class _FailingValidator:
        def __init__(self) -> None:
            self.calls = 0

        def validate(self, instance: Any) -> None:  # pragma: no cover - invoked in test
            self.calls += 1
            raise ValidationError("envelope invalid")

    failing_validator = _FailingValidator()

    class _RegistryStub:
        def load_envelope(self) -> _FailingValidator:  # pragma: no cover - invoked in test
            return failing_validator

        def load_tool_io(self, tool_id: str) -> Any:  # pragma: no cover - defensive
            raise AssertionError("tool validators should not be requested")

    service._validation_mode = ValidationMode.ENFORCE
    service._validation_registry = _RegistryStub()

    with pytest.raises(ValidationError):
        service.discover(_context("discover"))

    assert failing_validator.calls == 1
