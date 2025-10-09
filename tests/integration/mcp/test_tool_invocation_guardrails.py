from __future__ import annotations

import types
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("pydantic")

from apps.mcp_server.service.mcp_service import McpService, RequestContext

SCHEMA_DIR = Path("apps/mcp_server/schemas/mcp")
TOOLPACKS_DIR = Path("apps/mcp_server/toolpacks")
PROMPTS_DIR = Path("apps/mcp_server/prompts")


@pytest.fixture(autouse=True)
def _seed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGX_SEED", "7")


@pytest.fixture
def limited_service(tmp_path: Path) -> McpService:
    return McpService.create(
        toolpacks_dir=TOOLPACKS_DIR,
        prompts_dir=PROMPTS_DIR,
        schema_dir=SCHEMA_DIR,
        log_dir=tmp_path / "runs",
        schema_version="0.1.0",
        deterministic_logs=True,
        max_input_bytes=128,
        max_output_bytes=128,
        timeout_ms=25,
    )


def _context() -> RequestContext:
    return RequestContext(
        transport="http",
        route="tool",
        method="mcp.tool.invoke",
        deterministic_ids=True,
    )


def test_invoke_tool_rejects_oversized_input(limited_service: McpService) -> None:
    arguments = {
        "title": "Telemetry",
        "template": "{{ title }}" + "x" * 512,
    }
    envelope = limited_service.invoke_tool(
        tool_id="mcp.tool:exports.render.markdown",
        arguments=arguments,
        context=_context(),
    )

    assert envelope.ok is False
    assert envelope.error is not None
    assert envelope.error.code == "INVALID_INPUT"
    meta = envelope.model_dump(by_alias=True)["meta"]
    assert meta["execution"]["inputBytes"] > 128
    assert meta["execution"]["outputBytes"] == 0
    assert meta["idempotency"]["cacheHit"] is False


def test_invoke_tool_clamps_oversized_output(
    limited_service: McpService, monkeypatch: pytest.MonkeyPatch
) -> None:
    stats = types.SimpleNamespace(
        duration_ms=4.0,
        input_bytes=24,
        output_bytes=512,
        cache_hit=False,
        cache_key="stub",
    )

    class StubExecutor:
        def run_toolpack(self, toolpack, payload):  # type: ignore[no-untyped-def]
            return {"ok": True}

        def last_run_stats(self):  # type: ignore[no-untyped-def]
            return stats

    monkeypatch.setattr(limited_service, "_executor", StubExecutor())

    envelope = limited_service.invoke_tool(
        tool_id="mcp.tool:docs.load.fetch",
        arguments={"path": "ignored"},
        context=_context(),
    )

    assert envelope.ok is False
    assert envelope.error is not None
    assert envelope.error.code == "INVALID_OUTPUT"
    meta = envelope.model_dump(by_alias=True)["meta"]
    assert meta["execution"]["outputBytes"] == stats.output_bytes
    assert meta["idempotency"]["cacheHit"] is False


def test_invoke_tool_flags_timeouts(
    limited_service: McpService, monkeypatch: pytest.MonkeyPatch
) -> None:
    stats = types.SimpleNamespace(
        duration_ms=128.0,
        input_bytes=24,
        output_bytes=16,
        cache_hit=False,
        cache_key="slow",
    )

    class SlowExecutor:
        def run_toolpack(self, toolpack, payload):  # type: ignore[no-untyped-def]
            return {"ok": True}

        def last_run_stats(self):  # type: ignore[no-untyped-def]
            return stats

    monkeypatch.setattr(limited_service, "_executor", SlowExecutor())

    envelope = limited_service.invoke_tool(
        tool_id="mcp.tool:docs.load.fetch",
        arguments={"path": "ignored"},
        context=_context(),
    )

    assert envelope.ok is False
    assert envelope.error is not None
    assert envelope.error.code == "TIMEOUT"
    meta = envelope.model_dump(by_alias=True)["meta"]
    assert meta["execution"]["durationMs"] == pytest.approx(stats.duration_ms)
    assert meta["idempotency"]["cacheHit"] is False
