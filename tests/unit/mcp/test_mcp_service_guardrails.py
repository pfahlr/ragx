"""Unit tests for McpService guardrails and execution metadata."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pydantic")

from apps.mcp_server.service.mcp_service import (
    McpService,
    RequestContext,
    ServiceLimits,
    ValidationMode,
)
from apps.toolpacks.executor import ExecutionStats
from apps.toolpacks.loader import Toolpack


@dataclass
class _NoopValidator:
    def validate(self, _: Mapping[str, Any]) -> None:  # pragma: no cover - trivial
        return None


class _SchemaStoreStub:
    def validator(self, _: str) -> _NoopValidator:
        return _NoopValidator()


class _PromptRepoStub:
    def list(self) -> list[Any]:
        return []

    def get(self, prompt_id: str) -> Any:  # pragma: no cover - not used in tests
        raise KeyError(prompt_id)


class _ValidationRegistryStub:
    def load_tool_io(self, _: str) -> Any:
        return None

    def load_envelope(self) -> _NoopValidator:
        return _NoopValidator()


class _ValidationLogStub:
    def __init__(self) -> None:
        self._counter = 0
        self.events: list[Any] = []

    def next_step_id(self) -> int:
        self._counter += 1
        return self._counter

    def emit(self, event: Any) -> None:
        self.events.append(event)


class _ServerLogStub:
    def __init__(self) -> None:
        self._counter = 0
        self.records: list[Any] = []

    def next_step_id(self) -> int:
        self._counter += 1
        return self._counter

    def emit(self, payload: Any) -> None:
        self.records.append(payload)


class _QueueExecutor:
    def __init__(self) -> None:
        self._queue: list[tuple[dict[str, Any], ExecutionStats]] = []
        self._last_stats: ExecutionStats | None = None
        self.calls: list[tuple[Toolpack, Mapping[str, Any]]] = []

    def queue(self, result: Mapping[str, Any], stats: ExecutionStats) -> None:
        self._queue.append((dict(result), stats))

    def run_toolpack(
        self,
        toolpack: Toolpack,
        payload: Mapping[str, Any],
        *,
        cache_scope: str | None = None,
    ) -> Mapping[str, Any]:
        self.calls.append((toolpack, dict(payload)))
        if not self._queue:
            raise AssertionError("Executor queue exhausted")
        result, stats = self._queue.pop(0)
        self._last_stats = stats
        return result

    def last_run_stats(self) -> ExecutionStats | None:
        return self._last_stats


@pytest.fixture
def toolpack(tmp_path: Path) -> Toolpack:
    return Toolpack(
        id="tests.guardrail.tool",
        version="0.1.0",
        deterministic=True,
        timeout_ms=100,
        limits={"maxInputBytes": 64, "maxOutputBytes": 64},
        input_schema={"type": "object", "additionalProperties": True},
        output_schema={"type": "object", "additionalProperties": True},
        execution={"kind": "python", "module": "tests.helpers.toolpack_samples:echo"},
        caps={},
        env={},
        templating={},
        source_path=tmp_path / "guardrail.tool.yaml",
    )


def _service(
    executor: _QueueExecutor,
    toolpack: Toolpack,
    *,
    limits: ServiceLimits,
    validation_mode: ValidationMode = ValidationMode.OFF,
    validation_log: _ValidationLogStub | None = None,
) -> McpService:
    return McpService(
        toolpacks={toolpack.id: toolpack},
        executor=executor,
        prompts=_PromptRepoStub(),
        schema_store=_SchemaStoreStub(),
        log_manager=_ServerLogStub(),
        schema_version="0.1.0",
        validation_registry=_ValidationRegistryStub(),
        validation_log=validation_log or _ValidationLogStub(),
        validation_mode=validation_mode,
        limits=limits,
    )


def _context(*, deterministic: bool = True) -> RequestContext:
    return RequestContext(
        transport="http",
        route="tool",
        method="mcp.tool.invoke",
        deterministic_ids=deterministic,
    )


def test_invoke_tool_rejects_oversized_input(toolpack: Toolpack) -> None:
    executor = _QueueExecutor()
    service = _service(
        executor,
        toolpack,
        limits=ServiceLimits(max_input_bytes=16, max_output_bytes=256, timeout_ms=1000),
    )

    envelope = service.invoke_tool(
        tool_id=toolpack.id,
        arguments={"text": "x" * 50},
        context=_context(),
    )

    payload = envelope.model_dump(by_alias=True)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "INVALID_INPUT"
    execution = payload["meta"]["execution"]
    assert execution["inputBytes"] > 16
    assert execution["outputBytes"] == 0
    assert payload["meta"]["idempotency"]["cacheHit"] is False
    assert executor.calls == []


def test_invoke_tool_rejects_oversized_output(toolpack: Toolpack) -> None:
    executor = _QueueExecutor()
    stats = ExecutionStats(duration_ms=5.0, input_bytes=10, output_bytes=512, cache_hit=False)
    executor.queue(result={"echo": {"text": "data"}}, stats=stats)
    service = _service(
        executor,
        toolpack,
        limits=ServiceLimits(max_input_bytes=1024, max_output_bytes=128, timeout_ms=1000),
    )

    envelope = service.invoke_tool(
        tool_id=toolpack.id,
        arguments={"text": "data"},
        context=_context(),
    )

    payload = envelope.model_dump(by_alias=True)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "INVALID_OUTPUT"
    assert payload["meta"]["execution"]["outputBytes"] == stats.output_bytes


def test_invoke_tool_times_out(toolpack: Toolpack) -> None:
    executor = _QueueExecutor()
    stats = ExecutionStats(duration_ms=2500.0, input_bytes=10, output_bytes=10, cache_hit=False)
    executor.queue(result={"echo": {"text": "data"}}, stats=stats)
    service = _service(
        executor,
        toolpack,
        limits=ServiceLimits(max_input_bytes=1024, max_output_bytes=1024, timeout_ms=1000),
    )

    envelope = service.invoke_tool(
        tool_id=toolpack.id,
        arguments={"text": "data"},
        context=_context(),
    )

    payload = envelope.model_dump(by_alias=True)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "TIMEOUT"
    assert payload["meta"]["execution"]["durationMs"] == pytest.approx(stats.duration_ms)


def test_success_populates_execution_and_idempotency(toolpack: Toolpack) -> None:
    executor = _QueueExecutor()
    first_stats = ExecutionStats(
        duration_ms=12.5,
        input_bytes=24,
        output_bytes=40,
        cache_hit=False,
    )
    second_stats = ExecutionStats(
        duration_ms=3.2,
        input_bytes=24,
        output_bytes=40,
        cache_hit=True,
    )
    executor.queue(result={"echo": {"text": "hello"}}, stats=first_stats)
    executor.queue(result={"echo": {"text": "hello"}}, stats=second_stats)
    service = _service(
        executor,
        toolpack,
        limits=ServiceLimits(max_input_bytes=1024, max_output_bytes=1024, timeout_ms=1000),
    )

    first_envelope = service.invoke_tool(
        tool_id=toolpack.id,
        arguments={"text": "hello"},
        context=_context(),
    )
    first_payload = first_envelope.model_dump(by_alias=True)
    assert first_payload["ok"] is True
    assert first_payload["meta"]["execution"]["inputBytes"] == first_stats.input_bytes
    assert first_payload["meta"]["execution"]["outputBytes"] == first_stats.output_bytes
    assert first_payload["meta"]["idempotency"]["cacheHit"] is False

    second_envelope = service.invoke_tool(
        tool_id=toolpack.id,
        arguments={"text": "hello"},
        context=_context(),
    )
    second_payload = second_envelope.model_dump(by_alias=True)
    assert second_payload["ok"] is True
    assert second_payload["meta"]["idempotency"]["cacheHit"] is True
    assert second_payload["meta"]["execution"]["outputBytes"] == second_stats.output_bytes


def test_validation_logs_output_bytes_ignore_duration(
    toolpack: Toolpack, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor = _QueueExecutor()
    log_stub = _ValidationLogStub()
    service = _service(
        executor,
        toolpack,
        limits=ServiceLimits(max_input_bytes=1024, max_output_bytes=1024, timeout_ms=1000),
        validation_mode=ValidationMode.SHADOW,
        validation_log=log_stub,
    )

    durations = iter([12.345, 67.89])

    def _next_duration(_: float) -> float:
        return next(durations)

    monkeypatch.setattr(
        "apps.mcp_server.service.mcp_service._duration_ms", _next_duration
    )

    service.discover(context=_context())
    service.discover(context=_context())

    assert len(log_stub.events) == 2
    first, second = log_stub.events
    assert first.execution["outputBytes"] == second.execution["outputBytes"]
    assert first.execution["outputBytes"] > 0
