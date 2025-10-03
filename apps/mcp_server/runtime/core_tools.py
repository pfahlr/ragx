from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import count
from typing import Any
from uuid import uuid4

from apps.mcp_server.logging import JsonLogWriter, McpLogEvent
from apps.toolpacks.executor import Executor, ToolpackExecutionError
from apps.toolpacks.loader import Toolpack

__all__ = ["CoreToolsRuntime"]


@dataclass(slots=True)
class _InvocationContext:
    toolpack: Toolpack
    tool_id: str
    attempt: int
    step_id: int
    trace_id: str
    span_id: str
    run_id: str
    attempt_id: str


class CoreToolsRuntime:
    """Execute core MCP toolpacks with structured logging."""

    def __init__(
        self,
        *,
        toolpacks: Iterable[Toolpack],
        executor: Executor,
        log_writer: JsonLogWriter,
    ) -> None:
        self._toolpacks: dict[str, Toolpack] = {}
        for pack in toolpacks:
            self._toolpacks[pack.id] = pack
            self._toolpacks[f"mcp.tool:{pack.id}"] = pack
        self._executor = executor
        self._log_writer = log_writer
        self._step_counter = count(1)

    def invoke(self, tool_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        toolpack = self._toolpacks.get(tool_id)
        if toolpack is None:
            raise KeyError(f"Unknown tool id: {tool_id}")

        context = self._build_context(toolpack, tool_id)
        start_time = time.perf_counter()
        input_bytes = len(json.dumps(payload, ensure_ascii=False))

        self._log_writer.write(
            McpLogEvent(
                ts=datetime.now(UTC),
                agent_id="mcp_server",
                task_id="06ab_core_tools_minimal_subset",
                step_id=context.step_id,
                trace_id=context.trace_id,
                span_id=context.span_id,
                tool_id=tool_id,
                event="invoke.start",
                status="in_progress",
                duration_ms=0.0,
                attempt=context.attempt,
                input_bytes=input_bytes,
                output_bytes=0,
                metadata=self._metadata(toolpack, context),
                run_id=context.run_id,
                attempt_id=context.attempt_id,
            )
        )

        try:
            result = self._executor.run_toolpack(toolpack, payload)
        except ToolpackExecutionError as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000
            error_payload = {
                "code": self._error_code(exc),
                "message": str(exc),
            }
            self._log_writer.write(
                McpLogEvent(
                    ts=datetime.now(UTC),
                    agent_id="mcp_server",
                    task_id="06ab_core_tools_minimal_subset",
                    step_id=context.step_id,
                    trace_id=context.trace_id,
                    span_id=context.span_id,
                    tool_id=tool_id,
                    event="invoke.failure",
                    status="error",
                    duration_ms=duration_ms,
                    attempt=context.attempt,
                    input_bytes=input_bytes,
                    output_bytes=0,
                    metadata=self._metadata(toolpack, context),
                    error=error_payload,
                    run_id=context.run_id,
                    attempt_id=context.attempt_id,
                )
            )
            raise

        duration_ms = (time.perf_counter() - start_time) * 1000
        output_bytes = len(json.dumps(result, ensure_ascii=False))
        self._log_writer.write(
            McpLogEvent(
                ts=datetime.now(UTC),
                agent_id="mcp_server",
                task_id="06ab_core_tools_minimal_subset",
                step_id=context.step_id,
                trace_id=context.trace_id,
                span_id=context.span_id,
                tool_id=tool_id,
                event="invoke.success",
                status="success",
                duration_ms=duration_ms,
                attempt=context.attempt,
                input_bytes=input_bytes,
                output_bytes=output_bytes,
                metadata=self._metadata(toolpack, context),
                run_id=context.run_id,
                attempt_id=context.attempt_id,
            )
        )
        return result

    def _build_context(self, toolpack: Toolpack, tool_id: str) -> _InvocationContext:
        step_id = next(self._step_counter)
        return _InvocationContext(
            toolpack=toolpack,
            tool_id=tool_id,
            attempt=1,
            step_id=step_id,
            trace_id=str(uuid4()),
            span_id=str(uuid4()),
            run_id=str(uuid4()),
            attempt_id=str(uuid4()),
        )

    def _metadata(self, toolpack: Toolpack, context: _InvocationContext) -> dict[str, Any]:
        return {
            "toolpack_version": toolpack.version,
            "deterministic": toolpack.deterministic,
            "timeout_ms": toolpack.timeout_ms,
            "limits": dict(toolpack.limits),
        }

    def _error_code(self, exc: ToolpackExecutionError) -> str:
        message = str(exc).lower()
        if "schema" in message:
            return "validation_error"
        return "execution_error"
