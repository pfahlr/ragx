"""Runtime helper to invoke core MCP toolpacks with structured logging."""
from __future__ import annotations

import json
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from apps.mcp_server.logging import JsonLogWriter, McpLogEvent
from apps.toolpacks.executor import Executor, ToolpackExecutionError
from apps.toolpacks.loader import Toolpack


class CoreToolsRuntime:
    """Invoke Toolpack-backed MCP tools with structured logging."""

    def __init__(
        self,
        *,
        toolpacks: list[Toolpack],
        executor: Executor,
        log_writer: JsonLogWriter,
    ) -> None:
        self._toolpacks: dict[str, Toolpack] = {}
        for toolpack in toolpacks:
            self._toolpacks[toolpack.id] = toolpack
            self._toolpacks[f"mcp.tool:{toolpack.id}"] = toolpack
        self._executor = executor
        self._log_writer = log_writer
        self._step_counter = 0

    def invoke(self, tool_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        toolpack = self._toolpacks.get(tool_id)
        if toolpack is None:
            raise KeyError(f"Unknown tool id: {tool_id}")

        self._step_counter += 1
        step_id = self._step_counter
        attempt = 1
        trace_id = str(uuid4())
        span_id = str(uuid4())
        start_time = time.perf_counter()
        input_bytes = len(json.dumps(payload, sort_keys=True))

        self._log_writer.write(
            McpLogEvent(
                ts=datetime.now(tz=UTC),
                agent_id=self._log_writer.agent_id,
                task_id=self._log_writer.task_id,
                step_id=step_id,
                trace_id=trace_id,
                span_id=span_id,
                tool_id=tool_id,
                event="invocation_start",
                status="running",
                duration_ms=0.0,
                attempt=attempt,
                input_bytes=input_bytes,
                output_bytes=0,
                metadata={
                    "toolpack_version": toolpack.version,
                    "deterministic": toolpack.deterministic,
                    "schema_version": "0.1.0",
                },
                error=None,
            )
        )

        try:
            result = self._executor.run_toolpack(toolpack, payload)
        except ToolpackExecutionError as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self._log_writer.write(
                McpLogEvent(
                    ts=datetime.now(tz=UTC),
                    agent_id=self._log_writer.agent_id,
                    task_id=self._log_writer.task_id,
                    step_id=step_id,
                    trace_id=trace_id,
                    span_id=span_id,
                    tool_id=tool_id,
                    event="invocation_failure",
                    status="error",
                    duration_ms=duration_ms,
                    attempt=attempt,
                    input_bytes=input_bytes,
                    output_bytes=0,
                    metadata={
                        "toolpack_version": toolpack.version,
                        "deterministic": toolpack.deterministic,
                        "schema_version": "0.1.0",
                    },
                    error={"code": "EXECUTION_ERROR", "message": str(exc)},
                )
            )
            raise
        except Exception as exc:  # pragma: no cover - executor may raise arbitrary errors
            duration_ms = (time.perf_counter() - start_time) * 1000
            self._log_writer.write(
                McpLogEvent(
                    ts=datetime.now(tz=UTC),
                    agent_id=self._log_writer.agent_id,
                    task_id=self._log_writer.task_id,
                    step_id=step_id,
                    trace_id=trace_id,
                    span_id=span_id,
                    tool_id=tool_id,
                    event="invocation_failure",
                    status="error",
                    duration_ms=duration_ms,
                    attempt=attempt,
                    input_bytes=input_bytes,
                    output_bytes=0,
                    metadata={
                        "toolpack_version": toolpack.version,
                        "deterministic": toolpack.deterministic,
                        "schema_version": "0.1.0",
                    },
                    error={"code": "UNEXPECTED", "message": str(exc)},
                )
            )
            raise

        duration_ms = (time.perf_counter() - start_time) * 1000
        output_bytes = len(json.dumps(result, sort_keys=True)) if result else 0
        self._log_writer.write(
            McpLogEvent(
                ts=datetime.now(tz=UTC),
                agent_id=self._log_writer.agent_id,
                task_id=self._log_writer.task_id,
                step_id=step_id,
                trace_id=trace_id,
                span_id=span_id,
                tool_id=tool_id,
                event="invocation_success",
                status="success",
                duration_ms=duration_ms,
                attempt=attempt,
                input_bytes=input_bytes,
                output_bytes=output_bytes,
                metadata={
                    "toolpack_version": toolpack.version,
                    "deterministic": toolpack.deterministic,
                    "schema_version": "0.1.0",
                },
                error=None,
            )
        )
        return result
