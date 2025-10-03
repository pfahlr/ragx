from __future__ import annotations

import json
import time
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from apps.mcp_server.logging import JsonLogWriter, McpLogEvent
from apps.toolpacks.executor import Executor, ToolpackExecutionError
from apps.toolpacks.loader import Toolpack

__all__ = ["CoreToolsRuntime"]


def _byte_size(payload: Mapping[str, Any] | None) -> int:
    if payload is None:
        return 0
    serialised = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return len(serialised.encode("utf-8"))


class CoreToolsRuntime:
    """Runtime responsible for invoking the minimal core MCP toolpacks."""

    def __init__(
        self,
        *,
        toolpacks: Sequence[Toolpack],
        executor: Executor,
        log_writer: JsonLogWriter,
    ) -> None:
        if not toolpacks:
            raise ValueError("At least one toolpack must be supplied")
        self._toolpacks: dict[str, Toolpack] = {}
        for toolpack in toolpacks:
            self._toolpacks[toolpack.id] = toolpack
            self._toolpacks[toolpack.id.replace(":", ".")] = toolpack
            self._toolpacks[toolpack.id.replace(".", ":")] = toolpack
        self.executor = executor
        self.log_writer = log_writer
        self._step_id = 0
        self._run_id = f"core-tools-{uuid.uuid4()}"

    def invoke(
        self,
        tool_id: str,
        payload: Mapping[str, Any],
        *,
        max_attempts: int = 1,
    ) -> dict[str, Any]:
        canonical_id = tool_id.replace(":", ".")
        toolpack = self._toolpacks.get(tool_id) or self._toolpacks.get(canonical_id)
        if toolpack is None:
            raise KeyError(f"Unknown tool id '{tool_id}'")
        self._step_id += 1
        step_id = self._step_id
        attempts = max(1, max_attempts)
        last_error: Exception | None = None
        input_payload = dict(payload)
        simulate_error = bool(input_payload.pop("simulate_error", False))

        for attempt in range(1, attempts + 1):
            trace_id = str(uuid.uuid4())
            span_id = str(uuid.uuid4())
            start_time = time.perf_counter()
            self._log_event(
                event="invocation_start",
                status="running",
                toolpack=toolpack,
                tool_identifier=tool_id,
                step_id=step_id,
                attempt=attempt,
                input_payload=input_payload,
                output_payload=None,
                duration_ms=0.0,
                trace_id=trace_id,
                span_id=span_id,
                error=None,
            )

            try:
                if simulate_error:
                    raise RuntimeError("Simulated failure for retry testing")
                result = self.executor.run_toolpack(toolpack, input_payload)
                duration_ms = (time.perf_counter() - start_time) * 1000
                self._log_event(
                    event="invocation_success",
                    status="success",
                    toolpack=toolpack,
                    tool_identifier=tool_id,
                    step_id=step_id,
                    attempt=attempt,
                    input_payload=input_payload,
                    output_payload=result,
                    duration_ms=duration_ms,
                    trace_id=trace_id,
                    span_id=span_id,
                    error=None,
                )
                return result
            except (ToolpackExecutionError, RuntimeError) as exc:
                duration_ms = (time.perf_counter() - start_time) * 1000
                last_error = exc
                event_name = "invocation_retry" if attempt < attempts else "invocation_failure"
                self._log_event(
                    event=event_name,
                    status="error",
                    toolpack=toolpack,
                    tool_identifier=tool_id,
                    step_id=step_id,
                    attempt=attempt,
                    input_payload=input_payload,
                    output_payload=None,
                    duration_ms=duration_ms,
                    trace_id=trace_id,
                    span_id=span_id,
                    error={"code": exc.__class__.__name__, "message": str(exc)},
                )
                if attempt >= attempts:
                    raise RuntimeError(f"Tool {tool_id} failed after {attempts} attempt(s)") from exc
                simulate_error = False
                continue
        raise RuntimeError(f"Tool {tool_id} failed to execute") from last_error

    def _log_event(
        self,
        *,
        event: str,
        status: str,
        toolpack: Toolpack,
        tool_identifier: str,
        step_id: int,
        attempt: int,
        input_payload: Mapping[str, Any] | None,
        output_payload: Mapping[str, Any] | None,
        duration_ms: float,
        trace_id: str,
        span_id: str,
        error: dict[str, Any] | None,
    ) -> None:
        metadata = {
            "toolpack_id": toolpack.id,
            "toolpack_version": toolpack.version,
            "deterministic": toolpack.deterministic,
        }
        event_payload = McpLogEvent(
            ts=datetime.now(UTC),
            agent_id=self.log_writer.agent_id,
            task_id=self.log_writer.task_id,
            step_id=step_id,
            trace_id=trace_id,
            span_id=span_id,
            tool_id=tool_identifier,
            event=event,
            status=status,
            duration_ms=duration_ms,
            attempt=attempt,
            input_bytes=_byte_size(input_payload),
            output_bytes=_byte_size(output_payload),
            metadata=metadata,
            error=error,
            run_id=self._run_id,
        )
        self.log_writer.write(event_payload)
