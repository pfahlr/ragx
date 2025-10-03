"""Runtime orchestration for the minimal core MCP tools."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from apps.mcp_server.logging import JsonLogWriter, McpLogEvent
from apps.toolpacks.executor import Executor
from apps.toolpacks.loader import Toolpack


def _measure_bytes(payload: Any) -> int:
    """Return the number of UTF-8 bytes required to serialise ``payload``."""

    try:
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return len(encoded.encode("utf-8"))
    except TypeError:
        return len(str(payload).encode("utf-8"))


@dataclass(frozen=True)
class RuntimeContext:
    """Per-invocation context metadata shared across attempts."""

    toolpack: Toolpack
    step_id: int
    trace_id: str
    input_bytes: int

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "toolpack_id": self.toolpack.id,
            "toolpack_version": self.toolpack.version,
            "deterministic": self.toolpack.deterministic,
            "source_path": str(self.toolpack.source_path),
        }


class CoreToolsRuntime:
    """Load and execute the minimal set of MCP core tools."""

    def __init__(
        self,
        *,
        toolpacks: Sequence[Toolpack],
        executor: Executor,
        log_writer: JsonLogWriter,
    ) -> None:
        self._executor = executor
        self._log_writer = log_writer
        self._toolpacks = {toolpack.id: toolpack for toolpack in toolpacks}
        self._step_counter = 0

    def invoke(
        self,
        tool_id: str,
        payload: Mapping[str, Any],
        *,
        max_attempts: int = 1,
    ) -> dict[str, Any]:
        """Execute ``tool_id`` with ``payload`` and structured logging."""

        if tool_id not in self._toolpacks:
            raise KeyError(f"Unknown tool id: {tool_id}")

        toolpack = self._toolpacks[tool_id]
        self._step_counter += 1
        step_id = self._step_counter
        trace_id = uuid4().hex
        input_bytes = _measure_bytes(payload)
        attempts = max(1, int(max_attempts))
        context = RuntimeContext(toolpack=toolpack, step_id=step_id, trace_id=trace_id, input_bytes=input_bytes)

        last_error: Exception | None = None

        for attempt in range(1, attempts + 1):
            span_id = uuid4().hex
            start_event = self._build_event(
                context=context,
                span_id=span_id,
                attempt=attempt,
                event="invocation_start",
                status="running",
                duration_ms=0.0,
                output_bytes=0,
                metadata={**context.metadata, "phase": "start", "max_attempts": attempts},
                error=None,
            )
            self._log_writer.write(start_event)

            start = time.perf_counter()
            try:
                result = self._executor.run_toolpack(toolpack, payload)
            except Exception as exc:  # noqa: BLE001 - propagate but log first
                duration = (time.perf_counter() - start) * 1000
                remaining = max(attempts - attempt, 0)
                event_name = "invocation_retry" if remaining else "invocation_failure"
                metadata = {
                    **context.metadata,
                    "phase": "retry" if remaining else "failure",
                    "remaining_attempts": remaining,
                }
                error_payload = {"code": exc.__class__.__name__, "message": str(exc)}
                failure_event = self._build_event(
                    context=context,
                    span_id=span_id,
                    attempt=attempt,
                    event=event_name,
                    status="error",
                    duration_ms=duration,
                    output_bytes=0,
                    metadata=metadata,
                    error=error_payload,
                )
                self._log_writer.write(failure_event)

                last_error = exc
                if remaining:
                    continue
                raise

            duration = (time.perf_counter() - start) * 1000
            output_bytes = _measure_bytes(result)
            metadata = {
                **context.metadata,
                "phase": "success",
                "attempts_used": attempt,
            }
            success_event = self._build_event(
                context=context,
                span_id=span_id,
                attempt=attempt,
                event="invocation_success",
                status="success",
                duration_ms=duration,
                output_bytes=output_bytes,
                metadata=metadata,
                error=None,
            )
            self._log_writer.write(success_event)
            return result

        if last_error is not None:  # pragma: no cover - loop guarantees raise before
            raise last_error
        raise RuntimeError(f"Invocation for {tool_id} exited without producing a result")

    def _build_event(
        self,
        *,
        context: RuntimeContext,
        span_id: str,
        attempt: int,
        event: str,
        status: str,
        duration_ms: float,
        output_bytes: int,
        metadata: Mapping[str, Any],
        error: Mapping[str, Any] | None,
    ) -> McpLogEvent:
        return McpLogEvent(
            ts=datetime.now(timezone.utc),
            agent_id=self._log_writer.agent_id,
            task_id=self._log_writer.task_id,
            step_id=context.step_id,
            trace_id=context.trace_id,
            span_id=span_id,
            tool_id=context.toolpack.id,
            event=event,
            status=status,
            duration_ms=duration_ms,
            attempt=attempt,
            input_bytes=context.input_bytes,
            output_bytes=output_bytes,
            metadata=dict(metadata),
            error=error,
        )


__all__ = ["CoreToolsRuntime"]

