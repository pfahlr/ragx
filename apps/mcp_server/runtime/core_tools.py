from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from apps.mcp_server.logging import JsonLogWriter, McpLogEvent
from apps.toolpacks.executor import Executor, ToolpackExecutionError
from apps.toolpacks.loader import Toolpack

__all__ = ["CoreToolsRuntime"]


def _canonicalise(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _byte_size(payload: Mapping[str, Any]) -> int:
    return len(_canonicalise(payload).encode("utf-8"))


def _hash(payload: Mapping[str, Any]) -> str:
    import hashlib

    digest = hashlib.sha256()
    digest.update(_canonicalise(payload).encode("utf-8"))
    return digest.hexdigest()


@dataclass
class _InvocationContext:
    toolpack: Toolpack
    payload: Mapping[str, Any]
    attempt: int
    attempt_id: str
    step_id: int
    trace_id: str
    span_id: str
    start_monotonic: float
    payload_digest: str
    input_bytes: int


class CoreToolsRuntime:
    """Execute deterministic core tools with structured logging."""

    def __init__(
        self,
        *,
        toolpacks: list[Toolpack],
        executor: Executor,
        log_writer: JsonLogWriter,
        agent_id: str,
        task_id: str,
    ) -> None:
        self._toolpacks = {pack.id: pack for pack in toolpacks}
        self._executor = executor
        self._log_writer = log_writer
        self._agent_id = agent_id
        self._task_id = task_id
        self._step_counter = 0

    def invoke(self, tool_id: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        if tool_id not in self._toolpacks:
            raise KeyError(f"Unknown tool '{tool_id}'")

        toolpack = self._toolpacks[tool_id]
        context = self._start_invocation(toolpack, payload)
        self._log_writer.write(
            McpLogEvent(
                ts=datetime.now(UTC),
                agent_id=self._agent_id,
                task_id=self._task_id,
                step_id=context.step_id,
                trace_id=context.trace_id,
                span_id=context.span_id,
                tool_id=tool_id,
                event="tool.invoke",
                status="ok",
                attempt=context.attempt,
                execution={
                    "durationMs": 0.0,
                    "inputBytes": context.input_bytes,
                    "outputBytes": 0,
                },
                idempotency={"cacheHit": False},
                metadata={
                    "payloadDigest": context.payload_digest,
                    "toolpackVersion": toolpack.version,
                },
            ),
            attempt_id=context.attempt_id,
        )

        try:
            result = self._executor.run_toolpack(toolpack, payload)
        except ToolpackExecutionError as exc:
            self._log_writer.write(
                McpLogEvent(
                    ts=datetime.now(UTC),
                    agent_id=self._agent_id,
                    task_id=self._task_id,
                    step_id=context.step_id,
                    trace_id=context.trace_id,
                span_id=context.span_id,
                tool_id=tool_id,
                event="tool.err",
                status="err",
                attempt=context.attempt,
                execution={
                    "durationMs": self._duration_ms(context.start_monotonic),
                    "inputBytes": context.input_bytes,
                    "outputBytes": 0,
                },
                idempotency={"cacheHit": False},
                metadata={
                    "payloadDigest": context.payload_digest,
                    "toolpackVersion": toolpack.version,
                },
                error={"message": str(exc)},
                ),
                attempt_id=context.attempt_id,
            )
            raise

        output_mapping = dict(result)
        output_bytes = _byte_size(output_mapping)
        result_digest = _hash(output_mapping)

        self._log_writer.write(
            McpLogEvent(
                ts=datetime.now(UTC),
                agent_id=self._agent_id,
                task_id=self._task_id,
                step_id=context.step_id,
                trace_id=context.trace_id,
                span_id=context.span_id,
                tool_id=tool_id,
                event="tool.ok",
                status="ok",
                attempt=context.attempt,
                execution={
                    "durationMs": self._duration_ms(context.start_monotonic),
                    "inputBytes": context.input_bytes,
                    "outputBytes": output_bytes,
                },
                idempotency={"cacheHit": False},
                metadata={
                    "payloadDigest": context.payload_digest,
                    "resultDigest": result_digest,
                    "toolpackVersion": toolpack.version,
                },
            ),
            attempt_id=context.attempt_id,
        )
        return output_mapping

    def _start_invocation(
        self,
        toolpack: Toolpack,
        payload: Mapping[str, Any],
    ) -> _InvocationContext:
        self._step_counter += 1
        attempt_id = self._log_writer.new_attempt_id()
        trace_id = str(uuid4())
        span_id = str(uuid4())
        payload_mapping = dict(payload)
        payload_digest = _hash(payload_mapping)
        input_bytes = _byte_size(payload_mapping)
        return _InvocationContext(
            toolpack=toolpack,
            payload=payload_mapping,
            attempt=0,
            attempt_id=attempt_id,
            step_id=self._step_counter - 1,
            trace_id=trace_id,
            span_id=span_id,
            start_monotonic=time.perf_counter(),
            payload_digest=payload_digest,
            input_bytes=input_bytes,
        )

    @staticmethod
    def _duration_ms(start: float) -> float:
        return (time.perf_counter() - start) * 1000.0
