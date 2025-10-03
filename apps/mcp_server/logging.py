from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = ["McpLogEvent", "JsonLogWriter"]


@dataclass(slots=True)
class McpLogEvent:
    """Dataclass describing a structured MCP log event."""

    ts: datetime
    agent_id: str
    task_id: str
    step_id: int
    trace_id: str
    span_id: str
    tool_id: str
    event: str
    status: str
    duration_ms: float
    attempt: int
    input_bytes: int
    output_bytes: int
    metadata: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] | None = None
    run_id: str | None = None
    attempt_id: str | None = None

    def to_payload(self) -> dict[str, Any]:
        ts = self.ts if self.ts.tzinfo is not None else self.ts.replace(tzinfo=UTC)
        payload: dict[str, Any] = {
            "ts": ts.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            "agent_id": self.agent_id,
            "task_id": self.task_id,
            "step_id": self.step_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "tool_id": self.tool_id,
            "event": self.event,
            "status": self.status,
            "duration_ms": float(self.duration_ms),
            "attempt": int(self.attempt),
            "input_bytes": int(self.input_bytes),
            "output_bytes": int(self.output_bytes),
            "metadata": dict(self.metadata),
            "error": self.error if self.error is not None else None,
            "run_id": self.run_id,
            "attempt_id": self.attempt_id or f"{self.step_id}:{self.attempt}",
        }
        return payload


class JsonLogWriter:
    """Persist structured MCP log events as JSON lines."""

    def __init__(
        self,
        path: Path | str,
        *,
        agent_id: str,
        task_id: str,
        retention: str = "keep-last-5",
    ) -> None:
        self.path = Path(path)
        self.agent_id = agent_id
        self.task_id = task_id
        self._retention = retention
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self._open_handle()

    def _open_handle(self) -> Any:
        self._enforce_retention()
        return self.path.open("w", encoding="utf-8")

    def _enforce_retention(self) -> None:
        strategy = self._retention
        if not strategy.startswith("keep-last-"):
            return

        try:
            keep_count = int(strategy.split("-")[-1])
        except ValueError:
            keep_count = 5

        if keep_count <= 0:
            return

        for index in range(keep_count, 0, -1):
            rotated = self.path.with_suffix(self.path.suffix + f".{index}")
            if index == keep_count and rotated.exists():
                rotated.unlink()
            source = self.path if index == 1 else self.path.with_suffix(self.path.suffix + f".{index-1}")
            if source.exists():
                source.rename(rotated)

    def write(self, event: McpLogEvent) -> None:
        payload = event.to_payload()
        payload.setdefault("agent_id", self.agent_id)
        payload.setdefault("task_id", self.task_id)
        line = json.dumps(payload, separators=(",", ":"))
        self._handle.write(line + "\n")
        self.flush()

    def flush(self) -> None:
        if self._handle.closed:
            return
        self._handle.flush()
        try:
            os.fsync(self._handle.fileno())
        except OSError:
            # Some environments (e.g. tmpfs) do not support fsync; ignore.
            pass

    def close(self) -> None:
        if not self._handle.closed:
            self.flush()
            self._handle.close()

    def __enter__(self) -> "JsonLogWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
