"""Structured logging utilities for the MCP server."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4


@dataclass
class McpLogEvent:
    """In-memory representation of a structured MCP log event."""

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
    metadata: Mapping[str, Any] = field(default_factory=dict)
    error: Mapping[str, Any] | None = None

    def to_payload(self) -> dict[str, Any]:
        """Serialise the event to a JSON-compatible payload."""

        ts = self.ts
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        else:
            ts = ts.astimezone(timezone.utc)
        ts_value = ts.isoformat().replace("+00:00", "Z")

        metadata: Mapping[str, Any] = self.metadata or {}
        if not isinstance(metadata, Mapping):  # pragma: no cover - defensive
            metadata = {"value": metadata}

        error: Mapping[str, Any] | None
        if self.error is None:
            error = None
        elif isinstance(self.error, Mapping):
            error = dict(self.error)
        else:  # pragma: no cover - defensive
            error = {"message": str(self.error)}

        return {
            "ts": ts_value,
            "agent_id": self.agent_id,
            "task_id": self.task_id,
            "step_id": int(self.step_id),
            "trace_id": str(self.trace_id),
            "span_id": str(self.span_id),
            "tool_id": self.tool_id,
            "event": self.event,
            "status": self.status,
            "duration_ms": float(self.duration_ms),
            "attempt": int(self.attempt),
            "input_bytes": int(self.input_bytes),
            "output_bytes": int(self.output_bytes),
            "metadata": dict(metadata),
            "error": error,
        }


class JsonLogWriter:
    """Persist structured MCP log events to newline-delimited JSON."""

    def __init__(
        self,
        path: str | Path,
        *,
        agent_id: str,
        task_id: str,
        retention: int = 5,
    ) -> None:
        self.path = Path(path)
        self.agent_id = agent_id
        self.task_id = task_id
        self._retention = max(retention, 1)
        self._lock = threading.Lock()
        self._run_id = uuid4().hex
        self._sequence = 0
        self._buffer: list[str] = []
        self._handle = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._enforce_retention()

    def write(self, event: McpLogEvent) -> None:
        """Append ``event`` to the log file, retrying on transient errors."""

        payload = event.to_payload()
        payload["agent_id"] = self.agent_id
        payload["task_id"] = self.task_id
        payload.setdefault("step_id", event.step_id)
        payload.setdefault("event", event.event)
        payload["run_id"] = self._run_id
        payload["attempt_id"] = f"{self._run_id}:{payload['step_id']}:{event.attempt}:{self._sequence}"
        self._sequence += 1

        serialised = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        line = f"{serialised}\n"

        with self._lock:
            self._buffer.append(line)
            self._ensure_handle()
            self._flush_buffer()

    def flush(self) -> None:
        """Flush buffered log lines to disk."""

        with self._lock:
            self._ensure_handle()
            self._flush_buffer()

    def close(self) -> None:
        """Flush buffered events and close the underlying file handle."""

        with self._lock:
            self._ensure_handle()
            self._flush_buffer()
            if self._handle is not None:
                try:
                    self._handle.flush()
                finally:
                    self._handle.close()
                    self._handle = None

    def __enter__(self) -> "JsonLogWriter":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _ensure_handle(self) -> None:
        if self._handle is not None:
            return
        try:
            self._handle = self.path.open("a", encoding="utf-8")
        except OSError:
            self._handle = None

    def _flush_buffer(self) -> None:
        if not self._buffer or self._handle is None:
            return
        try:
            self._handle.writelines(self._buffer)
            self._handle.flush()
            self._buffer.clear()
        except OSError:
            # Leave the buffer intact and close the handle so we can retry.
            try:
                self._handle.close()
            finally:  # pragma: no branch - close best-effort
                self._handle = None

    def _enforce_retention(self) -> None:
        directory = self.path.parent
        try:
            candidates = sorted(
                (p for p in directory.glob("*.jsonl") if p.is_file()),
                key=lambda entry: entry.stat().st_mtime,
            )
        except OSError:
            return

        excess = len(candidates) - self._retention
        if excess <= 0:
            return

        for old_path in candidates[:excess]:
            if old_path == self.path:
                continue
            try:
                old_path.unlink()
            except OSError:
                continue


__all__ = ["JsonLogWriter", "McpLogEvent"]

