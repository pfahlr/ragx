"""Structured logging helpers for MCP core tools."""
from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass
class McpLogEvent:
    """Represents a single structured log event for the MCP runtime."""

    ts: datetime | str
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
    run_id: str | None = None
    attempt_id: str | None = None

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        ts_value = payload["ts"]
        if isinstance(ts_value, datetime):
            payload["ts"] = ts_value.astimezone(UTC).isoformat()
        return payload


class JsonLogWriter:
    """Persist :class:`McpLogEvent` instances as JSON lines."""

    def __init__(
        self,
        path: str | Path,
        *,
        agent_id: str,
        task_id: str,
        run_id: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._agent_id = agent_id
        self._task_id = task_id
        self._run_id = run_id or str(uuid4())
        self._lock = threading.Lock()
        self._handle = self.path.open("a", encoding="utf-8")

    def write(self, event: McpLogEvent) -> None:
        payload = event.to_payload()
        payload["agent_id"] = self._agent_id
        payload["task_id"] = self._task_id
        payload["run_id"] = self._run_id
        attempt_identifier = payload.get("attempt_id")
        if not attempt_identifier:
            attempt_identifier = f"{payload['tool_id']}:{payload['attempt']}"
        payload["attempt_id"] = attempt_identifier
        payload.setdefault("metadata", {})
        payload["metadata"] = dict(payload["metadata"])
        payload["metadata"].setdefault("schema_version", "0.1.0")
        payload["metadata"].setdefault("deterministic", True)

        with self._lock:
            self._handle.write(json.dumps(payload, sort_keys=True) + "\n")
            self._handle.flush()

    def flush(self) -> None:
        with self._lock:
            self._handle.flush()

    def close(self) -> None:
        with self._lock:
            if not self._handle.closed:
                self._handle.flush()
                self._handle.close()

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def run_id(self) -> str:
        return self._run_id

    def __enter__(self) -> JsonLogWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.close()

    def __del__(self) -> None:  # pragma: no cover - defensive close
        try:
            self.close()
        except Exception:
            pass
