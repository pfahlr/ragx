from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

__all__ = ["JsonLogWriter", "McpLogEvent"]


@dataclass(slots=True)
class McpLogEvent:
    """Structured log event emitted by the MCP runtime."""

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
    metadata: dict[str, Any]
    error: dict[str, Any] | None = None
    run_id: str | None = None
    attempt_id: str | None = None

    def serialise(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["ts"] = self.ts.astimezone(UTC).isoformat()
        return payload


class JsonLogWriter:
    """Persist structured JSONL logs with basic retention semantics."""

    def __init__(
        self,
        path: str | Path,
        *,
        agent_id: str,
        task_id: str,
        retention: int = 5,
    ) -> None:
        self.path = Path(path)
        self._agent_id = agent_id
        self._task_id = task_id
        self._lock = Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._apply_retention(retention)
        self._file = self.path.open("a", encoding="utf-8")

    def _apply_retention(self, retention: int) -> None:
        if retention <= 0:
            return
        existing = sorted(
            self.path.parent.glob(f"{self.path.stem}*.{self.path.suffix.lstrip('.')}")
        )
        if len(existing) <= retention:
            return
        for stale in existing[:-retention]:
            try:
                stale.unlink()
            except OSError:
                continue

    def write(self, event: McpLogEvent) -> None:
        record = event.serialise()
        record.setdefault("agent_id", self._agent_id)
        record.setdefault("task_id", self._task_id)
        line = json.dumps(record, sort_keys=True, separators=(",", ":"))
        with self._lock:
            self._file.write(f"{line}\n")
            self._file.flush()

    def flush(self) -> None:
        with self._lock:
            self._file.flush()

    def close(self) -> None:
        with self._lock:
            if not self._file.closed:
                self._file.flush()
                self._file.close()

    def __enter__(self) -> JsonLogWriter:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:  # pragma: no cover - trivial
        self.close()
