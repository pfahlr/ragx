from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from apps.mcp_server.logging import JsonLogWriter

__all__ = ["EnvelopeValidationEvent", "EnvelopeValidationLogManager"]


@dataclass(slots=True)
class EnvelopeValidationEvent:
    """Structured log event emitted for envelope validation outcomes."""

    ts: datetime
    request_id: str
    trace_id: str
    span_id: str
    transport: str
    route: str
    method: str
    status: str
    execution: dict[str, Any]
    idempotency: dict[str, Any]
    attempt: int
    metadata: dict[str, Any]
    step_id: int
    error: dict[str, Any] | None = None

    def to_serialisable(self) -> dict[str, Any]:
        return {
            "ts": self.ts.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            "agentId": "mcp_server",
            "taskId": "06cV2B_mcp_envelope_and_schema_validation_B",
            "stepId": self.step_id,
            "transport": self.transport,
            "route": self.route,
            "method": self.method,
            "status": self.status,
            "attempt": self.attempt,
            "execution": dict(self.execution),
            "idempotency": dict(self.idempotency),
            "requestId": self.request_id,
            "traceId": self.trace_id,
            "spanId": self.span_id,
            "metadata": dict(self.metadata),
            "error": self.error,
        }


class EnvelopeValidationLogManager:
    """Persist envelope validation events using :class:`JsonLogWriter`."""

    def __init__(
        self,
        *,
        log_dir: Path,
        schema_version: str,
        deterministic: bool,
        retention: int = 5,
    ) -> None:
        prefix = Path(log_dir) / "logs" / "mcp_server" / "envelope_validation"
        latest = Path(log_dir) / "logs" / "mcp_server" / "envelope_validation.latest.jsonl"
        self._writer = JsonLogWriter(
            agent_id="mcp_server",
            task_id="06cV2B_mcp_envelope_and_schema_validation_B",
            storage_prefix=prefix,
            latest_symlink=latest,
            schema_version=schema_version,
            deterministic=deterministic,
            root_dir=Path(log_dir),
            retention=retention,
        )
        self._step = 0

    @property
    def writer(self) -> JsonLogWriter:
        return self._writer

    def next_step_id(self) -> int:
        self._step += 1
        return self._step

    def emit(self, event: EnvelopeValidationEvent) -> None:
        attempt_id = self._writer.new_attempt_id()
        self._writer.write(event, attempt_id=attempt_id)
