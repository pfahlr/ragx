from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["Envelope", "EnvelopeMeta", "EnvelopeError", "ExecutionMeta", "IdempotencyMeta"]


class ExecutionMeta(BaseModel):
    """Execution telemetry associated with an envelope."""

    model_config = ConfigDict(frozen=True, populate_by_name=True, extra="forbid")

    duration_ms: float = Field(..., alias="durationMs")
    input_bytes: int = Field(..., alias="inputBytes")
    output_bytes: int = Field(..., alias="outputBytes")


class IdempotencyMeta(BaseModel):
    """Idempotency metadata describing cache behaviour."""

    model_config = ConfigDict(frozen=True, populate_by_name=True, extra="forbid")

    cache_hit: bool = Field(..., alias="cacheHit")
    cache_key: str | None = Field(default=None, alias="cacheKey")


class EnvelopeError(BaseModel):
    """Error payload returned when an envelope reports failure."""

    model_config = ConfigDict(frozen=True, populate_by_name=True, extra="forbid")

    code: str = Field(..., description="Stable error code")
    message: str = Field(..., description="Human readable error message")
    details: dict[str, Any] | None = Field(
        default=None,
        description="Optional structured details for debugging",
    )


class EnvelopeMeta(BaseModel):
    """Structured metadata attached to every envelope."""

    model_config = ConfigDict(frozen=True, populate_by_name=True, extra="forbid")

    request_id: str = Field(..., alias="requestId")
    trace_id: str = Field(..., alias="traceId")
    span_id: str = Field(..., alias="spanId")
    schema_version: str = Field(..., alias="schemaVersion")
    deterministic: bool
    transport: str
    route: str
    method: str
    duration_ms: float = Field(..., alias="durationMs")
    status: str
    attempt: int = 0
    input_bytes: int = Field(0, alias="inputBytes")
    output_bytes: int = Field(0, alias="outputBytes")
    tool_id: str | None = Field(default=None, alias="toolId")
    prompt_id: str | None = Field(default=None, alias="promptId")
    execution: ExecutionMeta
    idempotency: IdempotencyMeta

    @classmethod
    def from_ids(
        cls,
        *,
        request_id: str | UUID,
        trace_id: str | UUID,
        span_id: str | UUID,
        schema_version: str,
        deterministic: bool,
        transport: str,
        route: str,
        method: str,
        duration_ms: float,
        status: str,
        attempt: int = 0,
        input_bytes: int = 0,
        output_bytes: int = 0,
        tool_id: str | None = None,
        prompt_id: str | None = None,
        execution: ExecutionMeta | None = None,
        idempotency: IdempotencyMeta | None = None,
    ) -> EnvelopeMeta:
        execution_meta = execution or ExecutionMeta(
            durationMs=duration_ms,
            inputBytes=input_bytes,
            outputBytes=output_bytes,
        )
        idempotency_meta = idempotency or IdempotencyMeta(cacheHit=False, cacheKey=None)
        return cls(
            requestId=str(request_id),
            traceId=str(trace_id),
            spanId=str(span_id),
            schemaVersion=schema_version,
            deterministic=deterministic,
            transport=transport,
            route=route,
            method=method,
            durationMs=duration_ms,
            status=status,
            attempt=attempt,
            inputBytes=input_bytes,
            outputBytes=output_bytes,
            toolId=tool_id,
            promptId=prompt_id,
            execution=execution_meta,
            idempotency=idempotency_meta,
        )


class Envelope(BaseModel):
    """Canonical response envelope for the MCP server."""

    model_config = ConfigDict(frozen=True, populate_by_name=True, extra="forbid")

    ok: bool
    data: dict[str, Any] | None = None
    error: EnvelopeError | None = None
    meta: EnvelopeMeta

    @classmethod
    def success(cls, *, data: dict[str, Any], meta: EnvelopeMeta) -> Envelope:
        return cls(ok=True, data=data, error=None, meta=meta)

    @classmethod
    def failure(cls, *, error: EnvelopeError, meta: EnvelopeMeta) -> Envelope:
        return cls(ok=False, data=None, error=error, meta=meta)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation using canonical aliases."""

        return self.model_dump(by_alias=True)
