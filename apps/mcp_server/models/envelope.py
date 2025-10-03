"""Envelope models for MCP responses."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["Envelope", "EnvelopeMeta"]


class EnvelopeMeta(BaseModel):
    """Metadata associated with every MCP envelope."""

    model_config = ConfigDict(extra="forbid")

    tool: str
    version: str = Field(default="0.1.0")
    durationMs: int = Field(default=0, ge=0)
    traceId: str = Field(default_factory=lambda: str(uuid4()))
    warnings: list[str] = Field(default_factory=list)


class Envelope(BaseModel):
    """Uniform response envelope emitted by the MCP server."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    data: dict[str, Any]
    meta: EnvelopeMeta
    errors: list[dict[str, Any]] = Field(default_factory=list)

    @classmethod
    def success(
        cls,
        *,
        data: Mapping[str, Any] | None,
        tool: str,
        version: str = "0.1.0",
        duration_ms: int = 0,
        trace_id: str | None = None,
        warnings: Iterable[str] | None = None,
    ) -> Envelope:
        """Create a successful envelope with deterministic defaults."""

        meta = EnvelopeMeta(
            tool=tool,
            version=version,
            durationMs=duration_ms,
            traceId=trace_id or str(uuid4()),
            warnings=list(warnings or ()),
        )
        payload = dict(data or {})
        return cls(ok=True, data=payload, meta=meta, errors=[])

    @classmethod
    def failure(
        cls,
        *,
        tool: str,
        errors: Iterable[Mapping[str, Any]],
        data: Mapping[str, Any] | None = None,
        version: str = "0.1.0",
        duration_ms: int = 0,
        trace_id: str | None = None,
        warnings: Iterable[str] | None = None,
    ) -> Envelope:
        """Create an error envelope (placeholder for future tasks)."""

        meta = EnvelopeMeta(
            tool=tool,
            version=version,
            durationMs=duration_ms,
            traceId=trace_id or str(uuid4()),
            warnings=list(warnings or ()),
        )
        payload = dict(data or {})
        return cls(ok=False, data=payload, meta=meta, errors=[dict(item) for item in errors])

    def to_serialisable(self) -> dict[str, Any]:
        """Return an envelope as a plain serialisable mapping."""

        return self.model_dump()
