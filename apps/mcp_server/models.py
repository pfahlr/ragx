from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Envelope(BaseModel):
    """Canonical response envelope for MCP transports."""

    ok: bool = Field(default=True)
    data: Any = Field(default=None)
    meta: dict[str, Any] = Field(default_factory=dict)
    errors: list[dict[str, Any]] = Field(default_factory=list)

    model_config = ConfigDict(
        populate_by_name=True,
        extra="allow",
    )

    @classmethod
    def success(
        cls,
        *,
        data: Any,
        trace_id: str,
        transport: str,
        meta: dict[str, Any] | None = None,
    ) -> Envelope:
        payload_meta: dict[str, Any] = {"traceId": trace_id, "transport": transport}
        if meta:
            payload_meta.update(meta)
        return cls(ok=True, data=data, meta=payload_meta, errors=[])

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
