from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

__all__ = ["Envelope"]


class Envelope(BaseModel):
    """Response envelope shared across transports."""

    ok: bool
    data: dict[str, Any] | None = None
    meta: dict[str, Any] = Field(default_factory=dict)
    errors: list[dict[str, Any]] = Field(default_factory=list)

    @classmethod
    def success(
        cls,
        *,
        data: dict[str, Any] | None = None,
        transport: str,
        trace_id: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> "Envelope":
        payload_meta: dict[str, Any] = {"transport": transport, "trace_id": trace_id or str(uuid4())}
        if meta:
            payload_meta.update(meta)
        return cls(ok=True, data=data or {}, meta=payload_meta, errors=[])

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        return super().model_dump(*args, **kwargs)
