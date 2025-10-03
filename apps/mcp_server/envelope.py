"""Envelope model used by the MCP server transports."""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


def _next_trace_id() -> str:
    return str(uuid4())


@dataclass(slots=True)
class Envelope:
    """Canonical envelope returned by MCP endpoints."""

    ok: bool
    data: Any | None = None
    meta: MutableMapping[str, Any] = field(default_factory=dict)
    errors: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def success(
        cls,
        data: Any | None = None,
        *,
        meta: MutableMapping[str, Any] | None = None,
        trace_id: str | None = None,
    ) -> Envelope:
        payload_meta: MutableMapping[str, Any] = dict(meta or {})
        payload_meta.setdefault("trace_id", trace_id or _next_trace_id())
        return cls(ok=True, data=data, meta=payload_meta, errors=[])

    @classmethod
    def failure(
        cls,
        errors: list[dict[str, Any]],
        *,
        data: Any | None = None,
        meta: MutableMapping[str, Any] | None = None,
        trace_id: str | None = None,
    ) -> Envelope:
        payload_meta: MutableMapping[str, Any] = dict(meta or {})
        payload_meta.setdefault("trace_id", trace_id or _next_trace_id())
        return cls(ok=False, data=data, meta=payload_meta, errors=errors)

    def ensure_trace_id(self) -> str:
        trace_id = self.meta.get("trace_id")
        if not trace_id:
            trace_id = _next_trace_id()
            self.meta["trace_id"] = trace_id
        return str(trace_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "data": self.data,
            "meta": dict(self.meta),
            "errors": list(self.errors),
        }
