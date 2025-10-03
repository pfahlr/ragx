from __future__ import annotations

from apps.mcp_server.models import Envelope


def test_envelope_success_generates_trace_id() -> None:
    envelope = Envelope.success(data={"hello": "world"}, transport="http")
    assert envelope.ok is True
    assert envelope.data == {"hello": "world"}
    assert envelope.meta["transport"] == "http"
    trace_id = envelope.meta.get("trace_id")
    assert isinstance(trace_id, str) and len(trace_id) >= 8
    assert envelope.errors == []
