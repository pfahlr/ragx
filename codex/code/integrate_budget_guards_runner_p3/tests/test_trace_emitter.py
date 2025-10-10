"""Unit tests for TraceEventEmitter."""

from __future__ import annotations

import pytest

from codex.code.integrate_budget_guards_runner_p3.dsl.trace import TraceEventEmitter


def test_emit_records_event_and_forwards_to_sink() -> None:
    emitter = TraceEventEmitter()
    received: list = []

    def sink(event):  # type: ignore[no-untyped-def]
        received.append(event)

    emitter.attach_sink(sink)
    event = emitter.emit(
        "budget_charge",
        scope_type="run",
        scope_id="run-1",
        payload={"value": 10},
    )

    assert event.payload["value"] == 10
    with pytest.raises(TypeError):  # immutability enforced
        event.payload["value"] = 11  # type: ignore[index]

    assert received == [event]
    assert emitter.events == (event,)

    emitter.clear()
    assert emitter.events == ()


def test_attach_sink_none_disables_forwarding() -> None:
    emitter = TraceEventEmitter()
    received: list = []

    emitter.attach_sink(None)
    event = emitter.emit("policy_resolved", scope_type="stack", scope_id="root", payload={})

    assert received == []
    assert emitter.events == (event,)
