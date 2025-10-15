"""Phase 6 regression tests for TraceEventEmitter immutability."""

from __future__ import annotations

from types import MappingProxyType

import pytest

from pkgs.dsl.trace import TraceEventEmitter


def test_trace_emitter_deep_freezes_nested_structures() -> None:
    emitter = TraceEventEmitter()
    captured: list[MappingProxyType] = []

    def sink(event) -> None:
        captured.append(event.payload)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            event.payload["nested"]["inner"]["value"] = 42  # type: ignore[index]

    emitter.attach_sink(sink)

    payload = {
        "nested": {"inner": {"value": 1}},
        "items": [
            {"index": 0, "meta": {"label": "a"}},
            {"index": 1, "meta": {"label": "b"}},
        ],
    }
    event = emitter.emit(
        "custom_event",
        scope_type="run",
        scope_id="run-freeze",
        payload=payload,
    )

    assert isinstance(event.payload, MappingProxyType)
    nested = event.payload["nested"]
    assert isinstance(nested, MappingProxyType)
    assert isinstance(nested["inner"], MappingProxyType)
    with pytest.raises(TypeError):
        nested["inner"]["value"] = 99  # type: ignore[index]

    items = event.payload["items"]
    assert isinstance(items, tuple)
    assert all(isinstance(entry, MappingProxyType) for entry in items)
    assert all(isinstance(entry["meta"], MappingProxyType) for entry in items)

    # Sink receives the same immutable payload references
    assert captured and captured[0] is event.payload
    with pytest.raises(TypeError):
        captured[0]["nested"]["inner"]["value"] = 5  # type: ignore[index]
