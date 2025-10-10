from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "budget_integration.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("budget_integration", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    loader = spec.loader
    assert loader is not None
    sys.modules[spec.name] = module
    loader.exec_module(module)  # type: ignore[assignment]
    return module


@pytest.fixture(scope="module")
def budget_module():
    return load_module()


def test_trace_writer_enforces_schema(budget_module):
    TraceWriter = budget_module.ListTraceWriter

    writer = TraceWriter()
    writer.emit("policy_push", {"scope": "run", "data": {"policy": "default"}})

    event = writer.events[0]
    assert set(event) == {"timestamp", "scope", "event", "data", "sequence"}
    assert event["event"] == "policy_push"
    assert event["scope"] == "run"
    assert event["sequence"] == 0
    assert event["timestamp"].endswith("Z")

    with pytest.raises(ValueError):
        writer.emit("budget_preflight", {"data": {}})


def test_trace_writer_preserves_chronological_order(budget_module):
    TraceWriter = budget_module.ListTraceWriter

    writer = TraceWriter()
    writer.emit("policy_push", {"scope": "run", "data": {}})
    writer.emit("budget_charge", {"scope": "node:1", "data": {}})

    timestamps = [event["timestamp"] for event in writer.events]
    assert timestamps == sorted(timestamps)
    assert [event["sequence"] for event in writer.events] == [0, 1]
