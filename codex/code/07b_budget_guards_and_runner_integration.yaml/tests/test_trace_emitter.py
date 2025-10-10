import importlib.util
import pathlib
import sys
from types import MappingProxyType

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_module(name: str):
    module_path = pathlib.Path(__file__).resolve().parents[1] / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"task07b_{name}", module_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"unable to import {name}")
    if spec.name in sys.modules:
        return sys.modules[spec.name]
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


trace_mod = _load_module("trace")
budget_mod = _load_module("budget")

TraceEventEmitter = trace_mod.TraceEventEmitter
TraceEvent = trace_mod.TraceEvent

CostSnapshot = budget_mod.CostSnapshot
BudgetMode = budget_mod.BudgetMode
BudgetSpec = budget_mod.BudgetSpec
BudgetBreach = budget_mod.BudgetBreach
BudgetChargeOutcome = budget_mod.BudgetChargeOutcome


class FakeTraceWriter:
    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    def emit(self, event: TraceEvent) -> None:
        self.events.append(event)


def make_cost(value: int) -> CostSnapshot:
    return CostSnapshot({"time_ms": value})


def make_outcome(*, breach: BudgetBreach | None = None) -> BudgetChargeOutcome:
    spec = BudgetSpec(
        scope_type="loop",
        scope_id="loop-1",
        limit=make_cost(1000),
        mode=BudgetMode.HARD,
        breach_action="stop",
    )
    return BudgetChargeOutcome(
        scope_type="loop",
        scope_id="loop-1",
        cost=make_cost(300),
        spent=make_cost(700),
        remaining=make_cost(300),
        overages=make_cost(0),
        spec=spec,
        breach=breach,
    )


def test_emit_budget_charge_uses_immutable_payloads() -> None:
    writer = FakeTraceWriter()
    emitter = TraceEventEmitter(writer)
    outcome = make_outcome()

    emitter.emit_budget_charge(outcome)
    assert len(writer.events) == 1
    event = writer.events[0]
    assert event.event == "budget_charge"
    assert event.scope_type == "loop"
    assert isinstance(event.payload, MappingProxyType)
    assert isinstance(event.payload["spent"], MappingProxyType)
    assert event.payload["spent"]["time_ms"] == 700
    with pytest.raises(TypeError):
        event.payload["spent"]["time_ms"] = 999  # type: ignore[index]


def test_emit_budget_breach_includes_overages_and_kind() -> None:
    writer = FakeTraceWriter()
    emitter = TraceEventEmitter(writer)
    breach = BudgetBreach(
        scope_type="loop",
        scope_id="loop-1",
        kind="hard",
        action="stop",
        overages=make_cost(50),
        stop_reason="loop budget exceeded",
    )
    outcome = make_outcome(breach=breach)

    emitter.emit_budget_breach(outcome)
    assert len(writer.events) == 1
    event = writer.events[0]
    assert event.event == "budget_breach"
    assert event.payload["breach_kind"] == "hard"
    assert event.payload["overages"]["time_ms"] == 50
    assert event.payload["stop_reason"] == "loop budget exceeded"


def test_loop_summary_event_has_stop_reason_and_iteration() -> None:
    writer = FakeTraceWriter()
    emitter = TraceEventEmitter(writer)
    emitter.emit_loop_summary(
        loop_id="loop-1",
        iteration=2,
        stop_reason="budget-breach",
        remaining_iterations=0,
    )
    assert len(writer.events) == 1
    event = writer.events[0]
    assert event.event == "loop_summary"
    assert event.payload["iteration"] == 2
    assert event.payload["stop_reason"] == "budget-breach"
    assert event.scope_type == "loop"
