import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from codex.code.work.pkgs.dsl.costs import normalize_cost
from codex.code.work.pkgs.dsl.budget import BudgetSpec, BudgetMode, BudgetBreachError, BudgetStopSignal
from codex.code.work.pkgs.dsl.budget_manager import BudgetManager


class DummyTraceWriter:
    def __init__(self):
        self.events = []

    def write(self, event_name: str, payload: dict) -> None:
        self.events.append((event_name, payload))


def test_normalize_cost_converts_seconds_and_validates_metrics():
    normalized = normalize_cost({"seconds": 1.5, "calls": 2})
    assert normalized["duration_ms"] == 1500
    assert normalized["calls"] == 2

    with pytest.raises(ValueError):
        normalize_cost({"unknown": 1})


def test_budget_manager_warns_on_soft_breach_without_raising():
    manager = BudgetManager(trace_writer=DummyTraceWriter())
    manager.register_scope(
        scope_id="node:unit",
        spec=BudgetSpec(metric="calls", limit=1, mode=BudgetMode.SOFT, breach_action="warn"),
    )

    outcome = manager.charge("node:unit", {"calls": 2})

    assert outcome.breached is True
    assert outcome.action == "warn"
    assert outcome.remaining == 0
    assert outcome.overage == 1


def test_budget_manager_raises_on_hard_breach():
    manager = BudgetManager(trace_writer=DummyTraceWriter())
    manager.register_scope(
        scope_id="run",
        spec=BudgetSpec(metric="calls", limit=1, mode=BudgetMode.HARD, breach_action="error"),
    )

    with pytest.raises(BudgetBreachError) as exc:
        manager.charge("run", {"calls": 2})

    assert exc.value.outcome.scope_id == "run"
    assert exc.value.outcome.overage == 1
    assert exc.value.outcome.action == "error"


def test_budget_manager_emits_stop_signal():
    trace = DummyTraceWriter()
    manager = BudgetManager(trace_writer=trace)
    manager.register_scope(
        scope_id="loop:jobs",
        spec=BudgetSpec(metric="calls", limit=1, mode=BudgetMode.HARD, breach_action="stop"),
    )

    with pytest.raises(BudgetStopSignal) as exc:
        manager.charge("loop:jobs", {"calls": 3})

    assert exc.value.outcome.overage == 2
    assert exc.value.outcome.action == "stop"
    assert trace.events[-1][0] == "budget_breach"
    assert trace.events[-1][1]["scope"] == "loop:jobs"
