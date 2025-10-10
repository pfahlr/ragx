from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from codex.code.work.budget import (  # noqa: E402
    BudgetManager,
    BudgetMode,
    BudgetSpec,
)
from codex.code.work.trace import TraceEventEmitter  # noqa: E402


@pytest.fixture()
def trace_emitter() -> TraceEventEmitter:
    return TraceEventEmitter()


def _build_manager(*, mode: BudgetMode, breach_action: str, emitter: TraceEventEmitter) -> BudgetManager:
    spec = BudgetSpec(
        scope="run",
        limits={"tokens": 10, "milliseconds": 1000},
        mode=mode,
        breach_action=breach_action,
    )
    return BudgetManager({"run": spec}, trace_emitter=emitter)


def test_soft_budget_warns_but_allows_execution(trace_emitter: TraceEventEmitter) -> None:
    manager = _build_manager(mode=BudgetMode.SOFT, breach_action="warn", emitter=trace_emitter)

    check_ok = manager.preflight("run", {"tokens": 6})
    assert check_ok.stop_requested is False
    assert not check_ok.breaches

    outcome_ok = manager.commit("run", {"tokens": 6})
    assert outcome_ok.stop is False
    assert outcome_ok.remaining["tokens"] == 4
    assert not outcome_ok.warnings

    check_warn = manager.preflight("run", {"tokens": 5})
    assert check_warn.stop_requested is False
    assert check_warn.breaches

    outcome_warn = manager.commit("run", {"tokens": 5})
    assert outcome_warn.stop is False
    assert outcome_warn.overages["tokens"] == 1
    assert outcome_warn.warnings

    breach_events = [event for event in trace_emitter.events if event.event == "budget_breach"]
    assert breach_events, "expected a budget_breach trace event to be emitted"
    payload = breach_events[0].payload
    with pytest.raises(TypeError):
        payload["tokens"] = 0  # mapping_proxy enforces immutability


def test_hard_budget_requests_stop(trace_emitter: TraceEventEmitter) -> None:
    manager = _build_manager(mode=BudgetMode.HARD, breach_action="stop", emitter=trace_emitter)

    check_stop = manager.preflight("run", {"tokens": 11, "milliseconds": 800})
    assert check_stop.stop_requested is True
    assert any(b.metric == "tokens" for b in check_stop.breaches)

    outcome_stop = manager.commit("run", {"tokens": 11, "milliseconds": 800})
    assert outcome_stop.stop is True
    assert outcome_stop.overages["tokens"] == 1
    assert any(event.event == "budget_breach" for event in trace_emitter.events)


def test_unknown_scope_is_rejected(trace_emitter: TraceEventEmitter) -> None:
    manager = _build_manager(mode=BudgetMode.SOFT, breach_action="warn", emitter=trace_emitter)
    with pytest.raises(KeyError):
        manager.preflight("loop-1", {"tokens": 1})
