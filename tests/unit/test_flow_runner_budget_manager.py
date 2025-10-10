"""FlowRunner budget manager contract tests."""

from __future__ import annotations

import pytest

from pkgs.dsl.budget import BudgetBreachHard
from pkgs.dsl.runner import FlowRunner


def test_flow_runner_run_budget_blocks_node_on_preflight() -> None:
    runner = FlowRunner()
    spec = {
        "globals": {"run_budget": {"max_usd": 1.0}},
        "graph": {"nodes": [{"id": "alpha", "kind": "unit", "budget": {"max_usd": 0.9}}]},
    }

    runner.prepare_budgets(spec)

    with pytest.raises(BudgetBreachHard) as exc_info:
        runner.budget_manager.preflight_node("alpha", {"usd": 1.2})

    err = exc_info.value
    assert err.scope_type == "run"
    assert err.metrics == ("usd",)


def test_flow_runner_soft_budget_emits_warning_and_trace() -> None:
    runner = FlowRunner()
    spec = {
        "globals": {},
        "graph": {
            "nodes": [
                {
                    "id": "draft",
                    "kind": "unit",
                    "budget": None,
                    "spec": {"budget": {"mode": "soft", "max_tokens": 100}},
                }
            ]
        },
    }

    runner.prepare_budgets(spec)

    preflight = runner.budget_manager.preflight_node("draft", {"tokens": 150})
    assert len(preflight.warnings) == 1
    warning = preflight.warnings[0]
    assert warning.scope_id == "draft"
    assert warning.metrics == ("tokens",)
    assert warning.severity == "soft"

    commit = runner.budget_manager.commit_node("draft", {"tokens": 150})
    assert len(commit.warnings) == 1
    assert commit.warnings[0].scope_id == "draft"

    breach_events = [
        event
        for event in runner.trace.events
        if event.event == "budget_breach" and event.scope_id == "draft"
    ]
    assert breach_events, "soft breach should be traced"
    assert breach_events[-1].data["severity"] == "soft"


def test_flow_runner_loop_budget_stop_condition() -> None:
    runner = FlowRunner()
    spec = {
        "globals": {},
        "graph": {
            "nodes": [
                {
                    "id": "loop1",
                    "kind": "loop",
                    "stop": {"budget": {"max_calls": 2, "breach_action": "stop"}},
                }
            ]
        },
    }

    runner.prepare_budgets(spec)

    first = runner.budget_manager.commit_loop_iteration("loop1", {"calls": 1})
    assert first.should_stop is False

    second = runner.budget_manager.commit_loop_iteration("loop1", {"calls": 1})
    assert second.should_stop is True
    assert second.stop_reason == "budget_stop"

    breach_events = [
        event
        for event in runner.trace.events
        if event.event == "budget_breach" and event.scope_id == "loop1"
    ]
    assert breach_events[-1].data["action"] == "stop"


def test_flow_runner_budget_charge_records_trace_per_scope() -> None:
    runner = FlowRunner()
    spec = {
        "globals": {"run_budget": {"max_usd": 2.0}},
        "graph": {
            "nodes": [
                {
                    "id": "alpha",
                    "kind": "unit",
                    "budget": {"max_usd": 1.5},
                    "spec": {},
                }
            ]
        },
    }

    runner.prepare_budgets(spec)

    runner.budget_manager.preflight_node("alpha", {"usd": 0.6})
    runner.budget_manager.commit_node("alpha", {"usd": 0.6})

    charge_events = [event for event in runner.trace.events if event.event == "budget_charge"]
    scopes = {(event.scope_type, event.scope_id) for event in charge_events}
    assert ("run", "run") in scopes
    assert ("node", "alpha") in scopes

