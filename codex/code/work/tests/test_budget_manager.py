"""Unit tests for budget domain models and manager orchestration."""

from __future__ import annotations

import pytest

from codex.code.work.runner.budgeting import (
    BudgetBreach,
    BudgetChargeOutcome,
    BudgetMode,
    BudgetScope,
    BudgetSpec,
    CostSnapshot,
)
from codex.code.work.runner.manager import BudgetManager
from codex.code.work.runner.trace import TraceEventEmitter


@pytest.fixture()
def emitter() -> TraceEventEmitter:
    return TraceEventEmitter()


@pytest.fixture()
def manager(emitter: TraceEventEmitter) -> BudgetManager:
    return BudgetManager(trace_emitter=emitter)


def make_run_budget(limit_ms: int, *, mode: BudgetMode = BudgetMode.HARD) -> BudgetSpec:
    return BudgetSpec(
        scope=BudgetScope(scope_type="run", identifier="run-1"),
        limit=CostSnapshot(milliseconds=limit_ms, tokens_in=0, tokens_out=0, calls=0),
        mode=mode,
        breach_action="stop" if mode is BudgetMode.HARD else "warn",
    )


def make_node_budget(limit_ms: int, *, mode: BudgetMode = BudgetMode.HARD) -> BudgetSpec:
    return BudgetSpec(
        scope=BudgetScope(scope_type="node", identifier="node-1"),
        limit=CostSnapshot(milliseconds=limit_ms, tokens_in=0, tokens_out=0, calls=0),
        mode=mode,
        breach_action="stop" if mode is BudgetMode.HARD else "warn",
    )


def test_preflight_blocks_hard_budget(manager: BudgetManager) -> None:
    manager.register(make_run_budget(100))
    estimate = CostSnapshot.from_seconds(seconds=0.2)

    decision = manager.preflight(
        [BudgetScope(scope_type="run", identifier="run-1")], estimate
    )

    assert decision.blocked is True
    assert decision.hard_breaches == (
        BudgetBreach(
            scope=BudgetScope(scope_type="run", identifier="run-1"),
            mode=BudgetMode.HARD,
            action="stop",
            overage=CostSnapshot(milliseconds=100, tokens_in=0, tokens_out=0, calls=0),
        ),
    )


def test_commit_emits_charge_and_updates_remaining(
    manager: BudgetManager, emitter: TraceEventEmitter
) -> None:
    manager.register(make_run_budget(250))
    manager.register(make_node_budget(150))
    charge = CostSnapshot(milliseconds=120, tokens_in=10, tokens_out=5, calls=1)

    outcomes = manager.commit(
        [
            BudgetScope(scope_type="run", identifier="run-1"),
            BudgetScope(scope_type="node", identifier="node-1"),
        ],
        charge,
    )

    assert [out.scope for out in outcomes] == [
        BudgetScope(scope_type="run", identifier="run-1"),
        BudgetScope(scope_type="node", identifier="node-1"),
    ]
    run_outcome = outcomes[0]
    assert isinstance(run_outcome, BudgetChargeOutcome)
    assert run_outcome.spent == charge
    assert run_outcome.remaining == CostSnapshot(milliseconds=130, tokens_in=0, tokens_out=0, calls=0)
    assert emitter.events[-2].event == "budget_charge"
    assert emitter.events[-1].event == "budget_remaining"


def test_soft_budget_warns_without_blocking(
    manager: BudgetManager, emitter: TraceEventEmitter
) -> None:
    soft_budget = BudgetSpec(
        scope=BudgetScope(scope_type="run", identifier="run-1"),
        limit=CostSnapshot(milliseconds=90, tokens_in=0, tokens_out=0, calls=0),
        mode=BudgetMode.SOFT,
        breach_action="warn",
    )
    manager.register(soft_budget)
    estimate = CostSnapshot(milliseconds=100, tokens_in=0, tokens_out=0, calls=0)

    decision = manager.preflight([soft_budget.scope], estimate)

    assert decision.blocked is False
    assert decision.soft_breaches == (
        BudgetBreach(
            scope=soft_budget.scope,
            mode=BudgetMode.SOFT,
            action="warn",
            overage=CostSnapshot(milliseconds=10, tokens_in=0, tokens_out=0, calls=0),
        ),
    )

    manager.commit([soft_budget.scope], estimate)
    assert emitter.events[-1].event == "budget_warning"
    payload = emitter.events[-1].payload
    assert payload["scope_type"] == "run"
    assert payload["scope_id"] == "run-1"
    assert payload["overage"]["milliseconds"] == 10


def test_commit_raises_for_unregistered_scope(manager: BudgetManager) -> None:
    with pytest.raises(KeyError):
        manager.commit([BudgetScope(scope_type="node", identifier="missing")], CostSnapshot.zero())


def test_cost_snapshot_normalization() -> None:
    snapshot = CostSnapshot.from_seconds(seconds=1.5, tokens_in=2, tokens_out=4, calls=3)
    assert snapshot.milliseconds == 1500
    assert snapshot.tokens_in == 2
    assert snapshot.calls == 3
    assert snapshot.to_dict()["milliseconds"] == 1500
