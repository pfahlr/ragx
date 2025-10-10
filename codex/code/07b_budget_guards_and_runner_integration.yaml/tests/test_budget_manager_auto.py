"""Additional regression coverage for BudgetManager behaviour."""

from __future__ import annotations

from itertools import product

import pytest

from pkgs.dsl import budget_models as bm
from pkgs.dsl.budget_manager import BudgetManager


@pytest.fixture()
def manager() -> BudgetManager:
    specs = [
        bm.BudgetSpec(
            name="run-hard",
            scope_type="run",
            limit=bm.CostSnapshot.from_raw({"time_ms": 120, "tokens": 400}),
            mode="hard",
            breach_action="stop",
        ),
        bm.BudgetSpec(
            name="loop-soft",
            scope_type="loop",
            limit=bm.CostSnapshot.from_raw({"time_ms": 60}),
            mode="soft",
            breach_action="stop",
        ),
        bm.BudgetSpec(
            name="node-soft",
            scope_type="node",
            limit=bm.CostSnapshot.from_raw({"time_ms": 40}),
            mode="soft",
            breach_action="warn",
        ),
        bm.BudgetSpec(
            name="spec-budget",
            scope_type="spec",
            limit=bm.CostSnapshot.from_raw({"time_ms": 90, "tokens": 300}),
            mode="hard",
            breach_action="stop",
        ),
        bm.BudgetSpec(
            name="time-only",
            scope_type="node",
            limit=bm.CostSnapshot.from_raw({"time_ms": 25}),
            mode="soft",
            breach_action="warn",
        ),
    ]
    return BudgetManager(specs=specs)


def _enter_all_scopes(
    manager: BudgetManager,
) -> tuple[bm.ScopeKey, bm.ScopeKey, bm.ScopeKey, bm.ScopeKey]:
    run = bm.ScopeKey(scope_type="run", scope_id="run-1")
    loop = bm.ScopeKey(scope_type="loop", scope_id="loop-1")
    node = bm.ScopeKey(scope_type="node", scope_id="node-1")
    spec = bm.ScopeKey(scope_type="spec", scope_id="budget-A")
    manager.enter_scope(run)
    manager.enter_scope(loop)
    manager.enter_scope(node)
    manager.enter_scope(spec)
    return run, loop, node, spec


def test_budget_manager_nested_scope_accounting(manager: BudgetManager) -> None:
    run, loop, node, spec = _enter_all_scopes(manager)

    cost = bm.CostSnapshot.from_raw({"time_ms": 20, "tokens": 100})
    decision = manager.preview_charge(node, cost)
    manager.commit_charge(decision)

    assert manager.spent(node, "node-soft").time_ms == pytest.approx(20)
    assert manager.spent(run, "run-hard").time_ms == pytest.approx(0)
    assert manager.spent(loop, "loop-soft").time_ms == pytest.approx(0)

    loop_cost = bm.CostSnapshot.from_raw({"time_ms": 50, "tokens": 0})
    loop_decision = manager.preview_charge(loop, loop_cost)
    manager.commit_charge(loop_decision)

    assert manager.spent(loop, "loop-soft").time_ms == pytest.approx(50)
    assert manager.spent(node, "node-soft").time_ms == pytest.approx(20)
    assert manager.spent(spec, "spec-budget").time_ms == pytest.approx(0)


def test_budget_manager_handles_spec_scope_and_partial_metrics(manager: BudgetManager) -> None:
    run, loop, node, spec = _enter_all_scopes(manager)

    mixed_cost = bm.CostSnapshot.from_raw({"time_ms": 30, "tokens": 120})
    node_decision = manager.preview_charge(node, mixed_cost)
    manager.commit_charge(node_decision)

    spec_decision = manager.preview_charge(spec, mixed_cost)
    manager.commit_charge(spec_decision)

    assert manager.spent(spec, "spec-budget").tokens == 120
    assert manager.spent(spec, "spec-budget").time_ms == pytest.approx(30)

    # time-only spec should treat tokens as unbounded while still tracking elapsed time
    time_only = manager.spent(node, "time-only")
    assert time_only.time_ms == pytest.approx(30)
    assert time_only.tokens == 120

    # Previewing with a new actual cost recomputes totals without mutating prior state
    followup_cost = bm.CostSnapshot.from_raw({"time_ms": 10, "tokens": 20})
    followup = manager.preview_charge(node, followup_cost)
    assert followup.blocking is None
    assert followup.cost == followup_cost
    manager.commit_charge(followup)

    combined = manager.spent(node, "node-soft")
    assert combined.time_ms == pytest.approx(40)
    assert combined.tokens == 140


def test_property_based_budget_charge_precision() -> None:
    spec = bm.BudgetSpec(
        name="precision",
        scope_type="node",
        limit=bm.CostSnapshot.from_raw({"time_ms": 100, "tokens": 1000}),
        mode="soft",
        breach_action="warn",
    )
    prior = bm.CostSnapshot.zero()
    for time_ms, tokens in product([0, 0.1, 1.5, 10.25, 50.5], [0, 1, 5, 10, 25]):
        cost = bm.CostSnapshot(time_ms=time_ms, tokens=tokens)
        outcome = bm.BudgetChargeOutcome.compute(spec=spec, prior=prior, cost=cost)
        charge = outcome.charge
        # Remaining + overage should equal the delta between limit and new total.
        assert charge.new_total.time_ms == pytest.approx(prior.time_ms + time_ms)
        assert charge.new_total.tokens == prior.tokens + tokens
        if charge.new_total.time_ms <= spec.limit.time_ms:
            assert charge.overage.time_ms == pytest.approx(0.0)
        else:
            over_time = charge.new_total.time_ms - spec.limit.time_ms
            assert charge.overage.time_ms == pytest.approx(over_time)
        if charge.new_total.tokens <= spec.limit.tokens:
            assert charge.overage.tokens == 0
        else:
            assert charge.overage.tokens == charge.new_total.tokens - spec.limit.tokens
        prior = charge.new_total
