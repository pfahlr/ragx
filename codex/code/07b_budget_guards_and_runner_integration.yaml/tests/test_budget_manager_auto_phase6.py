"""Phase 6 regression tests for ``pkgs.dsl.budget_manager``."""

from __future__ import annotations

from collections import defaultdict
from itertools import product

import pytest

from pkgs.dsl import budget_models as bm
from pkgs.dsl.budget_manager import BudgetManager


@pytest.fixture()
def manager() -> BudgetManager:
    """Budget manager mirroring the phase 3 acceptance configuration."""

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
            breach_action="warn",
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
    ]
    return BudgetManager(specs=specs)


def _enter_scope_bundle(manager: BudgetManager) -> dict[str, bm.ScopeKey]:
    scopes = {
        "run": bm.ScopeKey(scope_type="run", scope_id="run-1"),
        "loop": bm.ScopeKey(scope_type="loop", scope_id="loop-1"),
        "node": bm.ScopeKey(scope_type="node", scope_id="node-1"),
        "spec": bm.ScopeKey(scope_type="spec", scope_id="budget-A"),
    }
    for scope in scopes.values():
        manager.enter_scope(scope)
    return scopes


def test_budget_manager_nested_scope_accounting(manager: BudgetManager) -> None:
    scopes = _enter_scope_bundle(manager)

    cost = bm.CostSnapshot.from_raw({"time_ms": 20, "tokens": 100})
    node_decision = manager.preview_charge(scopes["node"], cost)
    manager.commit_charge(node_decision)

    assert manager.spent(scopes["node"], "node-soft").time_ms == pytest.approx(20)
    assert manager.spent(scopes["run"], "run-hard").time_ms == pytest.approx(0)

    loop_cost = bm.CostSnapshot.from_raw({"time_ms": 45})
    loop_decision = manager.preview_charge(scopes["loop"], loop_cost)
    manager.commit_charge(loop_decision)

    assert manager.spent(scopes["loop"], "loop-soft").time_ms == pytest.approx(45)
    assert manager.spent(scopes["node"], "node-soft").tokens == 100


def test_budget_manager_handles_spec_scope_and_partial_metrics(
    manager: BudgetManager,
) -> None:
    scopes = _enter_scope_bundle(manager)

    cost = bm.CostSnapshot.from_raw({"time_ms": 30, "tokens": 120})
    node_decision = manager.preview_charge(scopes["node"], cost)
    manager.commit_charge(node_decision)

    spec_decision = manager.preview_charge(scopes["spec"], cost)
    manager.commit_charge(spec_decision)

    assert manager.spent(scopes["spec"], "spec-budget").tokens == 120
    assert manager.spent(scopes["spec"], "spec-budget").time_ms == pytest.approx(30)

    followup_cost = bm.CostSnapshot.from_raw({"time_ms": 10, "tokens": 20})
    followup = manager.preview_charge(scopes["node"], followup_cost)
    assert not followup.should_stop
    manager.commit_charge(followup)

    aggregate = manager.spent(scopes["node"], "node-soft")
    assert aggregate.time_ms == pytest.approx(40)
    assert aggregate.tokens == 140


def test_property_based_budget_charge_precision() -> None:
    spec = bm.BudgetSpec(
        name="precision",
        scope_type="node",
        limit=bm.CostSnapshot.from_raw({"time_ms": 100, "tokens": 1000}),
        mode="soft",
        breach_action="warn",
    )
    prior = defaultdict(float)
    running_total = bm.CostSnapshot.zero()
    for time_ms, tokens in product([0, 0.1, 1.5, 10.25, 50.5], [0, 1, 5, 10, 25]):
        cost = bm.CostSnapshot(time_ms=time_ms, tokens=tokens)
        outcome = bm.BudgetChargeOutcome.compute(
            spec=spec, prior=running_total, cost=cost
        )
        running_total = outcome.charge.new_total
        assert outcome.charge.new_total.time_ms == pytest.approx(
            prior["time_ms"] + time_ms
        )
        assert outcome.charge.new_total.tokens == prior["tokens"] + tokens
        prior["time_ms"] = outcome.charge.new_total.time_ms
        prior["tokens"] = outcome.charge.new_total.tokens
        if outcome.charge.new_total.time_ms <= spec.limit.time_ms:
            assert outcome.charge.overage.time_ms == pytest.approx(0.0)
        else:
            assert outcome.charge.overage.time_ms == pytest.approx(
                outcome.charge.new_total.time_ms - spec.limit.time_ms
            )
        if outcome.charge.new_total.tokens <= spec.limit.tokens:
            assert outcome.charge.overage.tokens == 0
        else:
            assert outcome.charge.overage.tokens == pytest.approx(
                outcome.charge.new_total.tokens - spec.limit.tokens
            )
