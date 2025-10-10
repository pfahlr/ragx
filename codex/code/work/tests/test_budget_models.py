import pytest

from codex.code.work.dsl import budget_models as bm


class TestCostSnapshot:
    def test_from_raw_normalizes_seconds_and_tokens(self) -> None:
        snapshot = bm.CostSnapshot.from_raw({"time_s": 0.5, "tokens": 42})
        assert snapshot.time_ms == pytest.approx(500.0)
        assert snapshot.tokens == 42

    def test_addition_is_immutable(self) -> None:
        base = bm.CostSnapshot.from_raw({"time_ms": 100})
        inc = bm.CostSnapshot.from_raw({"time_ms": 50, "tokens": 10})
        total = base + inc
        assert total.time_ms == pytest.approx(150.0)
        assert total.tokens == 10
        # Ensure operands unchanged
        assert base.time_ms == pytest.approx(100.0)
        assert base.tokens == 0
        assert inc.tokens == 10

    def test_subtraction_and_floor_zero(self) -> None:
        lhs = bm.CostSnapshot.from_raw({"time_ms": 120, "tokens": 30})
        rhs = bm.CostSnapshot.from_raw({"time_ms": 200, "tokens": 35})
        diff = lhs - rhs
        assert diff.time_ms == 0.0
        assert diff.tokens == 0


class TestBudgetChargeOutcome:
    def test_overage_and_remaining(self) -> None:
        spec = bm.BudgetSpec(
            name="node",
            scope_type="node",
            limit=bm.CostSnapshot.from_raw({"time_ms": 120}),
            mode="soft",
            breach_action="warn",
        )
        prior = bm.CostSnapshot.from_raw({"time_ms": 100})
        cost = bm.CostSnapshot.from_raw({"time_ms": 60})
        outcome = bm.BudgetChargeOutcome.compute(spec=spec, prior=prior, cost=cost)
        assert outcome.charge.remaining.time_ms == pytest.approx(-40.0)
        assert outcome.charge.overage.time_ms == pytest.approx(40.0)
        assert outcome.breached is True
        assert outcome.should_stop is False
        payload = outcome.to_trace_payload(scope_type="node", scope_id="node-1")
        assert payload["spec_name"] == "node"
        with pytest.raises(TypeError):
            payload["spec_name"] = "mutate"  # type: ignore[index]

    def test_hard_stop_detection(self) -> None:
        spec = bm.BudgetSpec(
            name="run",
            scope_type="run",
            limit=bm.CostSnapshot.from_raw({"time_ms": 200}),
            mode="hard",
            breach_action="stop",
        )
        outcome = bm.BudgetChargeOutcome.compute(
            spec=spec,
            prior=bm.CostSnapshot.zero(),
            cost=bm.CostSnapshot.from_raw({"time_ms": 250}),
        )
        assert outcome.breached is True
        assert outcome.should_stop is True


class TestBudgetDecision:
    def test_decision_identifies_blocking_outcome(self) -> None:
        spec_soft = bm.BudgetSpec(
            name="soft",
            scope_type="run",
            limit=bm.CostSnapshot.from_raw({"time_ms": 100}),
            mode="soft",
            breach_action="warn",
        )
        spec_hard = bm.BudgetSpec(
            name="hard",
            scope_type="run",
            limit=bm.CostSnapshot.from_raw({"time_ms": 80}),
            mode="hard",
            breach_action="stop",
        )
        prior = bm.CostSnapshot.from_raw({"time_ms": 60})
        cost = bm.CostSnapshot.from_raw({"time_ms": 30})
        soft_outcome = bm.BudgetChargeOutcome.compute(spec=spec_soft, prior=prior, cost=cost)
        hard_outcome = bm.BudgetChargeOutcome.compute(spec=spec_hard, prior=prior, cost=cost)
        decision = bm.BudgetDecision.make(
            scope=bm.ScopeKey(scope_type="run", scope_id="run-1"),
            cost=cost,
            outcomes=[soft_outcome, hard_outcome],
        )
        assert decision.breached is True
        assert decision.should_stop is True
        assert decision.blocking is hard_outcome
