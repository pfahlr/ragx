"""Tests for budget data models and meter invariants."""

from __future__ import annotations

import pytest

from ..budget.models import (
    BudgetChargeOutcome,
    BudgetSpec,
    CostSnapshot,
)
from ..budget.meter import BudgetMeter


class TestCostSnapshot:
    def test_from_mapping_normalizes_and_freezes(self) -> None:
        snapshot = CostSnapshot.from_mapping(
            {"calls": 1, "tokens_in": 32, "seconds": 1.25}
        )
        # Missing metrics default to zero and values are floats.
        assert snapshot.calls == pytest.approx(1.0)
        assert snapshot.tokens_out == pytest.approx(0.0)
        # Mapping is immutable
        mapping = snapshot.as_mapping()
        with pytest.raises(TypeError):
            mapping["calls"] = 2.0  # type: ignore[index]
        # Round trip preserves data
        rebuilt = CostSnapshot.from_mapping(mapping)
        assert rebuilt == snapshot

    def test_addition_and_subtraction_are_component_wise(self) -> None:
        base = CostSnapshot.from_mapping({"seconds": 1.5, "tokens_in": 10})
        extra = CostSnapshot.from_mapping({"seconds": 0.5, "tokens_out": 5})
        total = base.add(extra)
        assert total.seconds == pytest.approx(2.0)
        assert total.tokens_in == pytest.approx(10)
        assert total.tokens_out == pytest.approx(5)
        assert total.calls == pytest.approx(0)
        delta = total.subtract(extra)
        assert delta == base


class TestBudgetSpec:
    def test_from_mapping_applies_defaults(self) -> None:
        spec = BudgetSpec.from_mapping(
            scope="run",
            scope_id="global",
            data={"limit": {"seconds": 10}},
        )
        assert spec.mode == "soft"
        assert spec.breach_action == "warn"
        assert spec.limit.seconds == pytest.approx(10.0)
        assert spec.limit.tokens_in == pytest.approx(0.0)


class TestBudgetMeter:
    def test_preview_does_not_mutate_state(self) -> None:
        spec = BudgetSpec.from_mapping(
            scope="run",
            scope_id="global",
            data={"limit": {"seconds": 5}, "mode": "hard", "breach_action": "stop"},
        )
        meter = BudgetMeter(scope_type="run", scope_id="global", spec=spec)
        estimate = CostSnapshot.from_mapping({"seconds": 3})
        preview = meter.preview(estimate)
        assert isinstance(preview, BudgetChargeOutcome)
        assert preview.breached is False
        assert meter.spent.seconds == pytest.approx(0.0)
        # A breach preview leaves the meter untouched
        breach_preview = meter.preview(CostSnapshot.from_mapping({"seconds": 8}))
        assert breach_preview.breached is True
        assert breach_preview.stop is True
        assert meter.spent.seconds == pytest.approx(0.0)
        # Committing applies the spend
        outcome = meter.commit(estimate)
        assert outcome.breached is False
        assert meter.spent.seconds == pytest.approx(3.0)
