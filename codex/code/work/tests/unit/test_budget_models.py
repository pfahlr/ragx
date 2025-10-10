import pytest

from pkgs.dsl.budget import (
    BudgetBreachError,
    BudgetMeter,
    BudgetMode,
    BudgetSpec,
    CostSnapshot,
)
from pkgs.dsl.trace import TraceEventEmitter


def test_budget_spec_normalization_and_defaults() -> None:
    spec = BudgetSpec.from_mapping(
        {
            "max_usd": "1.5",
            "max_calls": 3,
            "max_tokens": 1200,
            "time_limit_sec": 2.5,
            "mode": "soft",
        }
    )
    assert spec.mode is BudgetMode.SOFT
    assert spec.time_limit_ms == 2500
    assert spec.max_usd == pytest.approx(1.5)
    assert spec.max_calls == 3
    assert spec.max_tokens == 1200


def test_cost_snapshot_arithmetic_and_meter_commit() -> None:
    spec = BudgetSpec.from_mapping({"max_usd": 2.0, "mode": "hard"})
    meter = BudgetMeter(scope_type="run", scope_id="run", spec=spec)
    preview = meter.preview(CostSnapshot(usd=1.0))
    assert not preview.breached
    decision = meter.commit(CostSnapshot(usd=0.75))
    assert decision.spent.usd == pytest.approx(0.75)
    assert decision.remaining.usd == pytest.approx(1.25)


def test_budget_meter_hard_breach_raises() -> None:
    spec = BudgetSpec.from_mapping({"max_usd": 1.0, "mode": "hard"})
    meter = BudgetMeter(scope_type="run", scope_id="run", spec=spec)
    meter.commit(CostSnapshot(usd=0.6))
    with pytest.raises(BudgetBreachError):
        meter.commit(CostSnapshot(usd=0.5))


def test_budget_meter_soft_budget_records_warning() -> None:
    spec = BudgetSpec.from_mapping({"max_usd": 1.0, "mode": "soft"})
    meter = BudgetMeter(scope_type="node", scope_id="n1", spec=spec)
    decision = meter.commit(CostSnapshot(usd=1.2))
    assert decision.breached
    assert decision.warnings == ("budget_soft_limit_exceeded",)


def test_trace_event_emitter_returns_immutable_payload() -> None:
    emitter = TraceEventEmitter()
    event = emitter.emit(
        event="budget_commit",
        scope_type="run",
        scope_id="run",
        payload={"spent": {"usd": 1.0}},
    )
    assert event.event == "budget_commit"
    with pytest.raises(TypeError):
        event.payload["spent"] = {"usd": 2.0}
    with pytest.raises(TypeError):
        event.payload["spent"]["usd"] = 2.0
