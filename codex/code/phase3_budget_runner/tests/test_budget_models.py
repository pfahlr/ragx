import pytest

from codex.code.phase3_budget_runner.dsl import budget_models as bm
from codex.code.phase3_budget_runner.dsl.trace import TraceEventEmitter


def test_cost_snapshot_from_raw_normalises_seconds_and_tokens():
    snapshot = bm.CostSnapshot.from_raw({"time_s": 1, "time_ms": 250, "tokens": "3"})
    assert snapshot.time_ms == pytest.approx(1250.0)
    assert snapshot.tokens == 3

    zero = bm.CostSnapshot.from_raw(None)
    assert zero.time_ms == 0.0
    assert zero.tokens == 0


def test_cost_snapshot_arithmetic_clamps_subtraction():
    base = bm.CostSnapshot(time_ms=100.0, tokens=10)
    delta = bm.CostSnapshot(time_ms=40.0, tokens=4)
    total = base + delta
    assert total.time_ms == pytest.approx(140.0)
    assert total.tokens == 14

    reduced = base - bm.CostSnapshot(time_ms=150.0, tokens=20)
    assert reduced.time_ms == 0.0
    assert reduced.tokens == 0


@pytest.mark.parametrize(
    "mode, breach_action, should_stop",
    [
        ("hard", "stop", True),
        ("soft", "stop", True),
        ("soft", "warn", False),
    ],
)
def test_budget_charge_outcome_trace_payload(mode, breach_action, should_stop):
    spec = bm.BudgetSpec(
        name="latency",
        scope_type="run",
        limit=bm.CostSnapshot(time_ms=100.0, tokens=100),
        mode=mode,
        breach_action=breach_action,
    )
    prior = bm.CostSnapshot(time_ms=80.0, tokens=10)
    cost = bm.CostSnapshot(time_ms=40.0, tokens=5)
    outcome = bm.BudgetChargeOutcome.compute(spec=spec, prior=prior, cost=cost)

    assert outcome.breached is True
    assert outcome.should_stop is should_stop

    payload = outcome.to_trace_payload(scope_type="run", scope_id="run-1")
    assert payload["scope_type"] == "run"
    assert payload["scope_id"] == "run-1"
    assert payload["spec_name"] == "latency"
    assert payload["mode"] == mode
    assert payload["breach_action"] == breach_action
    assert payload["breached"] is True
    assert payload["should_stop"] is should_stop
    assert payload["overage"]["time_ms"] == pytest.approx(20.0)
    assert payload["overage"]["tokens"] == 0

    with pytest.raises(TypeError):
        payload["new"] = "value"  # immutable mapping proxy


def test_budget_decision_identifies_blocking_and_emits_traces():
    emitter = TraceEventEmitter()
    scope = bm.ScopeKey(scope_type="node", scope_id="n1")
    hard_spec = bm.BudgetSpec(
        name="compute",
        scope_type="node",
        limit=bm.CostSnapshot(time_ms=50.0, tokens=20),
        mode="hard",
        breach_action="stop",
    )
    soft_spec = bm.BudgetSpec(
        name="tokens",
        scope_type="node",
        limit=bm.CostSnapshot(time_ms=999.0, tokens=30),
        mode="soft",
        breach_action="warn",
    )
    prior = bm.CostSnapshot.zero()
    cost = bm.CostSnapshot(time_ms=60.0, tokens=25)
    outcomes = [
        bm.BudgetChargeOutcome.compute(spec=hard_spec, prior=prior, cost=cost),
        bm.BudgetChargeOutcome.compute(spec=soft_spec, prior=prior, cost=cost),
    ]
    decision = bm.BudgetDecision.make(scope=scope, cost=cost, outcomes=outcomes)

    assert decision.should_stop is True
    assert decision.blocking is outcomes[0]

    decision.to_trace_records(emitter=emitter, event="budget_charge")
    events = emitter.events
    assert len(events) == 2
    assert {event.payload["spec_name"] for event in events} == {"compute", "tokens"}

