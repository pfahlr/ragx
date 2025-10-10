"""Integration tests for :mod:`codex.code.work.budget.manager`."""

from __future__ import annotations

from typing import Mapping

import pytest

from ..budget.manager import BudgetManager
from ..budget.models import BudgetSpec, CostSnapshot
from ..trace.emitter import TraceEventEmitter


def _register_run_budget(manager: BudgetManager, limit_seconds: float, *, mode: str = "hard", breach_action: str = "stop") -> None:
    manager.register_scope(
        scope_type="run",
        scope_id="global",
        spec=BudgetSpec.from_mapping(
            scope="run",
            scope_id="global",
            data={"limit": {"seconds": limit_seconds}, "mode": mode, "breach_action": breach_action},
        ),
    )


def _event_payloads(emitter: TraceEventEmitter) -> list[Mapping[str, object]]:
    return [event.payload for event in emitter.events]


class TestBudgetManager:
    def test_preflight_and_commit_emit_traces(self) -> None:
        emitter = TraceEventEmitter()
        manager = BudgetManager(emitter=emitter)
        _register_run_budget(manager, limit_seconds=5)

        estimate = CostSnapshot.from_mapping({"seconds": 2})
        preview = manager.preflight("run", "global", estimate, event_context={"stage": "estimate"})
        assert preview is not None
        assert preview.breached is False
        commit = manager.commit("run", "global", estimate, event_context={"stage": "execute"})
        assert commit is not None
        assert commit.breached is False
        assert commit.charge.spent_after.seconds == pytest.approx(2.0)

        payloads = _event_payloads(emitter)
        assert [event.event for event in emitter.events] == [
            "budget_preflight",
            "budget_charge",
        ]
        assert payloads[0]["context"] == {"stage": "estimate"}
        assert payloads[1]["breached"] is False

    def test_hard_stop_is_reported(self) -> None:
        emitter = TraceEventEmitter()
        manager = BudgetManager(emitter=emitter)
        _register_run_budget(manager, limit_seconds=3)

        manager.commit("run", "global", CostSnapshot.from_mapping({"seconds": 2}))
        breach = manager.preflight("run", "global", CostSnapshot.from_mapping({"seconds": 2}))
        assert breach is not None
        assert breach.stop is True
        assert breach.breached is True
        commit = manager.commit("run", "global", CostSnapshot.from_mapping({"seconds": 2}))
        assert commit is not None
        assert commit.stop is True
        assert commit.charge.overage["seconds"] == pytest.approx(1.0)

        breach_events = [event for event in emitter.events if event.event == "budget_breach"]
        assert len(breach_events) == 1
        payload = breach_events[0].payload
        assert payload["breached"] is True
        assert payload["stop"] is True

    def test_soft_budget_warns_but_does_not_stop(self) -> None:
        emitter = TraceEventEmitter()
        manager = BudgetManager(emitter=emitter)
        _register_run_budget(
            manager,
            limit_seconds=1,
            mode="soft",
            breach_action="warn",
        )

        outcome = manager.commit("run", "global", CostSnapshot.from_mapping({"seconds": 2}))
        assert outcome is not None
        assert outcome.breached is True
        assert outcome.stop is False
        warn_events = [event for event in emitter.events if event.event == "budget_breach"]
        assert len(warn_events) == 1
        assert warn_events[0].payload["stop"] is False
