from __future__ import annotations

import importlib.util
import pathlib
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .budget import BudgetChargeOutcome, CostSnapshot
else:
    _MODULE_DIR = pathlib.Path(__file__).resolve().parent

    def _load_budget_module():
        path = _MODULE_DIR / "budget.py"
        spec = importlib.util.spec_from_file_location(
            "task07b_budget", path
        )
        assert spec.loader is not None  # pragma: no cover - defensive
        if spec.name in sys.modules:
            module = sys.modules[spec.name]
        else:
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
        return module

    _budget = _load_budget_module()
    CostSnapshot = _budget.CostSnapshot
    BudgetChargeOutcome = _budget.BudgetChargeOutcome

__all__ = ["TraceEvent", "TraceWriter", "TraceEventEmitter"]


@dataclass(frozen=True, slots=True)
class TraceEvent:
    event: str
    scope_type: str
    scope_id: str
    payload: Mapping[str, object]


class TraceWriter(Protocol):
    def emit(self, event: "TraceEvent") -> None:  # pragma: no cover - protocol
        ...


class TraceEventEmitter:
    """Helper that formats structured events for policy and budget pipelines."""

    def __init__(self, writer: TraceWriter) -> None:
        self._writer = writer

    def emit_budget_charge(self, outcome: "BudgetChargeOutcome") -> None:
        payload = {
            "cost": _snapshot_payload(outcome.cost),
            "spent": _snapshot_payload(outcome.spent),
            "remaining": _snapshot_payload(outcome.remaining),
            "overages": _snapshot_payload(outcome.overages),
        }
        if outcome.breach is not None:
            payload["breach_kind"] = outcome.breach.kind
            payload["breach_action"] = outcome.breach.action
        self._writer.emit(
            TraceEvent(
                event="budget_charge",
                scope_type=outcome.scope_type,
                scope_id=outcome.scope_id,
                payload=_freeze(payload),
            )
        )

    def emit_budget_breach(self, outcome: "BudgetChargeOutcome") -> None:
        breach = outcome.breach
        if breach is None:  # pragma: no cover - defensive
            return
        payload = {
            "breach_kind": breach.kind,
            "breach_action": breach.action,
            "overages": _snapshot_payload(breach.overages),
            "stop_reason": breach.stop_reason,
        }
        self._writer.emit(
            TraceEvent(
                event="budget_breach",
                scope_type=breach.scope_type,
                scope_id=breach.scope_id,
                payload=_freeze(payload),
            )
        )

    def emit_loop_summary(
        self,
        *,
        loop_id: str,
        iteration: int,
        stop_reason: str,
        remaining_iterations: int,
    ) -> None:
        payload = {
            "iteration": iteration,
            "stop_reason": stop_reason,
            "remaining_iterations": remaining_iterations,
        }
        self._writer.emit(
            TraceEvent(
                event="loop_summary",
                scope_type="loop",
                scope_id=loop_id,
                payload=_freeze(payload),
            )
        )


def _snapshot_payload(snapshot: "CostSnapshot") -> Mapping[str, float]:
    return MappingProxyType(dict(snapshot.metrics))


def _freeze(payload: Mapping[str, object] | dict[str, object]) -> Mapping[str, object]:
    data = dict(payload)
    for key, value in list(data.items()):
        if isinstance(value, Mapping):
            data[key] = MappingProxyType(dict(value))
    return MappingProxyType(data)
