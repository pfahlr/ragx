"""BudgetManager orchestration for FlowRunner scopes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

from .budget import (
    BudgetBreachError,
    BudgetDecision,
    BudgetMeter,
    BudgetMode,
    BudgetSpec,
    CostSnapshot,
)
from .trace import TraceEventEmitter

__all__ = ["BudgetManager", "BudgetScope"]


@dataclass(frozen=True, slots=True)
class BudgetScope:
    """Hashable identifier for a budget scope."""

    scope_type: str
    scope_id: str

    def key(self) -> tuple[str, str]:
        return (self.scope_type, self.scope_id)


class BudgetManager:
    """Coordinate budget meters across run/node/loop scopes."""

    def __init__(
        self,
        *,
        run_spec: BudgetSpec | Mapping[str, object] | None,
        trace: TraceEventEmitter | None = None,
    ) -> None:
        self._trace = trace or TraceEventEmitter()
        self._meters: Dict[tuple[str, str], BudgetMeter] = {}
        self._warnings: list[BudgetDecision] = []
        self._loop_stack: list[BudgetScope] = []
        if run_spec is not None:
            spec = self._normalize_spec(run_spec)
            self.configure_scope(BudgetScope("run", "run"), spec)

    # ------------------------------------------------------------------
    # Scope configuration
    # ------------------------------------------------------------------
    def configure_scope(
        self,
        scope: BudgetScope,
        spec: BudgetSpec | Mapping[str, object] | None,
    ) -> None:
        spec_obj = self._normalize_spec(spec)
        self._meters[scope.key()] = BudgetMeter(
            scope_type=scope.scope_type,
            scope_id=scope.scope_id,
            spec=spec_obj,
        )

    def push_loop(self, loop_id: str, spec: BudgetSpec | Mapping[str, object] | None) -> BudgetScope:
        scope = BudgetScope("loop", loop_id)
        self.configure_scope(scope, spec)
        self._loop_stack.append(scope)
        return scope

    def pop_loop(self, expected_id: str | None = None) -> None:
        if not self._loop_stack:
            raise RuntimeError("loop stack underflow")
        scope = self._loop_stack.pop()
        if expected_id is not None and scope.scope_id != expected_id:
            self._loop_stack.append(scope)
            raise RuntimeError(
                f"loop scope mismatch: expected {expected_id!r}, got {scope.scope_id!r}"
            )

    # ------------------------------------------------------------------
    # Budget evaluation
    # ------------------------------------------------------------------
    def preflight(self, scope: BudgetScope, cost: CostSnapshot) -> BudgetDecision:
        meter = self._ensure_meter(scope)
        decision = meter.preview(cost)
        self._emit_decision("budget_preflight", decision)
        if decision.breached and decision.mode is BudgetMode.HARD and not decision.should_stop:
            self._emit_breach(decision)
            raise BudgetBreachError(decision)
        return decision

    def commit(self, scope: BudgetScope, cost: CostSnapshot) -> BudgetDecision:
        meter = self._ensure_meter(scope)
        try:
            decision = meter.commit(cost)
        except BudgetBreachError as exc:
            self._emit_breach(exc.decision)
            raise
        else:
            self._emit_decision("budget_commit", decision)
            if decision.breached:
                if decision.mode is BudgetMode.SOFT:
                    self._warnings.append(decision)
                    self._emit_warning(decision)
                if decision.should_stop or decision.mode is BudgetMode.HARD:
                    self._emit_breach(decision)
            return decision

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    @property
    def warnings(self) -> Sequence[BudgetDecision]:
        return tuple(self._warnings)

    @property
    def loop_stack(self) -> Sequence[BudgetScope]:
        return tuple(self._loop_stack)

    @property
    def trace(self) -> TraceEventEmitter:
        return self._trace

    def has_scope(self, scope: BudgetScope) -> bool:
        return scope.key() in self._meters

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_meter(self, scope: BudgetScope) -> BudgetMeter:
        meter = self._meters.get(scope.key())
        if meter is None:
            self.configure_scope(scope, None)
            meter = self._meters[scope.key()]
        return meter

    def _normalize_spec(
        self, spec: BudgetSpec | Mapping[str, object] | None
    ) -> BudgetSpec:
        if isinstance(spec, BudgetSpec):
            return spec
        return BudgetSpec.from_mapping(spec)

    def _emit_decision(self, event: str, decision: BudgetDecision) -> None:
        self._trace.emit(
            event=event,
            scope_type=decision.scope_type,
            scope_id=decision.scope_id,
            payload={
                "stage": decision.stage,
                "breached": decision.breached,
                "mode": decision.mode.value,
                "breach_action": decision.breach_action,
                "cost": self._snapshot_payload(decision.cost),
                "spent": self._snapshot_payload(decision.spent),
                "remaining": self._snapshot_payload(decision.remaining),
                "overage": self._snapshot_payload(decision.overage),
            },
        )

    def _emit_warning(self, decision: BudgetDecision) -> None:
        self._trace.emit(
            event="budget_warning",
            scope_type=decision.scope_type,
            scope_id=decision.scope_id,
            payload={
                "warnings": list(decision.warnings),
                "breach_action": decision.breach_action,
            },
        )

    def _emit_breach(self, decision: BudgetDecision) -> None:
        self._trace.emit(
            event="budget_breach",
            scope_type=decision.scope_type,
            scope_id=decision.scope_id,
            payload={
                "breach_action": decision.breach_action,
                "mode": decision.mode.value,
            },
        )

    @staticmethod
    def _snapshot_payload(snapshot: CostSnapshot) -> Mapping[str, object]:
        return {
            "usd": snapshot.usd,
            "calls": snapshot.calls,
            "tokens": snapshot.tokens,
            "elapsed_ms": snapshot.elapsed_ms,
        }
