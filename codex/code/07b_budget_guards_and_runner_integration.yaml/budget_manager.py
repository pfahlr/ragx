"""BudgetManager orchestrating scope lifecycles and breach handling."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from .budget_models import (
    BudgetCharge,
    BudgetMode,
    BudgetPreview,
    BudgetSpec,
    CostSnapshot,
    mapping_proxy,
)
from .costs import normalize_costs
from .trace_emitter import TraceEventEmitter

__all__ = ["BudgetManager", "BudgetBreachError"]


class BudgetBreachError(RuntimeError):
    """Raised when a hard-stop budget is breached."""

    def __init__(self, preview: BudgetPreview) -> None:
        super().__init__(f"Budget breached for scope '{preview.scope_id}'")
        self.preview = preview


@dataclass(slots=True)
class _BudgetContext:
    scope_type: str
    scope_id: str
    parent_scope: str | None
    spec: BudgetSpec | None
    spent: CostSnapshot = field(default_factory=CostSnapshot.zero)

    def preview(self, cost: CostSnapshot) -> BudgetCharge:
        if self.spec is None:
            return BudgetCharge(
                scope_id=self.scope_id,
                scope_type=self.scope_type,
                mode=BudgetMode.UNLIMITED,
                cost=cost,
                spent=self.spent,
                remaining=mapping_proxy({}),
                overages=mapping_proxy({}),
                breached=False,
            )

        new_totals = self.spent.add(cost)
        remaining: dict[str, float] = {}
        overages: dict[str, float] = {}
        breached = False
        for metric, limit in self.spec.limits.items():
            spent_value = new_totals.metrics.get(metric, 0.0)
            remaining_value = max(limit - spent_value, 0.0)
            remaining[metric] = remaining_value
            overage = max(spent_value - limit, 0.0)
            if overage > 0:
                breached = True
                overages[metric] = overage
        return BudgetCharge(
            scope_id=self.scope_id,
            scope_type=self.scope_type,
            mode=self.spec.mode,
            cost=cost,
            spent=self.spent,
            remaining=mapping_proxy(remaining),
            overages=mapping_proxy(overages),
            breached=breached,
        )

    def apply(self, cost: CostSnapshot) -> None:
        self.spent = self.spent.add(cost)


class BudgetManager:
    """Manage budgets across hierarchical scopes."""

    def __init__(self, *, emitter: TraceEventEmitter | None = None) -> None:
        self._emitter = emitter
        self._contexts: dict[str, _BudgetContext] = {}
        self._order: list[str] = []

    # ------------------------------------------------------------------
    # Scope lifecycle
    # ------------------------------------------------------------------
    def enter_scope(
        self,
        *,
        scope_type: str,
        scope_id: str,
        spec: BudgetSpec | None,
        parent_scope: str | None = None,
    ) -> None:
        if scope_id in self._contexts:
            raise ValueError(f"scope '{scope_id}' already active")
        if parent_scope is None and self._order:
            parent_scope = self._order[-1]
        context = _BudgetContext(
            scope_type=scope_type,
            scope_id=scope_id,
            parent_scope=parent_scope,
            spec=spec,
        )
        self._contexts[scope_id] = context
        self._order.append(scope_id)

    def exit_scope(self, scope_id: str) -> None:
        if not self._order or self._order[-1] != scope_id:
            raise ValueError(f"scope '{scope_id}' is not the latest active scope")
        self._order.pop()
        self._contexts.pop(scope_id, None)

    # ------------------------------------------------------------------
    # Preview + commit lifecycle
    # ------------------------------------------------------------------
    def preview(self, scope_id: str, cost: Mapping[str, object]) -> BudgetPreview:
        if scope_id not in self._contexts:
            raise KeyError(f"scope '{scope_id}' not registered")
        snapshot = normalize_costs(cost)
        charges: list[BudgetCharge] = []
        for context in self._iter_chain(scope_id):
            charges.append(context.preview(snapshot))
        return BudgetPreview(scope_id=scope_id, charges=tuple(charges), cost=snapshot)

    def commit(
        self,
        preview: BudgetPreview,
        *,
        node_id: str,
        loop_iteration: int,
    ) -> None:
        for charge in preview.charges:
            context = self._contexts.get(charge.scope_id)
            if context is None:
                continue
            context.apply(preview.cost)
            if self._emitter is not None and charge.mode != BudgetMode.UNLIMITED:
                self._emitter.emit_budget_charge(
                    node_id=node_id,
                    loop_iteration=loop_iteration,
                    charge=charge,
                )
                if charge.breached:
                    self._emitter.emit_budget_breach(
                        node_id=node_id,
                        loop_iteration=loop_iteration,
                        preview=preview,
                        charge=charge,
                    )
        if preview.hard_breach:
            raise BudgetBreachError(preview)

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------
    @property
    def active_scope_ids(self) -> Iterable[str]:
        return tuple(reversed(self._order))

    def remaining_budget(self, scope_id: str) -> CostSnapshot:
        context = self._contexts.get(scope_id)
        if context is None or context.spec is None:
            return CostSnapshot.zero()
        remaining = {
            metric: max(limit - context.spent.metrics.get(metric, 0.0), 0.0)
            for metric, limit in context.spec.limits.items()
        }
        return CostSnapshot(metrics=remaining)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _iter_chain(self, scope_id: str) -> Iterable[_BudgetContext]:
        current = self._contexts[scope_id]
        while True:
            yield current
            if current.parent_scope is None:
                break
            current = self._contexts[current.parent_scope]

