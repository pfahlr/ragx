"""BudgetManager coordinating hierarchical scope enforcement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

from .budget_models import BudgetChargeOutcome, BudgetScope, BudgetSpec, CostSnapshot

__all__ = ["BudgetBreachError", "BudgetManager"]


@dataclass
class BudgetBreachError(RuntimeError):
    """Raised when a hard budget is exceeded."""

    scope: BudgetScope
    outcome: BudgetChargeOutcome
    outcomes: Sequence[BudgetChargeOutcome] | None = None

    def __post_init__(self) -> None:
        message = (
            f"Budget breached at {self.scope}: action={self.outcome.action} overages={dict(self.outcome.overages)}"
        )
        super().__init__(message)


class BudgetManager:
    """Manages budget scopes and evaluates charges."""

    def __init__(self, *, budgets: Mapping[BudgetScope, BudgetSpec]):
        self._specs: Dict[BudgetScope, BudgetSpec] = dict(budgets)
        self._spent: MutableMapping[BudgetScope, CostSnapshot] = {
            scope: CostSnapshot.zero() for scope in self._specs
        }

    def preview(self, scopes: Sequence[BudgetScope], cost: CostSnapshot) -> List[BudgetChargeOutcome]:
        """Return outcomes without mutating internal state."""

        outcomes: List[BudgetChargeOutcome] = []
        for scope in scopes:
            spec = self._require_scope(scope)
            spent = self._spent.get(scope, CostSnapshot.zero()).add(cost)
            outcomes.append(
                BudgetChargeOutcome.build(scope=scope, spec=spec, cost=cost, spent=spent)
            )
        return outcomes

    def charge(self, scopes: Sequence[BudgetScope], cost: CostSnapshot) -> List[BudgetChargeOutcome]:
        """Apply a cost to multiple scopes and enforce hard breaches."""

        outcomes: List[BudgetChargeOutcome] = []
        hard_breach: BudgetBreachError | None = None
        for scope in scopes:
            spec = self._require_scope(scope)
            current = self._spent.get(scope, CostSnapshot.zero())
            updated = current.add(cost)
            self._spent[scope] = updated
            outcome = BudgetChargeOutcome.build(scope=scope, spec=spec, cost=cost, spent=updated)
            outcomes.append(outcome)
            if outcome.breached and outcome.action == "error" and hard_breach is None:
                hard_breach = BudgetBreachError(scope=scope, outcome=outcome)
        if hard_breach is not None:
            hard_breach.outcomes = tuple(outcomes)
            raise hard_breach
        return outcomes

    def reset_loop(self, scope: BudgetScope) -> None:
        """Reset loop spend when a new loop instance starts."""

        if scope in self._specs:
            self._spent[scope] = CostSnapshot.zero()

    def _require_scope(self, scope: BudgetScope) -> BudgetSpec:
        if scope not in self._specs:
            raise KeyError(f"Unknown budget scope: {scope}")
        return self._specs[scope]

    def spent(self, scope: BudgetScope) -> CostSnapshot:
        return self._spent.get(scope, CostSnapshot.zero())
