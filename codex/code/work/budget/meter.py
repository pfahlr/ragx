"""Budget metering utilities for run/node/loop scopes."""

from __future__ import annotations

from typing import Iterable

from .models import (
    BudgetCharge,
    BudgetChargeOutcome,
    BudgetSpec,
    CostSnapshot,
    overage_after,
    remaining_after,
)

__all__ = ["BudgetMeter"]


def _has_positive(values: Iterable[float]) -> bool:
    return any(value > 1e-9 for value in values)


class BudgetMeter:
    """Track spend against a :class:`BudgetSpec`."""

    def __init__(self, *, scope_type: str, scope_id: str, spec: BudgetSpec) -> None:
        self.scope_type = scope_type
        self.scope_id = scope_id
        self.spec = spec
        self._spent = CostSnapshot.zero()

    @property
    def spent(self) -> CostSnapshot:
        return self._spent

    def preview(self, cost: CostSnapshot) -> BudgetChargeOutcome:
        return self._build_outcome(cost, mutate=False)

    def commit(self, cost: CostSnapshot) -> BudgetChargeOutcome:
        outcome = self._build_outcome(cost, mutate=True)
        self._spent = outcome.charge.spent_after
        return outcome

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_outcome(self, cost: CostSnapshot, *, mutate: bool) -> BudgetChargeOutcome:
        spent_before = self._spent
        spent_after = spent_before.add(cost)
        remaining = remaining_after(self.spec.limit, spent_after)
        overage = overage_after(self.spec.limit, spent_after)
        breached = _has_positive(overage.values())
        stop = False
        reasons: tuple[str, ...]
        if breached:
            if self.spec.mode == "hard" or self.spec.breach_action == "stop":
                stop = True
                reasons = ("budget_stop",)
            else:
                reasons = ("budget_warn",)
        else:
            reasons = ()

        charge = BudgetCharge(
            scope_type=self.scope_type,
            scope_id=self.scope_id,
            spec=self.spec,
            cost=cost,
            spent_before=spent_before,
            spent_after=spent_after,
            remaining=remaining,
            overage=overage,
        )
        outcome = BudgetChargeOutcome(
            charge=charge,
            breached=breached,
            stop=stop,
            reasons=reasons,
        )
        if mutate:
            self._spent = spent_after
        return outcome
