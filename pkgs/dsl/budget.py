"""Budget metering utilities for the FlowRunner."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .models import mapping_proxy

__all__ = [
    "BudgetError",
    "BudgetBreachHard",
    "BudgetCharge",
    "BudgetMeter",
]


class BudgetError(RuntimeError):
    """Raised when budget configuration or enforcement fails."""


class BudgetBreachHard(BudgetError):
    """Raised when a hard budget is exceeded."""

    def __init__(self, scope: str, overages: Mapping[str, float]) -> None:
        formatted = ", ".join(
            f"{field}:+{amount:.4g}"
            for field, amount in overages.items()
            if amount > 0
        ) or "no remaining capacity"
        super().__init__(f"Budget exceeded for {scope}: {formatted}")
        self.scope = scope
        self.overages = mapping_proxy(
            {field: float(amount) for field, amount in overages.items() if amount > 0}
        )


@dataclass(frozen=True, slots=True)
class BudgetCharge:
    """Result from charging a budget meter."""

    cost: Mapping[str, float]
    remaining: Mapping[str, float | None]
    overages: Mapping[str, float]
    breached: bool
    mode: str


class BudgetMeter:
    """Track spend for run, loop, or node budgets."""

    _FIELD_KEYS = ("usd", "tokens", "calls", "time_ms")
    _CONFIG_MAP: dict[str, tuple[str, Callable[[Any], float]]] = {
        "max_usd": ("usd", float),
        "max_tokens": ("tokens", float),
        "max_calls": ("calls", float),
        "time_limit_sec": ("time_ms", lambda value: float(value) * 1000.0),
        "time_limit_ms": ("time_ms", float),
    }
    _COST_KEYS: dict[str, str] = {
        "usd": "usd",
        "tokens": "tokens",
        "calls": "calls",
        "time_ms": "time_ms",
    }

    def __init__(
        self,
        *,
        kind: str,
        subject: str,
        config: Mapping[str, Any] | None = None,
        mode: str | None = None,
        breach_action: str | None = None,
    ) -> None:
        self._kind = kind
        self._subject = subject
        raw_config = dict(config or {})
        self._mode = (mode or raw_config.get("mode") or "hard").lower()
        self._breach_action = (
            breach_action or raw_config.get("breach_action") or "error"
        ).lower()

        self._caps: dict[str, float | None] = {key: None for key in self._FIELD_KEYS}
        for key, value in raw_config.items():
            mapped = self._CONFIG_MAP.get(key)
            if mapped is None:
                continue
            name, transform = mapped
            numeric = transform(value)
            self._caps[name] = None if numeric <= 0 else float(numeric)

        self._spent: dict[str, float] = {key: 0.0 for key in self._FIELD_KEYS}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def kind(self) -> str:
        return self._kind

    @property
    def subject(self) -> str:
        return self._subject

    @property
    def scope(self) -> str:
        return f"{self._kind}:{self._subject}"

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def breach_action(self) -> str:
        return self._breach_action

    @property
    def exceeded(self) -> bool:
        return any(amount > 0 for amount in self.overages().values())

    # ------------------------------------------------------------------
    # Budget operations
    # ------------------------------------------------------------------
    def can_spend(self, cost: Mapping[str, float | int]) -> bool:
        """Return True if the cost fits under the configured caps."""

        if self._mode == "soft":
            return True

        normalized = self._normalize_cost(cost)
        for key, amount in normalized.items():
            cap = self._caps.get(key)
            if cap is None:
                continue
            if self._spent[key] + amount > cap:
                return False
        return True

    def charge(self, cost: Mapping[str, float | int]) -> BudgetCharge:
        """Record cost against the meter and raise on hard breaches."""

        normalized = self._normalize_cost(cost)
        for key, amount in normalized.items():
            self._spent[key] += amount

        overages = self.overages()
        breached = any(amount > 0 for amount in overages.values())
        remaining = self.remaining()

        if (
            breached
            and self._mode == "hard"
            and self._breach_action != "stop"
        ):
            raise BudgetBreachHard(self.scope, overages)

        return BudgetCharge(
            cost=mapping_proxy({key: normalized[key] for key in self._FIELD_KEYS}),
            remaining=remaining,
            overages=overages,
            breached=breached,
            mode=self._mode,
        )

    def remaining(self) -> Mapping[str, float | None]:
        """Return remaining capacity for each tracked field."""

        remaining: dict[str, float | None] = {}
        for key, cap in self._caps.items():
            if cap is None:
                remaining[key] = None
            else:
                remaining[key] = cap - self._spent[key]
        return mapping_proxy(remaining)

    def overages(self) -> Mapping[str, float]:
        """Return positive overages for each field (0 otherwise)."""

        return mapping_proxy(
            {
                key: max(self._spent[key] - cap, 0.0) if cap is not None else 0.0
                for key, cap in self._caps.items()
            }
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _normalize_cost(self, cost: Mapping[str, float | int]) -> dict[str, float]:
        normalized = {key: 0.0 for key in self._FIELD_KEYS}
        for key, raw_value in cost.items():
            try:
                field = self._COST_KEYS[key]
            except KeyError as exc:  # pragma: no cover - defensive branch
                raise BudgetError(f"Unsupported cost key: {key!r}") from exc
            normalized[field] += float(raw_value)
        return normalized
