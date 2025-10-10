"""Utilities for normalising cost metrics before applying budgets."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from decimal import Decimal

from .budget_models import CostSnapshot

_SECONDS_SUFFIX = "_seconds"


def _ensure_numeric(value: object) -> float:
    if isinstance(value, (int, float, Decimal)):
        numeric = float(value)
        if numeric < 0:
            raise ValueError("cost metrics must be non-negative")
        return numeric
    raise TypeError(f"cost metric must be numeric, got {type(value)!r}")


def normalize_costs(costs: Mapping[str, object]) -> CostSnapshot:
    """Normalise raw cost metrics into canonical units."""

    aggregated: dict[str, float] = defaultdict(float)
    for key, value in costs.items():
        numeric = _ensure_numeric(value)
        if key.endswith(_SECONDS_SUFFIX):
            normalized_key = f"{key[:-len(_SECONDS_SUFFIX)]}_ms"
            aggregated[normalized_key] += numeric * 1000.0
        else:
            aggregated[str(key)] += numeric
    return CostSnapshot(metrics=dict(aggregated))


__all__ = ["normalize_costs"]

