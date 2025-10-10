"""Cost normalization utilities for the Phase 3 budget integration sandbox.

This module synthesizes the deterministic cost arithmetic from
`codex/implement-budget-guards-with-test-first-approach-8wxk32` with the
millisecond storage discipline adopted in
`codex/implement-budget-guards-with-test-first-approach-fa0vm9`. The helper
exposes a single entry-point that converts heterogeneous inputs (seconds,
milliseconds, integer counts) into an immutable mapping suitable for the
BudgetManager.
"""
from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import MutableMapping


def _ensure_mapping(cost: Mapping[str, float | int]) -> MutableMapping[str, float]:
    result: MutableMapping[str, float] = {}
    for key, value in cost.items():
        if not isinstance(value, (int, float)):
            raise TypeError(f"Cost value for {key!r} must be numeric, got {type(value)!r}")
        numeric = float(value)
        if numeric < 0:
            raise ValueError(f"Cost value for {key!r} must be non-negative")
        result[key] = numeric
    return result


def normalize_cost(cost: Mapping[str, float | int]) -> Mapping[str, float]:
    """Normalise a raw cost mapping into canonical units and freeze the payload."""

    materialised = _ensure_mapping(cost)
    total: MutableMapping[str, float] = {}

    for key, value in materialised.items():
        if key == "time_seconds":
            total["time_ms"] = total.get("time_ms", 0.0) + value * 1000.0
        elif key == "time_ms":
            total["time_ms"] = total.get("time_ms", 0.0) + value
        else:
            total[key] = total.get(key, 0.0) + value

    return MappingProxyType({name: float(amount) for name, amount in total.items()})
