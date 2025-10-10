"""Cost normalization utilities for FlowRunner budget enforcement."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping

_SECONDS_KEYS = {"time_s", "seconds", "time_seconds"}
_MILLISECONDS_KEYS = {"time_ms", "milliseconds", "millis"}
_CANONICAL_TIME_KEY = "time_ms"


def normalize_cost(cost: Mapping[str, float | int]) -> Dict[str, float]:
    """Normalize a raw cost mapping to canonical metric names and floats.

    * Seconds-based time fields are converted to milliseconds and stored under ``time_ms``.
    * Millisecond fields are copied verbatim to ``time_ms``.
    * All numeric values are coerced to ``float`` for deterministic arithmetic.
    * Unknown keys are preserved to remain forward compatible with the DSL spec.
    """

    normalized: Dict[str, float] = {}
    for key, value in cost.items():
        numeric = float(value)
        if key in _SECONDS_KEYS:
            normalized[_CANONICAL_TIME_KEY] = numeric * 1000.0
        elif key in _MILLISECONDS_KEYS:
            normalized[_CANONICAL_TIME_KEY] = numeric
        else:
            normalized[key] = numeric
    return normalized


def combine_costs(costs: Iterable[Mapping[str, float]]) -> Dict[str, float]:
    """Aggregate multiple cost mappings into a single dictionary."""

    result: Dict[str, float] = {}
    for cost in costs:
        for key, value in cost.items():
            result[key] = result.get(key, 0.0) + float(value)
    return result


def subtract_costs(minuend: Mapping[str, float], subtrahend: Mapping[str, float]) -> Dict[str, float]:
    """Subtract one cost mapping from another (``minuend - subtrahend``)."""

    remaining: Dict[str, float] = {}
    for key, value in minuend.items():
        remaining[key] = float(value) - float(subtrahend.get(key, 0.0))
    return remaining
