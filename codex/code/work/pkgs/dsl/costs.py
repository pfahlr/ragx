"""Utilities for normalising cost metrics used by the FlowRunner budget pipeline."""

from __future__ import annotations

from typing import Dict, Mapping

_ALLOWED_METRICS = {
    "duration_ms",
    "calls",
    "tokens_in",
    "tokens_out",
    "tokens_total",
}


def _coerce_numeric(value: float | int) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Cost metric must be numeric, got {value!r}") from exc
    if numeric < 0:
        raise ValueError("Cost metrics cannot be negative")
    return numeric


def normalize_cost(cost: Mapping[str, float | int]) -> Dict[str, int]:
    """Normalise a raw cost mapping.

    * Accepts `seconds` and converts to integer milliseconds (`duration_ms`).
    * Validates metric names against the allowed schema.
    * Returns a dictionary with integer values where applicable for determinism.
    """

    normalized: Dict[str, int] = {}
    duration: float | None = None

    for key, value in cost.items():
        if key == "seconds":
            duration = _coerce_numeric(value) * 1000.0
            continue
        if key not in _ALLOWED_METRICS:
            raise ValueError(f"Unsupported cost metric: {key}")
        normalized[key] = int(round(_coerce_numeric(value)))

    if duration is not None:
        normalized["duration_ms"] = int(round(duration))

    return normalized


def accumulate_cost(base: Mapping[str, int], delta: Mapping[str, int]) -> Dict[str, int]:
    """Add two cost dictionaries while preserving allowed metric constraints."""

    combined: Dict[str, int] = {key: int(value) for key, value in base.items()}
    for key, value in delta.items():
        if key not in _ALLOWED_METRICS:
            raise ValueError(f"Unsupported cost metric during accumulation: {key}")
        combined[key] = combined.get(key, 0) + int(value)
    return combined
