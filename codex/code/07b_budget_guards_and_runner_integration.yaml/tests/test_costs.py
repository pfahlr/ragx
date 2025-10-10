"""Unit tests for cost normalization utilities."""

from __future__ import annotations

import decimal

import pytest

from . import load_module

costs = load_module("costs")
budget_models = load_module("budget_models")

normalize_costs = costs.normalize_costs
CostSnapshot = budget_models.CostSnapshot


class TestNormalizeCosts:
    def test_converts_seconds_to_milliseconds(self) -> None:
        snapshot = normalize_costs({"time_seconds": 1.5, "tokens": 10})
        assert isinstance(snapshot, CostSnapshot)
        assert snapshot.metrics["time_ms"] == pytest.approx(1500.0)
        assert snapshot.metrics["tokens"] == pytest.approx(10.0)

    def test_accepts_decimal_inputs(self) -> None:
        snapshot = normalize_costs({"time_seconds": decimal.Decimal("0.250"), "calls": 2})
        assert snapshot.metrics["time_ms"] == pytest.approx(250.0)
        assert snapshot.metrics["calls"] == pytest.approx(2.0)

    def test_rejects_negative_values(self) -> None:
        with pytest.raises(ValueError):
            normalize_costs({"tokens": -1})

    def test_rejects_non_numeric(self) -> None:
        with pytest.raises(TypeError):
            normalize_costs({"tokens": "a lot"})

    def test_preserves_unknown_metrics_without_conversion(self) -> None:
        snapshot = normalize_costs({"latency_ms": 42})
        assert snapshot.metrics["latency_ms"] == pytest.approx(42.0)

