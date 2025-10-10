import pytest

from dsl.costs import normalize_cost


def test_normalize_seconds_to_milliseconds_and_freeze_mapping():
    raw = {"time_seconds": 1.25, "tokens": 42}

    normalized = normalize_cost(raw)

    assert normalized["time_ms"] == pytest.approx(1250.0)
    assert normalized["tokens"] == pytest.approx(42.0)
    with pytest.raises(TypeError):
        normalized["tokens"] = 99  # type: ignore[misc]


def test_normalize_cost_rejects_negative_values():
    with pytest.raises(ValueError):
        normalize_cost({"time_seconds": -0.5})


def test_normalize_cost_preserves_existing_milliseconds():
    normalized = normalize_cost({"time_ms": 150.0})

    assert normalized["time_ms"] == pytest.approx(150.0)
    # idempotent call should not change data
    normalized_again = normalize_cost(normalized)
    assert normalized_again["time_ms"] == pytest.approx(150.0)
