"""Property-based executable spec for envelope validation edge cases."""

from __future__ import annotations

from typing import Any

import pytest
from jsonschema import ValidationError

from apps.mcp_server.validation.schema_registry import SchemaRegistry

hypothesis = pytest.importorskip("hypothesis")
given = hypothesis.given
st = hypothesis.strategies

if not hasattr(st, "booleans"):
    pytest.skip(
        "Hypothesis strategies module lacks required APIs; skipping property tests",
        allow_module_level=True,
    )

_ENVELOPE_VALIDATOR = SchemaRegistry().load_envelope()


def _fixed_dict(
    mapping: dict[str, st.SearchStrategy[Any]]
) -> st.SearchStrategy[dict[str, Any]]:
    if hasattr(st, "fixed_dictionaries"):
        return st.fixed_dictionaries(mapping)  # type: ignore[attr-defined]
    return st.builds(lambda **kwargs: kwargs, **mapping)


def _meta_strategy() -> st.SearchStrategy[dict[str, Any]]:
    required_fields = {
        "requestId": st.text(min_size=1),
        "traceId": st.text(min_size=1),
        "spanId": st.text(min_size=1),
        "schemaVersion": st.text(min_size=1),
        "deterministic": st.booleans(),
        "transport": st.sampled_from(["http", "stdio"]),
        "route": st.text(min_size=1),
        "method": st.text(min_size=1),
        "status": st.sampled_from(["ok", "error"]),
        "attempt": st.integers(min_value=0, max_value=3),
        "execution": _fixed_dict(
            {
                "durationMs": st.floats(
                    min_value=0, allow_nan=False, allow_infinity=False
                ),
                "inputBytes": st.integers(min_value=0, max_value=2048),
                "outputBytes": st.integers(min_value=0, max_value=2048),
            }
        ),
        "idempotency": _fixed_dict({"cacheHit": st.booleans()}),
    }
    optional_fields = {
        "toolId": st.one_of(st.none(), st.text()),
        "promptId": st.one_of(st.none(), st.text()),
    }
    return _fixed_dict({**required_fields, **optional_fields})


invalid_missing_meta = _fixed_dict(
    {
        "ok": st.booleans(),
        "data": st.one_of(st.none(), st.dictionaries(keys=st.text(), values=st.integers())),
        "error": st.one_of(st.none(), st.dictionaries(keys=st.text(), values=st.text())),
    }
)

invalid_success_with_error = _fixed_dict(
    {
        "ok": st.just(True),
        "data": st.dictionaries(keys=st.text(), values=st.text(), min_size=0, max_size=3),
        "error": _fixed_dict({"code": st.text(), "message": st.text()}),
        "meta": _meta_strategy(),
    }
)

invalid_envelopes = st.one_of(invalid_missing_meta, invalid_success_with_error)


@given(invalid_envelopes)
def test_envelope_validator_rejects_invalid_cases(payload: dict[str, Any]) -> None:
    """Any structurally invalid payload should be rejected by the JSON schema."""
    with pytest.raises(ValidationError):
        _ENVELOPE_VALIDATOR.validate(payload)
