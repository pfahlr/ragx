"""Property-based executable spec for envelope validation edge cases."""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st
from jsonschema import ValidationError

from apps.mcp_server.validation.schema_registry import SchemaRegistry


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
        "durationMs": st.floats(min_value=0, allow_nan=False, allow_infinity=False),
        "status": st.sampled_from(["ok", "error"]),
        "attempt": st.integers(min_value=0, max_value=3),
        "inputBytes": st.integers(min_value=0, max_value=2048),
        "outputBytes": st.integers(min_value=0, max_value=2048),
    }
    optional_fields = {
        "toolId": st.one_of(st.none(), st.text()),
        "promptId": st.one_of(st.none(), st.text()),
    }
    return st.fixed_dictionaries({**required_fields, **optional_fields})


invalid_missing_meta = st.fixed_dictionaries(
    {
        "ok": st.booleans(),
        "data": st.one_of(st.none(), st.dictionaries(keys=st.text(), values=st.integers())),
        "error": st.one_of(st.none(), st.dictionaries(keys=st.text(), values=st.text())),
    }
)

invalid_success_with_error = st.fixed_dictionaries(
    {
        "ok": st.just(True),
        "data": st.dictionaries(keys=st.text(), values=st.text(), min_size=0, max_size=3),
        "error": st.fixed_dictionaries({"code": st.text(), "message": st.text()}),
        "meta": _meta_strategy(),
    }
)

invalid_envelopes = st.one_of(invalid_missing_meta, invalid_success_with_error)


@given(invalid_envelopes)
def test_envelope_validator_rejects_invalid_cases(payload: dict[str, Any]) -> None:
    """Any structurally invalid payload should be rejected by the JSON schema."""
    validator = SchemaRegistry().load_envelope()
    with pytest.raises(ValidationError):
        validator.validate(payload)
