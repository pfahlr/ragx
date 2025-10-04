"""Property-based executable spec for envelope validation edge cases."""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st
from jsonschema import ValidationError

from apps.mcp_server.validation.schema_registry_stub import SchemaRegistry

invalid_missing_method = st.fixed_dictionaries(
    {
        "id": st.text(min_size=1),
        "jsonrpc": st.just("2.0"),
        "params": st.dictionaries(keys=st.text(), values=st.integers(), min_size=0, max_size=2),
    }
)

invalid_wrong_params = st.fixed_dictionaries(
    {
        "id": st.text(min_size=1),
        "jsonrpc": st.just("2.0"),
        "method": st.sampled_from(["mcp.tool.invoke", "mcp.discover", "mcp.prompt.get"]),
        "params": st.one_of(st.none(), st.integers(), st.text(), st.lists(st.integers())),
    }
)

invalid_envelopes = st.one_of(invalid_missing_method, invalid_wrong_params)


@pytest.mark.xfail(reason="Envelope validator not yet implemented", strict=True)
@given(invalid_envelopes)
def test_envelope_validator_rejects_invalid_cases(payload: dict[str, Any]) -> None:
    """Any structurally invalid payload should be rejected by the JSON schema."""
    validator = SchemaRegistry().load_envelope()
    with pytest.raises(ValidationError):
        validator.validate(payload)
