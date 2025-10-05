"""Property-based executable spec for envelope validation edge cases."""

from __future__ import annotations

from typing import Any

import pytest
from jsonschema import ValidationError

from apps.mcp_server.validation.schema_registry import SchemaRegistry

try:
    from hypothesis import given
    from hypothesis import strategies as st
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pytest.importorskip("hypothesis")

if "st" in locals() and not hasattr(st, "just"):
    pytest.skip("hypothesis strategies helpers unavailable", allow_module_level=True)

def _fixed_dictionaries(schema: dict[str, Any]) -> Any:
    """Return a fixed dictionary strategy compatible with older Hypothesis builds."""

    factory = getattr(st, "fixed_dictionaries", None)
    if factory is None:  # pragma: no cover - fallback for older releases
        def _builder(**kwargs: Any) -> dict[str, Any]:
            return kwargs

        return st.builds(_builder, **schema)
    return factory(schema)


invalid_missing_method = _fixed_dictionaries(
    {
        "id": st.text(min_size=1),
        "jsonrpc": st.just("2.0"),
        "params": st.dictionaries(keys=st.text(), values=st.integers(), min_size=0, max_size=2),
    }
)

invalid_wrong_params = _fixed_dictionaries(
    {
        "id": st.text(min_size=1),
        "jsonrpc": st.just("2.0"),
        "method": st.sampled_from(["mcp.tool.invoke", "mcp.discover", "mcp.prompt.get"]),
        "params": st.one_of(st.none(), st.integers(), st.text(), st.lists(st.integers())),
    }
)

invalid_envelopes = st.one_of(invalid_missing_method, invalid_wrong_params)


@given(invalid_envelopes)
def test_envelope_validator_rejects_invalid_cases(payload: dict[str, Any]) -> None:
    """Any structurally invalid payload should be rejected by the JSON schema."""
    validator = SchemaRegistry().load_envelope()
    with pytest.raises(ValidationError):
        validator.validate(payload)
