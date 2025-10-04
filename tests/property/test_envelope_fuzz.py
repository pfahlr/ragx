from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError, validators

try:  # pragma: no cover - optional dependency
    from hypothesis import given
    from hypothesis import strategies as st
except ModuleNotFoundError:  # pragma: no cover - stub fallback provided via conftest
    from hypothesis import given  # type: ignore[assignment]
    from hypothesis import strategies as st  # type: ignore[assignment]

SCHEMA_ROOT = Path("codex/specs/schemas")


def _load_envelope_validator() -> Draft202012Validator:
    schema_path = SCHEMA_ROOT / "envelope.schema.json"
    if not schema_path.exists():
        pytest.skip("Envelope schema not available")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator_cls = validators.validator_for(schema)
    validator_cls.check_schema(schema)
    return validator_cls(schema)


@given(st.text(min_size=1))
def test_envelope_params_must_be_object(bad_params: object) -> None:
    validator = _load_envelope_validator()
    payload = {
        "id": "req-789",
        "jsonrpc": "2.0",
        "method": "mcp.tool.invoke",
        "params": bad_params,
    }
    with pytest.raises(ValidationError):
        validator.validate(payload)
