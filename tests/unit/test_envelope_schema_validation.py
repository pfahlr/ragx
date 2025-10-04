from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError, validators

from apps.mcp_server.validation.schema_registry_stub import SchemaRegistry

FIXTURE_DIR = Path("tests/fixtures/mcp/envelope")
SCHEMA_ROOT = Path("codex/specs/schemas")


@pytest.fixture(scope="module")
def envelope_validator() -> Draft202012Validator:
    schema_path = SCHEMA_ROOT / "envelope.schema.json"
    if not schema_path.exists():
        pytest.fail(f"Envelope schema missing at {schema_path}")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator_cls = validators.validator_for(schema)
    validator_cls.check_schema(schema)
    return validator_cls(schema)


@pytest.mark.spec_xfail(strict=True, reason="SchemaRegistry.load_envelope not implemented")
@pytest.mark.xfail(strict=True, reason="SchemaRegistry.load_envelope not implemented")
def test_schema_registry_load_envelope_rejects_missing_method() -> None:
    registry = SchemaRegistry(schema_root=SCHEMA_ROOT)
    validator = registry.load_envelope()
    payload = json.loads((FIXTURE_DIR / "invalid_missing_method.json").read_text(encoding="utf-8"))
    with pytest.raises(ValidationError):
        validator.validate(payload)


@pytest.mark.spec_xfail(strict=True, reason="SchemaRegistry.load_tool_io not implemented")
@pytest.mark.xfail(strict=True, reason="SchemaRegistry.load_tool_io not implemented")
def test_schema_registry_load_tool_io_rejects_invalid_input() -> None:
    registry = SchemaRegistry(schema_root=SCHEMA_ROOT)
    validators_bundle = registry.load_tool_io("vector.query.search")
    invalid_input = {"topK": 3}
    with pytest.raises(ValidationError):
        validators_bundle.input_validator.validate(invalid_input)


def test_envelope_schema_rejects_missing_method(envelope_validator: Draft202012Validator) -> None:
    payload = json.loads((FIXTURE_DIR / "invalid_missing_method.json").read_text(encoding="utf-8"))
    with pytest.raises(ValidationError) as excinfo:
        envelope_validator.validate(payload)
    assert "method" in str(excinfo.value)


def test_envelope_schema_rejects_non_object_params(
    envelope_validator: Draft202012Validator,
) -> None:
    payload = json.loads((FIXTURE_DIR / "invalid_params_type.json").read_text(encoding="utf-8"))
    with pytest.raises(ValidationError) as excinfo:
        envelope_validator.validate(payload)
    assert "object" in str(excinfo.value)
