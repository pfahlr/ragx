"""Executable tests for the JSON-RPC envelope schema."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

SCHEMA_PATH = Path("codex/specs/schemas/envelope.schema.json")


@pytest.fixture(scope="module")
def jsonrpc_validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text())
    return Draft202012Validator(schema)


def test_notification_without_id_is_valid(jsonrpc_validator: Draft202012Validator) -> None:
    payload = {"jsonrpc": "2.0", "method": "$/cancel"}
    jsonrpc_validator.validate(payload)


def test_request_with_id_is_valid(jsonrpc_validator: Draft202012Validator) -> None:
    payload = {"jsonrpc": "2.0", "id": "req-1", "method": "mcp.discover", "params": {}}
    jsonrpc_validator.validate(payload)


def test_notification_with_params_without_id_is_valid(
    jsonrpc_validator: Draft202012Validator,
) -> None:
    payload = {"jsonrpc": "2.0", "method": "$/progress", "params": {"token": "t"}}
    jsonrpc_validator.validate(payload)


def test_response_requires_id(jsonrpc_validator: Draft202012Validator) -> None:
    payload = {"jsonrpc": "2.0", "result": {"ok": True}}
    with pytest.raises(ValidationError):
        jsonrpc_validator.validate(payload)


def test_response_with_error_or_result_is_valid(
    jsonrpc_validator: Draft202012Validator,
) -> None:
    success = {"jsonrpc": "2.0", "id": 1, "result": {"ok": True}}
    jsonrpc_validator.validate(success)

    error = {
        "jsonrpc": "2.0",
        "id": "req-2",
        "error": {"code": -32000, "message": "Failure"},
    }
    jsonrpc_validator.validate(error)



def test_response_rejects_result_and_error(jsonrpc_validator: Draft202012Validator) -> None:
    payload = {
        "jsonrpc": "2.0",
        "id": "req-3",
        "result": {"ok": True},
        "error": {"code": -32000, "message": "Failure"},
    }
    with pytest.raises(ValidationError):
        jsonrpc_validator.validate(payload)
