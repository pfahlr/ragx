"""Production schema registry for MCP envelope and tool IO validation."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from jsonschema import Draft202012Validator, validators

__all__ = ["SchemaRegistry", "ToolIOValidators", "ValidatorProtocol"]


class ValidatorProtocol(Protocol):
    """Protocol describing validator behaviour required by the registry."""

    def validate(self, instance: object) -> None:
        """Validate *instance* against the underlying JSON schema."""


@dataclass(slots=True)
class ToolIOValidators:
    """Container aggregating input/output validators for a tool."""

    input: ValidatorProtocol
    output: ValidatorProtocol


class SchemaRegistry:
    """Load and cache JSON schema validators used by the MCP server."""

    _SCHEMA_BASE = Path("codex/specs/schemas")

    def __init__(self, *, schema_base: Path | None = None) -> None:
        self._schema_base = Path(schema_base) if schema_base else self._SCHEMA_BASE
        self._validator_cache: dict[str, Draft202012Validator] = {}
        self._tool_cache: dict[str, ToolIOValidators] = {}

    # Public API -----------------------------------------------------
    def load_envelope(self) -> ValidatorProtocol:
        """Return the compiled validator for the envelope schema."""

        schema = self._read_schema("envelope.schema.json")
        return self._validator_from_schema(schema)

    def load_tool_io(self, tool_id: str) -> ToolIOValidators:
        """Return validators enforcing tool IO contracts for ``tool_id``."""

        if tool_id in self._tool_cache:
            return self._tool_cache[tool_id]

        base_schema = self._read_schema("tool_io.schema.json")
        input_schema = self._build_tool_schema(base_schema, tool_id, direction="input")
        output_schema = self._build_tool_schema(base_schema, tool_id, direction="output")

        validators_bundle = ToolIOValidators(
            input=self._validator_from_schema(input_schema),
            output=self._validator_from_schema(output_schema),
        )
        self._tool_cache[tool_id] = validators_bundle
        return validators_bundle

    # Internal helpers -----------------------------------------------
    def _read_schema(self, relative_path: str) -> dict[str, Any]:
        path = (self._schema_base / relative_path).resolve()
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _build_tool_schema(
        self, base_schema: dict[str, Any], tool_id: str, *, direction: str
    ) -> dict[str, Any]:
        schema = deepcopy(base_schema)
        properties = dict(schema.get("properties", {}))
        tool_property = dict(properties.get("tool", {}))
        tool_property["const"] = tool_id
        properties["tool"] = tool_property

        if direction == "input":
            schema["required"] = sorted({"tool", "input"})
        else:
            # Output payloads mirror the input contract but expose ``output``.
            properties.pop("input", None)
            properties["output"] = {
                "description": "Tool-specific output payload validated downstream.",
                "type": "object",
            }
            schema["required"] = sorted({"tool", "output"})
        schema["properties"] = properties
        return schema

    def _validator_from_schema(self, schema: dict[str, Any]) -> Draft202012Validator:
        fingerprint = self._fingerprint(schema)
        if fingerprint in self._validator_cache:
            return self._validator_cache[fingerprint]
        validator_cls = validators.validator_for(schema)
        validator_cls.check_schema(schema)
        validator = validator_cls(schema)
        self._validator_cache[fingerprint] = validator
        return validator

    @staticmethod
    def _fingerprint(schema: dict[str, Any]) -> str:
        payload = json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

