"""Schema registry stub for MCP envelope validation.

This module intentionally provides a minimal interface so tests can
express the desired behavior without shipping production-ready
validation logic. The real implementation will replace the
``NotImplementedValidator`` placeholders with jsonschema validators that
honour caching and schema fingerprints.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

__all__ = ["SchemaRegistry", "ToolIOValidators", "ValidatorProtocol"]


class ValidatorProtocol(Protocol):
    """Protocol for validator objects returned by the registry."""

    def validate(self, instance: object) -> None:  # pragma: no cover - stub contract
        """Validate an instance against a JSON schema."""


@dataclass(slots=True)
class ToolIOValidators:
    """Container for per-tool input/output validators."""

    input: ValidatorProtocol
    output: ValidatorProtocol


class NotImplementedValidator:
    """Placeholder validator that raises ``NotImplementedError`` when used."""

    def __init__(self, schema_path: Path | None = None) -> None:
        self._schema_path = schema_path

    @property
    def schema_path(self) -> Path | None:
        return self._schema_path

    def validate(self, instance: object) -> None:  # pragma: no cover - stub contract
        raise NotImplementedError("Schema validation is not yet implemented for this task")


class SchemaRegistry:
    """Stub registry returning placeholder validators for schemas.

    The registry exposes ``load_envelope`` and ``load_tool_io`` as
    described in the spec. The actual implementation will provide
    caching keyed by schema fingerprint.
    """

    _SCHEMA_BASE = Path("codex/specs/schemas")

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str], ValidatorProtocol] = {}

    def load_envelope(self) -> ValidatorProtocol:
        """Return a validator for the envelope schema (placeholder)."""
        key = ("envelope", "envelope.schema.json")
        if key not in self._cache:
            self._cache[key] = NotImplementedValidator(self._schema_path(key[1]))
        return self._cache[key]

    def load_tool_io(self, tool_id: str) -> ToolIOValidators:
        """Return placeholder validators for a specific tool."""
        input_key = (tool_id, "tool_io.input.schema.json")
        output_key = (tool_id, "tool_io.output.schema.json")
        schema_path = self._schema_path("tool_io.schema.json")
        input_validator = self._cache.setdefault(
            input_key,
            NotImplementedValidator(schema_path),
        )
        output_validator = self._cache.setdefault(
            output_key,
            NotImplementedValidator(schema_path),
        )
        return ToolIOValidators(input=input_validator, output=output_validator)

    def _schema_path(self, relative: str) -> Path:
        return self._SCHEMA_BASE / relative
