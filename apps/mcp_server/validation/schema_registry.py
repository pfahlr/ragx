"""Production schema registry for MCP envelope and tool IO validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from jsonschema import Draft202012Validator, validators

__all__ = ["SchemaRegistry", "ToolIOValidators", "ValidatorProtocol"]


class ValidatorProtocol(Protocol):
    """Protocol describing the validator interface exposed to callers."""

    def validate(self, instance: object) -> None:  # pragma: no cover - interface contract
        """Validate *instance* against the JSON schema represented by the validator."""


@dataclass(slots=True)
class ToolIOValidators:
    """Container holding input/output validators for a tool."""

    input: ValidatorProtocol
    output: ValidatorProtocol


class SchemaRegistry:
    """Load and cache JSON Schema validators for MCP envelope and tool IO payloads."""

    def __init__(self, *, schema_root: Path | None = None) -> None:
        self._schema_root = Path(schema_root) if schema_root is not None else Path(
            "codex/specs/schemas"
        )
        self._validator_cache: dict[Path, tuple[str, Draft202012Validator]] = {}
        self._tool_cache: dict[str, ToolIOValidators] = {}

    def load_envelope(self) -> Draft202012Validator:
        """Return a cached validator for the canonical envelope schema."""

        schema_path = self._schema_root / "envelope.schema.json"
        return self._load_validator(schema_path)

    def load_tool_io(self, tool_id: str) -> ToolIOValidators:
        """Return validators for the shared tool IO schema keyed by *tool_id*."""

        if tool_id in self._tool_cache:
            return self._tool_cache[tool_id]

        schema_path = self._schema_root / "tool_io.schema.json"
        validator = self._load_validator(schema_path)
        tool_validators = ToolIOValidators(input=validator, output=validator)
        self._tool_cache[tool_id] = tool_validators
        return tool_validators

    # Internal helpers -------------------------------------------------

    def _load_validator(self, path: Path) -> Draft202012Validator:
        if not path.exists():
            raise FileNotFoundError(f"Schema file not found: {path}")

        contents = path.read_bytes()
        fingerprint = hashlib.sha256(contents).hexdigest()
        cached = self._validator_cache.get(path)
        if cached and cached[0] == fingerprint:
            return cached[1]

        schema = json.loads(contents.decode("utf-8"))
        validator_cls = validators.validator_for(schema)
        validator_cls.check_schema(schema)
        validator = validator_cls(schema)
        self._validator_cache[path] = (fingerprint, validator)
        return validator
