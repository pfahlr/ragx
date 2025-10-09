from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from jsonschema import Draft202012Validator, ValidationError, validators

__all__ = ["SchemaRegistry", "ToolIOValidators", "ValidatorProtocol"]


class ValidatorProtocol(Protocol):
    """Protocol representing a compiled JSON schema validator."""

    def validate(self, instance: object) -> None:
        """Validate the given instance against a schema."""


@dataclass(slots=True)
class ToolIOValidators:
    """Container for per-tool input and output validators."""

    input: ValidatorProtocol
    output: ValidatorProtocol


class _ToolPayloadValidator:
    """Composite validator enforcing shared and tool-specific constraints."""

    def __init__(
        self,
        *,
        tool_id: str,
        base: ValidatorProtocol,
        payload: Draft202012Validator,
        field: str,
    ) -> None:
        self._tool_id = tool_id
        self._base = base
        self._payload = payload
        self._field = field

    def validate(self, instance: object) -> None:
        if not isinstance(instance, Mapping):
            raise ValidationError(f"{self._field} envelope must be a mapping")
        if self._field == "input":
            self._base.validate(instance)
        tool_value = instance.get("tool")
        if tool_value != self._tool_id:
            raise ValidationError(
                f"tool field must equal '{self._tool_id}' (received {tool_value!r})"
            )
        payload_value = instance.get(self._field)
        if payload_value is None:
            raise ValidationError(f"{self._field} field is required for tool '{self._tool_id}'")
        self._payload.validate(payload_value)


_MODULE_ROOT = Path(__file__).resolve().parent.parent


class SchemaRegistry:
    """Load and cache JSON schema validators for envelopes and tool IO."""

    _DEFAULT_ENVELOPE_SCHEMA = _MODULE_ROOT / "schemas" / "mcp" / "envelope.schema.json"
    _DEFAULT_TOOL_IO_SCHEMA = Path("codex/specs/schemas/tool_io.schema.json")
    _DEFAULT_TOOL_SCHEMA_ROOT = _MODULE_ROOT / "schemas" / "tools"

    def __init__(
        self,
        *,
        envelope_schema_path: Path | None = None,
        tool_io_schema_path: Path | None = None,
        tool_schema_root: Path | None = None,
    ) -> None:
        self._envelope_schema_path = Path(
            envelope_schema_path or self._DEFAULT_ENVELOPE_SCHEMA
        )
        self._tool_io_schema_path = Path(
            tool_io_schema_path or self._DEFAULT_TOOL_IO_SCHEMA
        )
        self._tool_schema_root = Path(tool_schema_root or self._DEFAULT_TOOL_SCHEMA_ROOT)
        self._fingerprint_cache: dict[str, Draft202012Validator] = {}
        self._path_fingerprints: dict[Path, str] = {}
        self._tool_cache: dict[str, ToolIOValidators] = {}
        self._tool_schema_index: dict[str, str] = self._build_tool_schema_index()

    def load_envelope(self) -> Draft202012Validator:
        """Return a cached validator for the canonical envelope schema."""
        return self._load_validator(self._envelope_schema_path)

    def load_tool_io(self, tool_id: str) -> ToolIOValidators:
        """Return cached validators for the given tool's input and output payloads."""
        if tool_id in self._tool_cache:
            return self._tool_cache[tool_id]

        stem = self._tool_schema_stem(tool_id)
        base_validator = self._load_validator(self._tool_io_schema_path)
        input_validator = self._load_validator(
            self._tool_schema_root / f"{stem}.input.schema.json"
        )
        output_validator = self._load_validator(
            self._tool_schema_root / f"{stem}.output.schema.json"
        )
        validators_bundle = ToolIOValidators(
            input=_ToolPayloadValidator(
                tool_id=tool_id,
                base=base_validator,
                payload=input_validator,
                field="input",
            ),
            output=_ToolPayloadValidator(
                tool_id=tool_id,
                base=base_validator,
                payload=output_validator,
                field="output",
            ),
        )
        self._tool_cache[tool_id] = validators_bundle
        return validators_bundle

    # Internal helpers -----------------------------------------------------------------

    def _load_validator(self, path: Path) -> Draft202012Validator:
        fingerprint = self._path_fingerprints.get(path)
        schema: dict[str, Any] | None = None
        if fingerprint is None:
            schema = self._read_schema(path)
            fingerprint = self._fingerprint(schema)
            self._path_fingerprints[path] = fingerprint
        if fingerprint in self._fingerprint_cache:
            return self._fingerprint_cache[fingerprint]
        if schema is None:
            schema = self._read_schema(path)
        validator_cls = validators.validator_for(schema)
        validator_cls.check_schema(schema)
        validator = validator_cls(schema)
        self._fingerprint_cache[fingerprint] = validator
        return validator

    def _read_schema(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Schema file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _fingerprint(self, schema: Mapping[str, Any]) -> str:
        payload = json.dumps(schema, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _tool_schema_stem(self, tool_id: str) -> str:
        if tool_id in self._tool_schema_index:
            return self._tool_schema_index[tool_id]
        # Rebuild the index to pick up newly-added schemas during runtime.
        self._tool_schema_index = self._build_tool_schema_index()
        if tool_id not in self._tool_schema_index:
            raise KeyError(f"Unknown tool identifier: {tool_id}")
        return self._tool_schema_index[tool_id]

    def _build_tool_schema_index(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        if not self._tool_schema_root.exists():
            return mapping
        for path in sorted(self._tool_schema_root.glob("*.input.schema.json")):
            stem = path.name[: -len(".input.schema.json")]
            tool_id = f"mcp.tool:{stem.replace('_', '.')}"
            mapping[tool_id] = stem
        return mapping
