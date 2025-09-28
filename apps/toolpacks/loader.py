from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from jsonschema import validators
from jsonschema.exceptions import SchemaError


class ToolpackValidationError(Exception):
    """Raised when a Toolpack definition fails validation."""


_VALID_EXECUTION_KINDS = {"python", "node", "php", "cli", "http"}


@dataclass(frozen=True)
class Toolpack:
    """In-memory representation of a Toolpack configuration."""

    id: str
    version: str
    deterministic: bool
    timeout_ms: int
    limits: Mapping[str, Any]
    input_schema: Mapping[str, Any]
    output_schema: Mapping[str, Any]
    execution: Mapping[str, Any]
    caps: Mapping[str, Any]
    env: Mapping[str, Any]
    templating: Mapping[str, Any]
    source_path: Path

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], source_path: Path) -> Toolpack:
        if not isinstance(data, Mapping):
            raise ToolpackValidationError(
                f"Expected mapping for toolpack {source_path}, got {type(data).__name__}"
            )

        required_fields = [
            "id",
            "version",
            "deterministic",
            "timeoutMs",
            "limits",
            "inputSchema",
            "outputSchema",
            "execution",
        ]
        missing = [field for field in required_fields if field not in data]
        if missing:
            raise ToolpackValidationError(
                f"Toolpack {source_path} missing required field(s): {', '.join(missing)}"
            )

        tool_id = _require_str(data["id"], "id", source_path)
        version = _require_str(data["version"], "version", source_path)
        deterministic = data["deterministic"]
        if not isinstance(deterministic, bool):
            raise ToolpackValidationError(
                f"Toolpack {tool_id} deterministic must be a boolean"
            )

        timeout_ms = data["timeoutMs"]
        if not isinstance(timeout_ms, int) or timeout_ms <= 0:
            raise ToolpackValidationError(
                f"Toolpack {tool_id} timeoutMs must be a positive integer"
            )

        limits = _require_mapping(data["limits"], "limits", tool_id)
        for key in ("maxInputBytes", "maxOutputBytes"):
            if key not in limits:
                raise ToolpackValidationError(
                    f"Toolpack {tool_id} limits missing required key '{key}'"
                )
            value = limits[key]
            if not isinstance(value, int) or value <= 0:
                raise ToolpackValidationError(
                    f"Toolpack {tool_id} limits['{key}'] must be a positive integer"
                )

        caps = data.get("caps", {})
        caps = _require_mapping(caps, "caps", tool_id)

        env = data.get("env", {})
        env = _require_mapping(env, "env", tool_id)

        templating = data.get("templating", {})
        templating = _require_mapping(templating, "templating", tool_id)

        input_schema = _resolve_schema(data["inputSchema"], source_path.parent, tool_id)
        output_schema = _resolve_schema(data["outputSchema"], source_path.parent, tool_id)

        execution = _require_mapping(data["execution"], "execution", tool_id)
        kind = execution.get("kind")
        if kind not in _VALID_EXECUTION_KINDS:
            raise ToolpackValidationError(
                f"Toolpack {tool_id} execution.kind must be one of {_VALID_EXECUTION_KINDS}"
            )

        return cls(
            id=tool_id,
            version=version,
            deterministic=deterministic,
            timeout_ms=timeout_ms,
            limits=dict(limits),
            input_schema=input_schema,
            output_schema=output_schema,
            execution=dict(execution),
            caps=dict(caps),
            env=dict(env),
            templating=dict(templating),
            source_path=source_path,
        )


class ToolpackLoader:
    """Load and expose Toolpack configurations."""

    def __init__(self) -> None:
        self._toolpacks: dict[str, Toolpack] = {}

    def load_dir(self, directory: Path | str) -> None:
        base_dir = Path(directory)
        if not base_dir.exists():
            raise ToolpackValidationError(f"Toolpacks directory not found: {base_dir}")

        toolpacks: dict[str, Toolpack] = {}
        for path in sorted(base_dir.rglob("*.tool.yaml")):
            with path.open("r", encoding="utf-8") as handle:
                try:
                    data = yaml.safe_load(handle)
                except yaml.YAMLError as exc:
                    raise ToolpackValidationError(
                        f"Failed to parse YAML for toolpack {path}: {exc}"
                    ) from exc

            toolpack = Toolpack.from_dict(data, path)
            if toolpack.id in toolpacks:
                raise ToolpackValidationError(
                    f"Duplicate toolpack id '{toolpack.id}' found in {path}"
                )
            toolpacks[toolpack.id] = toolpack

        self._toolpacks = dict(sorted(toolpacks.items(), key=lambda item: item[0]))

    def list(self) -> list[Toolpack]:
        return list(self._toolpacks.values())

    def get(self, tool_id: str) -> Toolpack:
        return self._toolpacks[tool_id]


def _require_str(value: Any, field: str, source: Path) -> str:
    if not isinstance(value, str) or not value:
        raise ToolpackValidationError(
            f"Field '{field}' in toolpack {source} must be a non-empty string"
        )
    return value


def _require_mapping(value: Any, field: str, tool_id: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ToolpackValidationError(
            f"Toolpack {tool_id} field '{field}' must be a mapping"
        )
    return value


def _resolve_schema(
    schema_spec: Any, base_dir: Path, tool_id: str
) -> Mapping[str, Any]:
    if not isinstance(schema_spec, Mapping):
        raise ToolpackValidationError(
            f"Toolpack {tool_id} schema definition must be a mapping"
        )

    if "$ref" in schema_spec:
        ref = schema_spec["$ref"]
        if not isinstance(ref, str) or not ref:
            raise ToolpackValidationError(
                f"Toolpack {tool_id} schema $ref must be a non-empty string"
            )
        schema_path = (base_dir / ref).resolve()
        if not schema_path.is_file():
            raise ToolpackValidationError(
                f"Toolpack {tool_id} schema reference not found: {ref}"
            )
        try:
            with schema_path.open("r", encoding="utf-8") as handle:
                schema = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ToolpackValidationError(
                f"Toolpack {tool_id} failed to load schema {ref}: {exc}"
            ) from exc
    else:
        schema = dict(schema_spec)

    _validate_json_schema(schema, tool_id)
    return schema


def _validate_json_schema(schema: Mapping[str, Any], tool_id: str) -> None:
    try:
        validator_cls = validators.validator_for(schema)
        validator_cls.check_schema(schema)
    except SchemaError as exc:
        raise ToolpackValidationError(
            f"Toolpack {tool_id} schema failed validation: {exc.message}"
        ) from exc
    except Exception as exc:  # pragma: no cover - safeguard for unexpected validator errors
        raise ToolpackValidationError(
            f"Toolpack {tool_id} schema failed validation: {exc}"
        ) from exc
