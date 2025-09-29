from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from jsonschema import validators
from jsonschema.exceptions import SchemaError
from packaging.version import InvalidVersion, Version


class ToolpackValidationError(Exception):
    """Raised when a Toolpack definition fails validation."""


_VALID_EXECUTION_KINDS = {"python", "node", "php", "cli", "http"}
_TOOL_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:\.[a-z0-9]+)+$")
_NETWORK_VALUE_PATTERN = re.compile(r"^[a-z][a-z0-9+.-]*$")


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
        _validate_tool_id(tool_id, source_path)
        version = _require_str(data["version"], "version", source_path)
        _validate_version(version, tool_id)
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

        limits = _validate_limits(data["limits"], tool_id)

        caps = _normalise_caps(data.get("caps"), tool_id, timeout_ms, limits)

        env = _normalise_env(data.get("env"), tool_id)

        templating = _normalise_templating(data.get("templating"), tool_id)

        input_schema = _resolve_schema(data["inputSchema"], source_path.parent, tool_id)
        output_schema = _resolve_schema(data["outputSchema"], source_path.parent, tool_id)

        execution = _require_mapping(data["execution"], "execution", tool_id)
        kind = execution.get("kind")
        if kind not in _VALID_EXECUTION_KINDS:
            raise ToolpackValidationError(
                f"Toolpack {tool_id} execution.kind must be one of {_VALID_EXECUTION_KINDS}"
            )
        _validate_execution_config(tool_id, kind, execution)

        return cls(
            id=tool_id,
            version=version,
            deterministic=deterministic,
            timeout_ms=timeout_ms,
            limits=dict(limits),
            input_schema=input_schema,
            output_schema=output_schema,
            execution=dict(execution),
            caps=caps,
            env=env,
            templating=templating,
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


def _validate_tool_id(tool_id: str, source_path: Path) -> None:
    if not _TOOL_ID_PATTERN.fullmatch(tool_id):
        raise ToolpackValidationError(
            f"Toolpack identifier '{tool_id}' in {source_path} must be lowercase dotted form"
        )


def _validate_version(version: str, tool_id: str) -> None:
    try:
        parsed = Version(version)
    except InvalidVersion as exc:
        raise ToolpackValidationError(
            f"Toolpack {tool_id} version must follow semantic versioning"
        ) from exc
    if len(parsed.release) < 3:
        raise ToolpackValidationError(
            f"Toolpack {tool_id} version must include major.minor.patch"
        )


def _validate_limits(value: Any, tool_id: str) -> dict[str, int]:
    limits = _require_mapping(value, "limits", tool_id)
    validated: dict[str, int] = {}
    for key in ("maxInputBytes", "maxOutputBytes"):
        if key not in limits:
            raise ToolpackValidationError(
                f"Toolpack {tool_id} limits missing required key '{key}'"
            )
        limit_value = limits[key]
        if not isinstance(limit_value, int) or limit_value <= 0:
            raise ToolpackValidationError(
                f"Toolpack {tool_id} limits['{key}'] must be a positive integer"
            )
        validated[key] = limit_value
    return validated


def _normalise_caps(
    value: Any, tool_id: str, timeout_ms: int, limits: Mapping[str, int]
) -> dict[str, Any]:
    caps_mapping = _require_mapping(value, "caps", tool_id)
    normalised: dict[str, Any] = {
        key: val for key, val in caps_mapping.items() if key not in {"network"}
    }

    for key, expected in (
        ("timeoutMs", timeout_ms),
        ("maxInputBytes", limits["maxInputBytes"]),
        ("maxOutputBytes", limits["maxOutputBytes"]),
    ):
        existing = caps_mapping.get(key)
        if existing is not None and existing != expected:
            raise ToolpackValidationError(
                f"Toolpack {tool_id} caps.{key} must match declared {key}"
            )
        normalised[key] = expected

    network_raw = caps_mapping.get("network", [])
    if isinstance(network_raw, str):
        network_values = [network_raw]
    elif isinstance(network_raw, list):
        network_values = network_raw
    elif network_raw in (None, {}):
        network_values = []
    else:
        raise ToolpackValidationError(
            f"Toolpack {tool_id} caps.network must be a list of strings"
        )

    cleaned: list[str] = []
    seen: set[str] = set()
    for entry in network_values:
        if not isinstance(entry, str) or not entry.strip():
            raise ToolpackValidationError(
                f"Toolpack {tool_id} caps.network must contain non-empty strings"
            )
        candidate = entry.strip().lower()
        if not _NETWORK_VALUE_PATTERN.fullmatch(candidate):
            raise ToolpackValidationError(
                f"Toolpack {tool_id} caps.network value '{entry}' is invalid"
            )
        if candidate not in seen:
            cleaned.append(candidate)
            seen.add(candidate)

    normalised["network"] = cleaned
    return normalised


def _normalise_env(value: Any, tool_id: str) -> dict[str, Any]:
    env_mapping = _require_mapping(value, "env", tool_id)
    normalised: dict[str, Any] = {}
    for key, env_value in env_mapping.items():
        if not isinstance(key, str) or not key:
            raise ToolpackValidationError(
                f"Toolpack {tool_id} env keys must be non-empty strings"
            )
        if isinstance(env_value, list):
            items: list[str] = []
            for entry in env_value:
                if not isinstance(entry, str) or not entry:
                    raise ToolpackValidationError(
                        f"Toolpack {tool_id} env['{key}'] entries must be non-empty strings"
                    )
                items.append(entry)
            normalised[key] = items
        elif isinstance(env_value, Mapping):
            nested: dict[str, str] = {}
            for nested_key, nested_value in env_value.items():
                if not isinstance(nested_key, str) or not nested_key:
                    raise ToolpackValidationError(
                        f"Toolpack {tool_id} env['{key}'] keys must be non-empty strings"
                    )
                if not isinstance(nested_value, str):
                    raise ToolpackValidationError(
                        f"Toolpack {tool_id} env['{key}']['{nested_key}'] must be a string"
                    )
                nested[nested_key] = nested_value
            normalised[key] = nested
        elif isinstance(env_value, str | type(None)):
            normalised[key] = env_value
        else:
            raise ToolpackValidationError(
                f"Toolpack {tool_id} env['{key}'] must be string, list of strings, or mapping"
            )
    return normalised


def _normalise_templating(value: Any, tool_id: str) -> dict[str, Any]:
    templating_mapping = _require_mapping(value, "templating", tool_id)
    normalised: dict[str, Any] = {}
    for key, templating_value in templating_mapping.items():
        if not isinstance(key, str) or not key:
            raise ToolpackValidationError(
                f"Toolpack {tool_id} templating keys must be non-empty strings"
            )
        if not isinstance(templating_value, str | bool) and templating_value is not None:
            raise ToolpackValidationError(
                f"Toolpack {tool_id} templating['{key}'] must be string, bool, or null"
            )
        normalised[key] = templating_value
    return normalised


def _validate_execution_config(tool_id: str, kind: str, execution: Mapping[str, Any]) -> None:
    def _require_non_empty_string(field: str) -> None:
        value = execution.get(field)
        if not isinstance(value, str) or not value:
            raise ToolpackValidationError(
                f"Toolpack {tool_id} execution.{field} must be a non-empty string"
            )

    if kind == "python":
        module = execution.get("module")
        script = execution.get("script")
        if module is not None and (not isinstance(module, str) or not module):
            raise ToolpackValidationError(
                f"Toolpack {tool_id} execution.module must be a non-empty string"
            )
        if script is not None and (not isinstance(script, str) or not script):
            raise ToolpackValidationError(
                f"Toolpack {tool_id} execution.script must be a non-empty string"
            )
        if module is None and script is None:
            raise ToolpackValidationError(
                f"Toolpack {tool_id} python execution requires 'module' or 'script'"
            )
    elif kind == "cli":
        cmd = execution.get("cmd")
        if not isinstance(cmd, list) or not cmd:
            raise ToolpackValidationError(
                f"Toolpack {tool_id} cli execution requires cmd list"
            )
        for arg in cmd:
            if not isinstance(arg, str) or not arg:
                raise ToolpackValidationError(
                    f"Toolpack {tool_id} cli cmd entries must be non-empty strings"
                )
    elif kind == "http":
        _require_non_empty_string("url")
    elif kind == "node":
        if not any(
            isinstance(execution.get(option), str) and execution.get(option)
            for option in ("node", "script", "module")
        ):
            raise ToolpackValidationError(
                f"Toolpack {tool_id} node execution requires 'node', 'script', or 'module' entry"
            )
    elif kind == "php":
        if not any(
            isinstance(execution.get(option), str) and execution.get(option)
            for option in ("php", "script")
        ):
            raise ToolpackValidationError(
                f"Toolpack {tool_id} php execution requires 'php' or 'script' entry"
            )


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
