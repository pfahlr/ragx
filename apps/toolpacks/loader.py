from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
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

        caps = _validate_caps(data.get("caps", {}), tool_id)

        env = _validate_env(data.get("env", {}), tool_id)

        templating = _validate_templating(data.get("templating", {}), tool_id)

        input_schema = _resolve_schema(data["inputSchema"], source_path.parent, tool_id)
        output_schema = _resolve_schema(data["outputSchema"], source_path.parent, tool_id)

        execution = _validate_execution(data.get("execution"), tool_id)

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


def _validate_caps(value: Any, tool_id: str) -> Mapping[str, Any]:
    caps = _require_mapping(value, "caps", tool_id)

    network = caps.get("network")
    if network is not None:
        if not isinstance(network, Sequence) or isinstance(network, str | bytes):
            raise ToolpackValidationError(
                f"Toolpack {tool_id} caps.network must be a list of strings"
            )
        for entry in network:
            if not isinstance(entry, str) or not entry:
                raise ToolpackValidationError(
                    f"Toolpack {tool_id} caps.network must contain non-empty strings"
                )

    return caps


def _validate_env(value: Any, tool_id: str) -> Mapping[str, Any]:
    env = _require_mapping(value, "env", tool_id)

    passthrough = env.get("passthrough")
    if passthrough is not None:
        if not isinstance(passthrough, Sequence) or isinstance(passthrough, str | bytes):
            raise ToolpackValidationError(
                f"Toolpack {tool_id} env.passthrough must be a list of strings"
            )
        for entry in passthrough:
            if not isinstance(entry, str) or not entry:
                raise ToolpackValidationError(
                    f"Toolpack {tool_id} env.passthrough must contain non-empty strings"
                )

    return env


def _validate_templating(value: Any, tool_id: str) -> Mapping[str, Any]:
    templating = _require_mapping(value, "templating", tool_id)

    engine = templating.get("engine")
    if engine is not None:
        if not isinstance(engine, str) or not engine:
            raise ToolpackValidationError(
                f"Toolpack {tool_id} templating.engine must be a non-empty string"
            )

    cache_key = templating.get("cacheKey")
    if cache_key is not None:
        if not isinstance(cache_key, str) or not cache_key:
            raise ToolpackValidationError(
                f"Toolpack {tool_id} templating.cacheKey must be a non-empty string"
            )

    return templating


def _validate_execution(value: Any, tool_id: str) -> Mapping[str, Any]:
    if value is None:
        raise ToolpackValidationError(
            f"Toolpack {tool_id} execution must be defined"
        )

    execution = _require_mapping(value, "execution", tool_id)

    kind = execution.get("kind")
    if kind not in _VALID_EXECUTION_KINDS:
        raise ToolpackValidationError(
            f"Toolpack {tool_id} execution.kind must be one of {_VALID_EXECUTION_KINDS}"
        )

    _EXECUTION_VALIDATORS[kind](execution, tool_id)
    return execution


def _validate_python_execution(execution: Mapping[str, Any], tool_id: str) -> None:
    module = execution.get("module")
    script = execution.get("script")

    if module is None and script is None:
        raise ToolpackValidationError(
            f"Toolpack {tool_id} python execution requires 'module' or 'script'"
        )

    if module is not None:
        if not isinstance(module, str) or not module:
            raise ToolpackValidationError(
                f"Toolpack {tool_id} python execution module must be a non-empty string"
            )
        if ":" not in module:
            raise ToolpackValidationError(
                f"Toolpack {tool_id} python module entrypoint must use 'module:callable'"
            )

    if script is not None:
        if not isinstance(script, str) or not script:
            raise ToolpackValidationError(
                f"Toolpack {tool_id} python execution script must be a non-empty string"
            )


def _validate_cli_execution(execution: Mapping[str, Any], tool_id: str) -> None:
    cmd = execution.get("cmd")
    if not isinstance(cmd, Sequence) or isinstance(cmd, str | bytes) or not cmd:
        raise ToolpackValidationError(
            f"Toolpack {tool_id} cli execution requires a non-empty 'cmd' list of strings"
        )

    for idx, part in enumerate(cmd):
        if not isinstance(part, str) or not part:
            raise ToolpackValidationError(
                f"Toolpack {tool_id} cli execution cmd[{idx}] must be a non-empty string"
            )


def _validate_node_execution(execution: Mapping[str, Any], tool_id: str) -> None:
    entry = execution.get("script") or execution.get("node")
    if entry is None:
        raise ToolpackValidationError(
            f"Toolpack {tool_id} node execution requires 'script' or 'node' entrypoint"
        )

    if not isinstance(entry, str) or not entry:
        raise ToolpackValidationError(
            f"Toolpack {tool_id} node execution entrypoint must be a non-empty string"
        )

    args = execution.get("args")
    if args is not None:
        if not isinstance(args, Sequence) or isinstance(args, str | bytes):
            raise ToolpackValidationError(
                f"Toolpack {tool_id} node execution args must be a list of strings"
            )
        for idx, arg in enumerate(args):
            if not isinstance(arg, str) or not arg:
                raise ToolpackValidationError(
                    f"Toolpack {tool_id} node execution args[{idx}] must be a non-empty string"
                )


def _validate_php_execution(execution: Mapping[str, Any], tool_id: str) -> None:
    script = execution.get("php") or execution.get("script")
    if script is None:
        raise ToolpackValidationError(
            f"Toolpack {tool_id} php execution requires 'php' or 'script' entrypoint"
        )

    if not isinstance(script, str) or not script:
        raise ToolpackValidationError(
            f"Toolpack {tool_id} php execution entrypoint must be a non-empty string"
        )

    php_binary = execution.get("phpBinary")
    if php_binary is not None and (not isinstance(php_binary, str) or not php_binary):
        raise ToolpackValidationError(
            f"Toolpack {tool_id} php execution phpBinary must be a non-empty string"
        )


def _validate_http_execution(execution: Mapping[str, Any], tool_id: str) -> None:
    url = execution.get("url")
    if not isinstance(url, str) or not url:
        raise ToolpackValidationError(
            f"Toolpack {tool_id} http execution requires a non-empty 'url'"
        )

    method = execution.get("method")
    if method is not None and (not isinstance(method, str) or not method):
        raise ToolpackValidationError(
            f"Toolpack {tool_id} http execution method must be a non-empty string"
        )

    headers = execution.get("headers")
    if headers is not None:
        if not isinstance(headers, Mapping):
            raise ToolpackValidationError(
                f"Toolpack {tool_id} http execution headers must be a mapping"
            )
        for key, value in headers.items():
            if not isinstance(key, str) or not key:
                raise ToolpackValidationError(
                    f"Toolpack {tool_id} http execution headers must use non-empty string keys"
                )
            if not isinstance(value, str):
                raise ToolpackValidationError(
                    f"Toolpack {tool_id} http execution header '{key}' must be a string value"
                )


_EXECUTION_VALIDATORS: dict[str, Callable[[Mapping[str, Any], str], None]] = {
    "python": _validate_python_execution,
    "cli": _validate_cli_execution,
    "node": _validate_node_execution,
    "php": _validate_php_execution,
    "http": _validate_http_execution,
}


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
