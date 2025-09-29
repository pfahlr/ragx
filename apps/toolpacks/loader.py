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


class ToolpackValidationError(Exception):
    """Raised when a Toolpack definition fails validation."""


_VALID_EXECUTION_KINDS = {"python", "node", "php", "cli", "http"}
_SEMVER_PATTERN = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)
_ALLOWED_CAP_KEYS = {"network", "filesystem", "subprocess"}
_ALLOWED_NETWORK_PROTOCOLS = {"http", "https"}
_ALLOWED_TEMPLATING_ENGINES = {"jinja2"}
_ENV_VAR_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")


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
        _validate_semver(version, tool_id)
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

        caps_raw = data.get("caps", {})
        caps_mapping = _require_mapping(caps_raw, "caps", tool_id)
        caps = _validate_caps(caps_mapping, tool_id)

        env_raw = data.get("env", {})
        env_mapping = _require_mapping(env_raw, "env", tool_id)
        env = _validate_env(env_mapping, tool_id)

        templating_raw = data.get("templating", {})
        templating_mapping = _require_mapping(templating_raw, "templating", tool_id)
        templating = _validate_templating(templating_mapping, tool_id)

        input_schema = _resolve_schema(data["inputSchema"], source_path.parent, tool_id)
        output_schema = _resolve_schema(data["outputSchema"], source_path.parent, tool_id)

        execution_raw = _require_mapping(data["execution"], "execution", tool_id)
        execution = _validate_execution(execution_raw, tool_id)

        return cls(
            id=tool_id,
            version=version,
            deterministic=deterministic,
            timeout_ms=timeout_ms,
            limits=dict(limits),
            input_schema=input_schema,
            output_schema=output_schema,
            execution=execution,
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


def _validate_semver(version: str, tool_id: str) -> None:
    if not _SEMVER_PATTERN.match(version):
        raise ToolpackValidationError(
            f"Toolpack {tool_id} version must follow SemVer (e.g. 1.2.3)"
        )


def _validate_execution(execution: Mapping[str, Any], tool_id: str) -> dict[str, Any]:
    kind = execution.get("kind")
    if kind not in _VALID_EXECUTION_KINDS:
        raise ToolpackValidationError(
            f"Toolpack {tool_id} execution.kind must be one of {_VALID_EXECUTION_KINDS}"
        )

    if kind == "python":
        module = execution.get("module")
        script = execution.get("script")
        if module is None and script is None:
            raise ToolpackValidationError(
                f"Toolpack {tool_id} python execution requires 'module' or 'script'"
            )
        if module is not None:
            if not isinstance(module, str) or ":" not in module:
                raise ToolpackValidationError(
                    f"Toolpack {tool_id} python module must be 'pkg.module:callable'"
                )
        if script is not None and (not isinstance(script, str) or not script):
            raise ToolpackValidationError(
                f"Toolpack {tool_id} python script must be a non-empty string"
            )
    elif kind == "cli":
        cmd = execution.get("cmd")
        if not isinstance(cmd, list) or not cmd:
            raise ToolpackValidationError(
                f"Toolpack {tool_id} cli execution requires 'cmd' list"
            )
        for index, part in enumerate(cmd):
            if not isinstance(part, str) or not part:
                raise ToolpackValidationError(
                    f"Toolpack {tool_id} cli cmd[{index}] must be a non-empty string"
                )
    elif kind == "http":
        url = execution.get("url")
        if not isinstance(url, str) or not url:
            raise ToolpackValidationError(
                f"Toolpack {tool_id} http execution requires 'url' string"
            )
        method = execution.get("method")
        if method is not None and (not isinstance(method, str) or not method):
            raise ToolpackValidationError(
                f"Toolpack {tool_id} http method must be a non-empty string"
            )
        headers = execution.get("headers")
        if headers is not None:
            if not isinstance(headers, Mapping):
                raise ToolpackValidationError(
                    f"Toolpack {tool_id} http headers must be a mapping"
                )
            for header, value in headers.items():
                if not isinstance(header, str) or not header:
                    raise ToolpackValidationError(
                        f"Toolpack {tool_id} http header names must be non-empty strings"
                    )
                if not isinstance(value, str):
                    raise ToolpackValidationError(
                        f"Toolpack {tool_id} http header '{header}' must map to a string"
                    )
        timeout = execution.get("timeoutMs")
        if timeout is not None and (not isinstance(timeout, int) or timeout <= 0):
            raise ToolpackValidationError(
                f"Toolpack {tool_id} http timeoutMs must be a positive integer"
            )
    elif kind == "node":
        entry = execution.get("node")
        if not isinstance(entry, str) or not entry:
            raise ToolpackValidationError(
                f"Toolpack {tool_id} node execution requires 'node' script path"
            )
    elif kind == "php":
        script = execution.get("php")
        if not isinstance(script, str) or not script:
            raise ToolpackValidationError(
                f"Toolpack {tool_id} php execution requires 'php' script path"
            )
        binary = execution.get("phpBinary")
        if binary is not None and (not isinstance(binary, str) or not binary):
            raise ToolpackValidationError(
                f"Toolpack {tool_id} phpBinary must be a non-empty string"
            )

    return dict(execution)


def _validate_caps(caps: Mapping[str, Any], tool_id: str) -> dict[str, Any]:
    validated: dict[str, Any] = {}
    for key, value in caps.items():
        if key not in _ALLOWED_CAP_KEYS:
            raise ToolpackValidationError(
                f"Toolpack {tool_id} caps contains unknown key '{key}'"
            )
        if key == "network":
            if not isinstance(value, list) or not value:
                raise ToolpackValidationError(
                    f"Toolpack {tool_id} caps.network must be a non-empty list"
                )
            entries: list[str] = []
            for idx, protocol in enumerate(value):
                if not isinstance(protocol, str) or not protocol:
                    raise ToolpackValidationError(
                        f"Toolpack {tool_id} caps.network[{idx}] must be a string"
                    )
                lowered = protocol.lower()
                if lowered not in _ALLOWED_NETWORK_PROTOCOLS:
                    raise ToolpackValidationError(
                        f"Toolpack {tool_id} caps.network[{idx}] unsupported protocol '{protocol}'"
                    )
                entries.append(lowered)
            validated[key] = entries
        elif key == "filesystem":
            if not isinstance(value, Mapping):
                raise ToolpackValidationError(
                    f"Toolpack {tool_id} caps.filesystem must be a mapping"
                )
            fs_allowed = {"read", "write"}
            fs_validated: dict[str, list[str]] = {}
            for mode, paths in value.items():
                if mode not in fs_allowed:
                    raise ToolpackValidationError(
                        f"Toolpack {tool_id} caps.filesystem[{mode}] is not supported"
                    )
                if not isinstance(paths, list):
                    raise ToolpackValidationError(
                        f"Toolpack {tool_id} caps.filesystem[{mode}] must be a list"
                    )
                normalised: list[str] = []
                for idx, path in enumerate(paths):
                    if not isinstance(path, str) or not path:
                        raise ToolpackValidationError(
                            f"Toolpack {tool_id} caps.filesystem[{mode}] entry {idx} "
                            "must be a non-empty string"
                        )
                    normalised.append(path)
                fs_validated[mode] = normalised
            validated[key] = fs_validated
        elif key == "subprocess":
            if not isinstance(value, bool):
                raise ToolpackValidationError(
                    f"Toolpack {tool_id} caps.subprocess must be a boolean"
                )
            validated[key] = value
    return validated


def _validate_env(env: Mapping[str, Any], tool_id: str) -> dict[str, Any]:
    validated: dict[str, Any] = {}
    for key, value in env.items():
        if key == "passthrough":
            if not isinstance(value, list):
                raise ToolpackValidationError(
                    f"Toolpack {tool_id} env.passthrough must be a list"
                )
            names: list[str] = []
            for idx, item in enumerate(value):
                if not isinstance(item, str) or not item:
                    raise ToolpackValidationError(
                        f"Toolpack {tool_id} env.passthrough[{idx}] must be a non-empty string"
                    )
                if not _ENV_VAR_PATTERN.fullmatch(item):
                    raise ToolpackValidationError(
                        f"Toolpack {tool_id} env.passthrough[{idx}] must be "
                        "uppercase A-Z, 0-9, or '_'"
                    )
                names.append(item)
            validated[key] = names
        elif key == "set":
            if not isinstance(value, Mapping):
                raise ToolpackValidationError(
                    f"Toolpack {tool_id} env.set must be a mapping"
                )
            assignments: dict[str, str] = {}
            for env_name, env_value in value.items():
                if not isinstance(env_name, str) or not env_name:
                    raise ToolpackValidationError(
                        f"Toolpack {tool_id} env.set keys must be non-empty strings"
                    )
                if not _ENV_VAR_PATTERN.fullmatch(env_name):
                    raise ToolpackValidationError(
                        f"Toolpack {tool_id} env.set key '{env_name}' must be "
                        "uppercase A-Z, 0-9, or '_'"
                    )
                if not isinstance(env_value, str):
                    raise ToolpackValidationError(
                        f"Toolpack {tool_id} env.set['{env_name}'] must be a string"
                    )
                assignments[env_name] = env_value
            validated[key] = assignments
        else:
            raise ToolpackValidationError(
                f"Toolpack {tool_id} env contains unknown key '{key}'"
            )
    return validated


def _validate_templating(templating: Mapping[str, Any], tool_id: str) -> dict[str, Any]:
    validated: dict[str, Any] = {}
    for key in templating.keys():
        if key not in {"engine", "cacheKey", "context"}:
            raise ToolpackValidationError(
                f"Toolpack {tool_id} templating contains unknown key '{key}'"
            )

    engine = templating.get("engine")
    if engine is not None:
        if not isinstance(engine, str) or engine not in _ALLOWED_TEMPLATING_ENGINES:
            raise ToolpackValidationError(
                f"Toolpack {tool_id} templating.engine must be one of {_ALLOWED_TEMPLATING_ENGINES}"
            )
        validated["engine"] = engine

    cache_key = templating.get("cacheKey")
    if cache_key is not None:
        if not isinstance(cache_key, str) or not cache_key:
            raise ToolpackValidationError(
                f"Toolpack {tool_id} templating.cacheKey must be a non-empty string"
            )
        validated["cacheKey"] = cache_key

    context = templating.get("context")
    if context is not None:
        if not isinstance(context, Mapping):
            raise ToolpackValidationError(
                f"Toolpack {tool_id} templating.context must be a mapping"
            )
        try:
            json.dumps(context)
        except (TypeError, ValueError) as exc:
            raise ToolpackValidationError(
                f"Toolpack {tool_id} templating.context must be JSON serialisable: {exc}"
            ) from exc
        validated["context"] = dict(context)

    return validated


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
