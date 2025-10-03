from __future__ import annotations

import copy
import json
import logging
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

LOGGER = logging.getLogger(__name__)

_LEGACY_TOP_LEVEL_KEYS = {
    "timeout_ms": "timeoutMs",
    "input_schema": "inputSchema",
    "output_schema": "outputSchema",
}

_LEGACY_LIMIT_KEYS = {
    "max_input_bytes": "maxInputBytes",
    "max_output_bytes": "maxOutputBytes",
}

_TOOL_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:\.[a-z0-9]+)+$")
_ALLOWED_CAP_KEYS = {"network", "filesystem", "subprocess"}
_ALLOWED_NETWORK_PROTOCOLS = {"http", "https"}
_ENV_VAR_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")
_ALLOWED_TEMPLATING_ENGINES = {"jinja2"}


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
    def from_dict(
        cls,
        data: Mapping[str, Any],
        source_path: Path,
        *,
        schema_cache: dict[tuple[Path, str], Any],
    ) -> Toolpack:
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

        caps = _validate_caps(data.get("caps"), tool_id)

        env = _validate_env(data.get("env"), tool_id)

        templating = _validate_templating(data.get("templating"), tool_id)

        input_schema = _resolve_schema(
            data["inputSchema"],
            source_path.parent,
            tool_id,
            schema_cache,
        )
        output_schema = _resolve_schema(
            data["outputSchema"],
            source_path.parent,
            tool_id,
            schema_cache,
        )

        execution = _validate_execution(tool_id, data["execution"])

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
        base_dir = Path(directory).expanduser().resolve()
        if not base_dir.exists():
            raise ToolpackValidationError(f"Toolpacks directory not found: {base_dir}")

        toolpacks: dict[str, Toolpack] = {}
        schema_cache: dict[tuple[Path, str], Any] = {}
        for path in sorted(base_dir.rglob("*.tool.yaml")):
            with path.open("r", encoding="utf-8") as handle:
                try:
                    data = yaml.safe_load(handle)
                except yaml.YAMLError as exc:
                    raise ToolpackValidationError(
                        f"Failed to parse YAML for toolpack {path}: {exc}"
                    ) from exc

            data = _apply_legacy_shim(data, path)

            toolpack = Toolpack.from_dict(
                data,
                path,
                schema_cache=schema_cache,
            )
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
    schema_spec: Any,
    base_dir: Path,
    tool_id: str,
    cache: dict[tuple[Path, str], Any],
) -> Mapping[str, Any]:
    if not isinstance(schema_spec, Mapping):
        raise ToolpackValidationError(
            f"Toolpack {tool_id} schema definition must be a mapping"
        )

    resolved = _resolve_refs(
        schema_spec,
        base_dir,
        cache,
        tool_id,
        current_document=schema_spec,
    )
    _validate_json_schema(resolved, tool_id)
    return resolved


def _resolve_refs(
    node: Any,
    base_dir: Path,
    cache: dict[tuple[Path, str], Any],
    tool_id: str,
    *,
    current_path: Path | None = None,
    current_document: Any | None = None,
) -> Any:
    if isinstance(node, Mapping):
        if set(node.keys()) == {"$ref"}:
            ref = node.get("$ref")
            if not isinstance(ref, str) or not ref:
                raise ToolpackValidationError(
                    f"Toolpack {tool_id} schema $ref must be a non-empty string"
                )
            path_part, fragment = _split_reference(ref)
            if not path_part and current_document is not None:
                fragment_data = (
                    _apply_json_pointer(current_document, fragment)
                    if fragment
                    else current_document
                )
                return _resolve_refs(
                    fragment_data,
                    base_dir,
                    cache,
                    tool_id,
                    current_path=current_path,
                    current_document=current_document,
                )
            return _load_ref(
                ref,
                base_dir,
                cache,
                tool_id,
                current_path=current_path,
                current_document=current_document,
            )
        return {
            key: _resolve_refs(
                value,
                base_dir,
                cache,
                tool_id,
                current_path=current_path,
                current_document=current_document,
            )
            for key, value in node.items()
        }
    if isinstance(node, list):
        return [
            _resolve_refs(
                item,
                base_dir,
                cache,
                tool_id,
                current_path=current_path,
                current_document=current_document,
            )
            for item in node
        ]
    return node


def _load_ref(
    reference: str,
    base_dir: Path,
    cache: dict[tuple[Path, str], Any],
    tool_id: str,
    *,
    current_path: Path | None = None,
    current_document: Any | None = None,
) -> Any:
    path_part, fragment = _split_reference(reference)
    if path_part:
        target_path = Path(path_part)
        if not target_path.is_absolute():
            target_path = (base_dir / target_path).resolve()
    else:
        if current_path is None:
            if current_document is None:
                raise ToolpackValidationError(
                    f"Toolpack {tool_id} schema reference lacks base path: {reference}"
                )
            fragment_data = (
                _apply_json_pointer(current_document, fragment)
                if fragment
                else current_document
            )
            resolved_inline = _resolve_refs(
                fragment_data,
                base_dir,
                cache,
                tool_id,
                current_path=current_path,
                current_document=current_document,
            )
            return copy.deepcopy(resolved_inline)
        target_path = current_path

    cache_key = (target_path, fragment or "")
    if cache_key in cache:
        return copy.deepcopy(cache[cache_key])

    if not target_path.exists():
        raise ToolpackValidationError(
            f"Toolpack {tool_id} schema reference not found: {reference}"
        )

    try:
        with target_path.open("r", encoding="utf-8") as handle:
            if target_path.suffix in {".yaml", ".yml"}:
                try:
                    document = yaml.safe_load(handle) or {}
                except yaml.YAMLError as exc:
                    raise ToolpackValidationError(
                        f"Toolpack {tool_id} failed to parse schema {reference}: {exc}"
                    ) from exc
            else:
                try:
                    document = json.load(handle)
                except json.JSONDecodeError as exc:
                    raise ToolpackValidationError(
                        f"Toolpack {tool_id} failed to load schema {reference}: {exc}"
                    ) from exc
    except OSError as exc:
        raise ToolpackValidationError(
            f"Toolpack {tool_id} failed to open schema {reference}: {exc}"
        ) from exc

    fragment_data = _apply_json_pointer(document, fragment) if fragment else document
    resolved = _resolve_refs(
        fragment_data,
        target_path.parent,
        cache,
        tool_id,
        current_path=target_path,
        current_document=document,
    )
    cache[cache_key] = copy.deepcopy(resolved)
    return copy.deepcopy(resolved)


def _split_reference(reference: str) -> tuple[str, str | None]:
    if "#" not in reference:
        return reference, None
    path_part, fragment = reference.split("#", 1)
    if not fragment:
        return path_part, None
    if not fragment.startswith("/"):
        fragment = "/" + fragment
    return path_part, fragment


def _apply_json_pointer(document: Any, pointer: str | None) -> Any:
    if not pointer:
        return document

    parts = pointer.lstrip("/").split("/") if pointer != "/" else []
    current = document
    for raw_part in parts:
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            index = int(part)
            current = current[index]
        elif isinstance(current, Mapping):
            current = current[part]
        else:  # pragma: no cover - defensive
            raise KeyError(f"Cannot apply pointer '{pointer}' to non-container value")
    return current


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


def _apply_legacy_shim(data: Any, source_path: Path) -> Any:
    if not isinstance(data, Mapping):
        return data

    updated = dict(data)
    rewritten: list[str] = []

    for legacy_key, modern_key in _LEGACY_TOP_LEVEL_KEYS.items():
        if legacy_key in updated and modern_key not in updated:
            updated[modern_key] = updated.pop(legacy_key)
            rewritten.append(legacy_key)

    limits = updated.get("limits")
    if isinstance(limits, Mapping):
        limits_copy = dict(limits)
        changed = False
        for legacy_key, modern_key in _LEGACY_LIMIT_KEYS.items():
            if legacy_key in limits_copy and modern_key not in limits_copy:
                limits_copy[modern_key] = limits_copy.pop(legacy_key)
                rewritten.append(f"limits.{legacy_key}")
                changed = True
        if changed:
            updated["limits"] = limits_copy

    if rewritten:
        LOGGER.warning(
            "Toolpack %s used legacy snake_case keys; converted to camelCase (%s).",
            source_path,
            ", ".join(sorted(rewritten)),
        )

    return updated


def _validate_tool_id(tool_id: str, source_path: Path) -> None:
    candidate = tool_id.replace(":", ".")
    if not _TOOL_ID_PATTERN.fullmatch(candidate):
        message = (
            f"Toolpack {source_path} id '{tool_id}' must use dotted lowercase segments "
            "(e.g. 'pkg.tool')"
        )
        raise ToolpackValidationError(message)


def _validate_version(version: str, tool_id: str) -> None:
    try:
        parsed = Version(version)
    except InvalidVersion as exc:
        raise ToolpackValidationError(
            f"Toolpack {tool_id} version must follow semantic versioning: {exc}"
        ) from exc

    if len(parsed.release) != 3:
        raise ToolpackValidationError(
            f"Toolpack {tool_id} version '{version}' must include major.minor.patch"
        )


def _validate_caps(value: Any, tool_id: str) -> dict[str, Any]:
    if value is None:
        return {}

    caps_mapping = _require_mapping(value, "caps", tool_id)
    validated: dict[str, Any] = {}

    for key in caps_mapping:
        if key not in _ALLOWED_CAP_KEYS:
            raise ToolpackValidationError(
                f"Toolpack {tool_id} caps contains unknown key '{key}'"
            )

    if "network" in caps_mapping:
        network_value = caps_mapping["network"]
        entries: list[str]
        if isinstance(network_value, list):
            entries = network_value
        elif isinstance(network_value, str):
            entries = [network_value]
        else:
            raise ToolpackValidationError(
                f"Toolpack {tool_id} caps.network must be a string or list of strings"
            )
        normalised: list[str] = []
        for idx, entry in enumerate(entries):
            if not isinstance(entry, str) or not entry.strip():
                raise ToolpackValidationError(
                    f"Toolpack {tool_id} caps.network[{idx}] must be a non-empty string"
                )
            protocol = entry.strip().lower()
            if protocol not in _ALLOWED_NETWORK_PROTOCOLS:
                raise ToolpackValidationError(
                    f"Toolpack {tool_id} caps.network[{idx}] unsupported protocol '{entry}'"
                )
            normalised.append(protocol)
        validated["network"] = normalised

    if "filesystem" in caps_mapping:
        fs_mapping = _require_mapping(caps_mapping["filesystem"], "caps.filesystem", tool_id)
        allowed_modes = {"read", "write"}
        fs_validated: dict[str, list[str]] = {}
        for mode, paths in fs_mapping.items():
            if mode not in allowed_modes:
                raise ToolpackValidationError(
                    f"Toolpack {tool_id} caps.filesystem[{mode}] is not supported"
                )
            if not isinstance(paths, list):
                raise ToolpackValidationError(
                    f"Toolpack {tool_id} caps.filesystem[{mode}] must be a list"
                )
            cleaned: list[str] = []
            for idx, path in enumerate(paths):
                if not isinstance(path, str) or not path:
                    message = (
                        f"Toolpack {tool_id} caps.filesystem[{mode}][{idx}] must "
                        "be a non-empty string"
                    )
                    raise ToolpackValidationError(message)
                cleaned.append(path)
            fs_validated[mode] = cleaned
        if fs_validated:
            validated["filesystem"] = fs_validated

    if "subprocess" in caps_mapping:
        subprocess_value = caps_mapping["subprocess"]
        if not isinstance(subprocess_value, bool):
            raise ToolpackValidationError(
                f"Toolpack {tool_id} caps.subprocess must be a boolean"
            )
        validated["subprocess"] = subprocess_value

    return validated


def _validate_env(value: Any, tool_id: str) -> dict[str, Any]:
    if value is None:
        return {}

    env_mapping = _require_mapping(value, "env", tool_id)
    validated: dict[str, Any] = {}

    for key in env_mapping:
        if key not in {"passthrough", "set"}:
            raise ToolpackValidationError(
                f"Toolpack {tool_id} env contains unknown key '{key}'"
            )

    if "passthrough" in env_mapping:
        passthrough = env_mapping["passthrough"]
        if not isinstance(passthrough, list):
            raise ToolpackValidationError(
                f"Toolpack {tool_id} env.passthrough must be a list"
            )
        names: list[str] = []
        for idx, name in enumerate(passthrough):
            if not isinstance(name, str) or not name:
                raise ToolpackValidationError(
                    f"Toolpack {tool_id} env.passthrough[{idx}] must be a non-empty string"
                )
            if not _ENV_VAR_PATTERN.fullmatch(name):
                raise ToolpackValidationError(
                    f"Toolpack {tool_id} env.passthrough[{idx}] must be uppercase A-Z, 0-9, or '_'"
                )
            names.append(name)
        validated["passthrough"] = names

    if "set" in env_mapping:
        set_mapping = _require_mapping(env_mapping["set"], "env.set", tool_id)
        assignments: dict[str, str] = {}
        for name, value in set_mapping.items():
            if not isinstance(name, str) or not name:
                raise ToolpackValidationError(
                    f"Toolpack {tool_id} env.set keys must be non-empty strings"
                )
            if not _ENV_VAR_PATTERN.fullmatch(name):
                raise ToolpackValidationError(
                    f"Toolpack {tool_id} env.set key '{name}' must be uppercase A-Z, 0-9, or '_'"
                )
            if not isinstance(value, str):
                raise ToolpackValidationError(
                    f"Toolpack {tool_id} env.set['{name}'] must be a string"
                )
            assignments[name] = value
        validated["set"] = assignments

    return validated


def _validate_templating(value: Any, tool_id: str) -> dict[str, Any]:
    if value is None:
        return {}

    templating_mapping = _require_mapping(value, "templating", tool_id)
    validated: dict[str, Any] = {}

    for key in templating_mapping:
        if key not in {"engine", "cacheKey", "context"}:
            raise ToolpackValidationError(
                f"Toolpack {tool_id} templating contains unknown key '{key}'"
            )

    if "engine" in templating_mapping:
        engine = templating_mapping["engine"]
        if not isinstance(engine, str) or engine not in _ALLOWED_TEMPLATING_ENGINES:
            raise ToolpackValidationError(
                f"Toolpack {tool_id} templating.engine must be one of {_ALLOWED_TEMPLATING_ENGINES}"
            )
        validated["engine"] = engine

    if "cacheKey" in templating_mapping:
        cache_key = templating_mapping["cacheKey"]
        if not isinstance(cache_key, str) or not cache_key:
            raise ToolpackValidationError(
                f"Toolpack {tool_id} templating.cacheKey must be a non-empty string"
            )
        validated["cacheKey"] = cache_key

    if "context" in templating_mapping:
        context_mapping = _require_mapping(
            templating_mapping["context"],
            "templating.context",
            tool_id,
        )
        try:
            json.dumps(context_mapping)
        except (TypeError, ValueError) as exc:
            raise ToolpackValidationError(
                f"Toolpack {tool_id} templating.context must be JSON serialisable: {exc}"
            ) from exc
        validated["context"] = dict(context_mapping)

    return validated


def _validate_execution(tool_id: str, execution_raw: Any) -> dict[str, Any]:
    execution = _require_mapping(execution_raw, "execution", tool_id)
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
            if not isinstance(module, str) or not module:
                raise ToolpackValidationError(
                    f"Toolpack {tool_id} execution.module must be a non-empty string"
                )
            if ":" not in module or module.startswith(":") or module.endswith(":"):
                raise ToolpackValidationError(
                    f"Toolpack {tool_id} execution.module must use 'module:callable' format"
                )
        if script is not None and (not isinstance(script, str) or not script):
            raise ToolpackValidationError(
                f"Toolpack {tool_id} execution.script must be a non-empty string"
            )

    elif kind == "cli":
        cmd = execution.get("cmd")
        if not isinstance(cmd, list) or not cmd:
            raise ToolpackValidationError(
                f"Toolpack {tool_id} cli execution requires 'cmd' list"
            )
        for idx, part in enumerate(cmd):
            if not isinstance(part, str) or not part:
                raise ToolpackValidationError(
                    f"Toolpack {tool_id} execution.cmd[{idx}] must be a non-empty string"
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
                f"Toolpack {tool_id} http execution.method must be a non-empty string"
            )
        headers = execution.get("headers")
        if headers is not None:
            headers_map = _require_mapping(headers, "execution.headers", tool_id)
            for header, value in headers_map.items():
                if not isinstance(header, str) or not header:
                    raise ToolpackValidationError(
                        f"Toolpack {tool_id} http headers must use non-empty string keys"
                    )
                if not isinstance(value, str):
                    raise ToolpackValidationError(
                        f"Toolpack {tool_id} http header '{header}' must be a string value"
                    )
        timeout = execution.get("timeoutMs")
        if timeout is not None and (not isinstance(timeout, int) or timeout <= 0):
            raise ToolpackValidationError(
                f"Toolpack {tool_id} http timeoutMs must be a positive integer"
            )

    elif kind == "node":
        entry = execution.get("script") or execution.get("node") or execution.get("module")
        if not isinstance(entry, str) or not entry:
            raise ToolpackValidationError(
                f"Toolpack {tool_id} node execution requires 'script', 'node', or 'module' entry"
            )
        args = execution.get("args")
        if args is not None:
            if not isinstance(args, list):
                raise ToolpackValidationError(
                    f"Toolpack {tool_id} node execution args must be a list"
                )
            for idx, arg in enumerate(args):
                if not isinstance(arg, str) or not arg:
                    raise ToolpackValidationError(
                        f"Toolpack {tool_id} node execution args[{idx}] must be a non-empty string"
                    )

    elif kind == "php":
        entry = execution.get("php") or execution.get("script")
        if not isinstance(entry, str) or not entry:
            raise ToolpackValidationError(
                f"Toolpack {tool_id} php execution requires 'php' or 'script' entry"
            )
        binary = execution.get("phpBinary")
        if binary is not None and (not isinstance(binary, str) or not binary):
            raise ToolpackValidationError(
                f"Toolpack {tool_id} php execution phpBinary must be a non-empty string"
            )

    return dict(execution)
