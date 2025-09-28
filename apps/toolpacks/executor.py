from __future__ import annotations

import copy
import importlib
import inspect
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from jsonschema import ValidationError, validate

from .loader import Toolpack, ToolpackLoader


@dataclass(frozen=True)
class ExecutionContext:
    """Runtime context passed to toolpack handlers."""

    env: Mapping[str, str]


class ToolpackExecutor:
    """Execute python toolpacks with schema validation and caching."""

    def __init__(
        self,
        *,
        loader: ToolpackLoader | None = None,
        base_environment: Mapping[str, str] | None = None,
    ) -> None:
        self._loader = loader
        self._base_env = {str(k): str(v) for k, v in (base_environment or {}).items()}
        self._cache: dict[tuple[str, str], dict[str, Any]] = {}

    def run(self, tool_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        if self._loader is None:
            raise ValueError(
                "ToolpackExecutor was constructed without a loader; "
                "use run_toolpack or provide a loader."
            )
        toolpack = self._loader.get(tool_id)
        return self.run_toolpack(toolpack, payload)

    def run_toolpack(
        self,
        toolpack: Toolpack,
        payload: Mapping[str, Any],
        *,
        context: ExecutionContext | None = None,
    ) -> dict[str, Any]:
        if toolpack.config.get("kind", "python") != "python":
            raise ValueError(
                "Unsupported toolpack kind "
                f"'{toolpack.config.get('kind')}'. Only 'python' toolpacks are supported."
            )

        execution = toolpack.config.get("execution")
        if not isinstance(execution, Mapping):
            raise ValueError("Toolpack execution block must be a mapping")

        handler_path = execution.get("handler")
        if not isinstance(handler_path, str) or ":" not in handler_path:
            raise ValueError(
                "Python toolpack execution.handler must be in 'module:callable' format"
            )

        runtime = execution.get("runtime", "python")
        if runtime != "python":
            raise ValueError(f"Unsupported python runtime '{runtime}'.")

        env = self._build_environment(toolpack)
        if context is not None:
            env.update({str(k): str(v) for k, v in context.env.items()})
        context_obj = ExecutionContext(env=env)

        self._validate_schema(toolpack.input_schema, payload, label="input payload")

        cache_key = None
        if bool(toolpack.config.get("deterministic", False)):
            cache_key = self._make_cache_key(toolpack.id, payload, env)
            cached = self._cache.get(cache_key)
            if cached is not None:
                return copy.deepcopy(cached)

        handler = self._load_handler(handler_path)

        try:
            result = self._invoke_handler(handler, payload, context_obj)
        except Exception as exc:  # pragma: no cover - propagate tool runtime errors with context
            raise RuntimeError(f"Toolpack '{toolpack.id}' failed: {exc}") from exc

        if not isinstance(result, Mapping):
            raise ValueError("Toolpack handler must return a mapping")

        result_dict = dict(result)
        self._validate_schema(toolpack.output_schema, result_dict, label="output payload")

        if cache_key is not None:
            self._cache[cache_key] = copy.deepcopy(result_dict)

        return result_dict

    def _build_environment(self, toolpack: Toolpack) -> dict[str, str]:
        env = dict(self._base_env)
        raw_env = toolpack.config.get("env", {})
        if raw_env:
            if not isinstance(raw_env, Mapping):
                raise ValueError("Toolpack env must be a mapping")
            env.update({str(k): str(v) for k, v in raw_env.items()})
        return env

    def _load_handler(self, handler_path: str) -> Any:
        module_name, attr_path = handler_path.split(":", 1)
        module = importlib.import_module(module_name)
        target = module
        for attr in attr_path.split("."):
            target = getattr(target, attr)
        return target

    def _invoke_handler(
        self,
        handler: Any,
        payload: Mapping[str, Any],
        context: ExecutionContext,
    ) -> Any:
        signature = inspect.signature(handler)
        params = list(signature.parameters.values())

        if not params:
            return handler()

        first = params[0]
        if (
            len(params) == 1
            and first.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ):
            return handler(payload)

        keyword_only_names = {
            parameter.name
            for parameter in params
            if parameter.kind == inspect.Parameter.KEYWORD_ONLY
        }
        if "context" in keyword_only_names:
            return handler(payload, context=context)

        return handler(payload, context)

    def _validate_schema(
        self,
        schema: Mapping[str, Any],
        payload: Mapping[str, Any],
        *,
        label: str,
    ) -> None:
        if not schema:
            return
        try:
            validate(instance=payload, schema=schema)
        except ValidationError as exc:
            raise ValueError(f"{label} does not conform to schema: {exc.message}") from exc

    def _make_cache_key(
        self,
        tool_id: str,
        payload: Mapping[str, Any],
        env: Mapping[str, str],
    ) -> tuple[str, str]:
        key_material = {
            "payload": payload,
            "env": dict(sorted(env.items())),
        }
        serialized = json.dumps(key_material, sort_keys=True, default=self._json_default)
        return tool_id, serialized

    @staticmethod
    def _json_default(obj: Any) -> str:
        return repr(obj)


__all__ = ["ToolpackExecutor", "ExecutionContext"]
