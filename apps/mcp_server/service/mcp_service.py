from __future__ import annotations

import json
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4, uuid5

from jsonschema import Draft202012Validator, validators
from jsonschema.exceptions import ValidationError

from apps.mcp_server.logging import JsonLogWriter
from apps.toolpacks.executor import (
    Executor,
    ToolpackExecutionError,
    ToolpackSchemaValidationError,
)
from apps.toolpacks.loader import Toolpack, ToolpackLoader

from .envelope import Envelope, EnvelopeError, EnvelopeMeta

__all__ = ["McpService", "RequestContext", "ServerLogManager"]

_AGENT_ID = "mcp_server"
_TASK_ID = "06b_mcp_server_bootstrap"
_SCHEMA_VERSION_DEFAULT = "0.1.0"
_DETERMINISTIC_NAMESPACE = UUID("c1fd1c20-77b7-4f73-b39c-8ed2dd2f2d8c")


@dataclass(frozen=True)
class ToolSchemaValidators:
    input: Draft202012Validator
    output: Draft202012Validator


@dataclass(slots=True)
class RequestContext:
    """Per-request metadata supplied by transports."""

    transport: str
    route: str
    method: str
    deterministic_ids: bool = False
    request_payload: Mapping[str, Any] | None = None
    start_time: float = field(default_factory=time.perf_counter)
    attempt: int = 0

    def clone_with_payload(self, payload: Mapping[str, Any]) -> RequestContext:
        return RequestContext(
            transport=self.transport,
            route=self.route,
            method=self.method,
            deterministic_ids=self.deterministic_ids,
            request_payload=payload,
            start_time=time.perf_counter(),
            attempt=self.attempt,
        )


@dataclass(slots=True)
class Prompt:
    prompt_id: str
    name: str
    description: str
    major: int
    tags: list[str]
    messages: list[dict[str, str]]

    def to_discovery_dict(self) -> dict[str, Any]:
        return {
            "id": self.prompt_id,
            "name": self.name,
            "description": self.description,
            "version": {"major": self.major},
            "tags": list(self.tags),
        }

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.prompt_id,
            "name": self.name,
            "description": self.description,
            "version": {"major": self.major},
            "messages": list(self.messages),
        }


class PromptRepository:
    """Load prompt definitions from disk."""

    def __init__(self, base_dir: Path) -> None:
        self._prompts: dict[str, Prompt] = {}
        self._load(base_dir)

    def _load(self, base_dir: Path) -> None:
        directory = Path(base_dir)
        if not directory.exists():
            raise FileNotFoundError(f"Prompts directory not found: {directory}")
        prompts: dict[str, Prompt] = {}
        for path in sorted(directory.rglob("*.prompt.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            prompt = _prompt_from_payload(data, path)
            if prompt.prompt_id in prompts:
                raise ValueError(f"Duplicate prompt id '{prompt.prompt_id}' in {path}")
            prompts[prompt.prompt_id] = prompt
        self._prompts = prompts

    def list(self) -> list[Prompt]:
        return list(self._prompts.values())

    def get(self, prompt_id: str) -> Prompt:
        if prompt_id not in self._prompts:
            raise KeyError(f"Unknown prompt '{prompt_id}'")
        return self._prompts[prompt_id]


class SchemaStore:
    """JSON schema loader with caching."""

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = Path(base_dir)
        self._cache: dict[Path, Draft202012Validator] = {}

    def validator(self, name: str) -> Draft202012Validator:
        path = self._base_dir / name
        if path in self._cache:
            return self._cache[path]
        if not path.exists():
            raise FileNotFoundError(f"Schema not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            schema = json.load(handle)
        validator_cls = validators.validator_for(schema)
        validator_cls.check_schema(schema)
        validator = validator_cls(schema)
        self._cache[path] = validator
        return validator


@dataclass(slots=True)
class ServerLogEvent:
    ts: datetime
    request_id: str
    trace_id: str
    span_id: str
    transport: str
    route: str
    method: str
    status: str
    duration_ms: float
    attempt: int
    input_bytes: int
    output_bytes: int
    metadata: dict[str, Any]
    step_id: int
    error: dict[str, Any] | None = None

    def to_serialisable(self) -> dict[str, Any]:
        return {
            "ts": self.ts.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            "agentId": _AGENT_ID,
            "taskId": _TASK_ID,
            "stepId": self.step_id,
            "transport": self.transport,
            "route": self.route,
            "method": self.method,
            "traceId": self.trace_id,
            "spanId": self.span_id,
            "requestId": self.request_id,
            "status": self.status,
            "durationMs": self.duration_ms,
            "attempt": self.attempt,
            "inputBytes": self.input_bytes,
            "outputBytes": self.output_bytes,
            "metadata": dict(self.metadata),
            "error": self.error,
        }


class ServerLogManager:
    """Wrapper around :class:`JsonLogWriter` for MCP server events."""

    def __init__(
        self,
        *,
        log_dir: Path,
        schema_version: str,
        deterministic: bool,
        retention: int = 5,
    ) -> None:
        self._storage_prefix = Path(log_dir) / "mcp_server" / "bootstrap"
        self._latest_symlink = Path(log_dir) / "mcp_server" / "bootstrap.latest.jsonl"
        self._writer = JsonLogWriter(
            agent_id=_AGENT_ID,
            task_id=_TASK_ID,
            storage_prefix=self._storage_prefix,
            latest_symlink=self._latest_symlink,
            schema_version=schema_version,
            deterministic=deterministic,
            retention=retention,
        )
        self._step_counter = 0

    @property
    def latest_symlink(self) -> Path:
        return self._latest_symlink

    @property
    def writer(self) -> JsonLogWriter:
        return self._writer

    def next_step_id(self) -> int:
        self._step_counter += 1
        return self._step_counter

    def emit(self, event: ServerLogEvent) -> None:
        attempt_id = self._writer.new_attempt_id()
        self._writer.write(event, attempt_id=attempt_id)


class McpService:
    """High-level service implementing discovery, prompts, and tool invocation."""

    def __init__(
        self,
        *,
        toolpacks: dict[str, Toolpack],
        tool_validators: dict[str, ToolSchemaValidators],
        executor: Executor,
        prompts: PromptRepository,
        schema_store: SchemaStore,
        log_manager: ServerLogManager,
        schema_version: str,
    ) -> None:
        self._toolpacks = toolpacks
        self._tool_validators = tool_validators
        self._executor = executor
        self._prompts = prompts
        self._schemas = schema_store
        self._log_manager = log_manager
        self._schema_version = schema_version

    @property
    def log_manager(self) -> ServerLogManager:
        return self._log_manager

    @classmethod
    def create(
        cls,
        *,
        toolpacks_dir: Path,
        prompts_dir: Path,
        schema_dir: Path,
        log_dir: Path,
        schema_version: str = _SCHEMA_VERSION_DEFAULT,
        deterministic_logs: bool = True,
        logger: ServerLogManager | None = None,
    ) -> McpService:
        loader = ToolpackLoader()
        loader.load_dir(toolpacks_dir)
        toolpacks = {pack.id: pack for pack in loader.list()}
        validators_map = {
            toolpack.id: ToolSchemaValidators(
                input=_json_schema_validator_for(toolpack.input_schema),
                output=_json_schema_validator_for(toolpack.output_schema),
            )
            for toolpack in toolpacks.values()
        }
        executor = Executor()
        prompts = PromptRepository(prompts_dir)
        schema_store = SchemaStore(schema_dir)
        log_manager = logger or ServerLogManager(
            log_dir=log_dir,
            schema_version=schema_version,
            deterministic=deterministic_logs,
        )
        return cls(
            toolpacks=toolpacks,
            tool_validators=validators_map,
            executor=executor,
            prompts=prompts,
            schema_store=schema_store,
            log_manager=log_manager,
            schema_version=schema_version,
        )

    def discover(self, context: RequestContext | None = None) -> Envelope:
        payload: dict[str, Any] = {}
        ctx = self._normalise_context(context, "discover", "mcp.discover", payload)
        tools = [self._tool_to_dict(tool) for tool in self._toolpacks.values()]
        prompts = [prompt.to_discovery_dict() for prompt in self._prompts.list()]
        data = {"tools": tools, "prompts": prompts}
        self._schemas.validator("discover.response.schema.json").validate(data)
        envelope = self._finalise_envelope(data, ctx, tool_id=None, prompt_id=None)
        return envelope

    def get_prompt(self, prompt_id: str, context: RequestContext | None = None) -> Envelope:
        payload = {"promptId": prompt_id}
        ctx = self._normalise_context(context, "prompt", "mcp.prompt.get", payload)
        try:
            prompt = self._prompts.get(prompt_id)
        except KeyError:
            return self._error_response(
                code="NOT_FOUND",
                message=f"Prompt '{prompt_id}' not found",
                context=ctx,
                payload=payload,
                prompt_id=prompt_id,
            )
        data = prompt.to_payload()
        self._schemas.validator("prompt.response.schema.json").validate(data)
        return self._finalise_envelope(data, ctx, prompt_id=prompt_id)

    def invoke_tool(
        self,
        *,
        tool_id: str,
        arguments: Mapping[str, Any],
        context: RequestContext | None = None,
    ) -> Envelope:
        argument_payload = dict(arguments)
        payload = {"toolId": tool_id, "arguments": argument_payload}
        ctx = self._normalise_context(context, "tool", "mcp.tool.invoke", payload)
        if tool_id not in self._toolpacks:
            return self._error_response(
                code="NOT_FOUND",
                message=f"Tool '{tool_id}' not found",
                context=ctx,
                payload=payload,
                tool_id=tool_id,
            )
        toolpack = self._toolpacks[tool_id]
        validators = self._tool_validators[tool_id]
        try:
            validators.input.validate(argument_payload)
        except ValidationError as exc:
            return self._error_response(
                code="INVALID_INPUT",
                message=f"Tool '{tool_id}' input failed validation: {exc.message}",
                context=ctx,
                payload=payload,
                tool_id=tool_id,
                details=_schema_error_details(
                    tool_id=tool_id,
                    stage="tool.input",
                    error=exc,
                ),
            )
        try:
            result = self._executor.run_toolpack(toolpack, argument_payload)
        except ToolpackSchemaValidationError as exc:
            stage_code = "INVALID_INPUT" if exc.stage == "input" else "INTERNAL"
            return self._error_response(
                code=stage_code,
                message=str(exc),
                context=ctx,
                payload=payload,
                tool_id=tool_id,
                details=_schema_error_details(
                    tool_id=tool_id,
                    stage=f"tool.{exc.stage}",
                    error=exc.error,
                ),
            )
        except ToolpackExecutionError as exc:
            return self._error_response(
                code="INTERNAL",
                message=str(exc),
                context=ctx,
                payload=payload,
                tool_id=tool_id,
            )
        result_payload = dict(result)
        try:
            validators.output.validate(result_payload)
        except ValidationError as exc:
            return self._error_response(
                code="INTERNAL",
                message=f"Tool '{tool_id}' output failed validation: {exc.message}",
                context=ctx,
                payload=payload,
                tool_id=tool_id,
                details=_schema_error_details(
                    tool_id=tool_id,
                    stage="tool.output",
                    error=exc,
                ),
            )
        data = {
            "toolId": tool_id,
            "result": result_payload,
            "metadata": {
                "toolpack": {
                    "id": toolpack.id,
                    "version": toolpack.version,
                    "deterministic": toolpack.deterministic,
                    "timeoutMs": toolpack.timeout_ms,
                }
            },
        }
        try:
            self._schemas.validator("tool.response.schema.json").validate(data)
        except ValidationError as exc:
            return self._error_response(
                code="INTERNAL",
                message=f"Tool '{tool_id}' response failed validation: {exc.message}",
                context=ctx,
                payload=payload,
                tool_id=tool_id,
                details=_schema_error_details(
                    tool_id=tool_id,
                    stage="tool.response",
                    error=exc,
                ),
            )
        return self._finalise_envelope(data, ctx, tool_id=tool_id)

    def health(self, context: RequestContext | None = None) -> dict[str, Any]:
        _ = self._normalise_context(context, "health", "mcp.health", {})
        return {"status": "ok"}

    # Internal helpers -------------------------------------------------

    def _finalise_envelope(
        self,
        data: Mapping[str, Any],
        context: RequestContext,
        *,
        tool_id: str | None = None,
        prompt_id: str | None = None,
    ) -> Envelope:
        duration_ms = _duration_ms(context.start_time)
        ids = self._request_ids(context)
        step_id = self._log_manager.next_step_id()
        meta = EnvelopeMeta.from_ids(
            request_id=ids["request_id"],
            trace_id=ids["trace_id"],
            span_id=ids["span_id"],
            schema_version=self._schema_version,
            deterministic=context.deterministic_ids,
            transport=context.transport,
            route=context.route,
            method=context.method,
            duration_ms=duration_ms,
            status="ok",
            attempt=context.attempt,
            input_bytes=ids["input_bytes"],
            output_bytes=_payload_size(data),
            tool_id=tool_id,
            prompt_id=prompt_id,
        )
        envelope = Envelope.success(data=dict(data), meta=meta)
        self._schemas.validator("envelope.schema.json").validate(
            envelope.model_dump(by_alias=True)
        )
        self._log_manager.emit(
            ServerLogEvent(
                ts=datetime.now(UTC),
                request_id=meta.request_id,
                trace_id=meta.trace_id,
                span_id=meta.span_id,
                transport=context.transport,
                route=context.route,
                method=context.method,
                status="ok",
                duration_ms=duration_ms,
                attempt=context.attempt,
                input_bytes=ids["input_bytes"],
                output_bytes=_payload_size(data),
                metadata={
                    "toolId": tool_id,
                    "promptId": prompt_id,
                    "schemaVersion": self._schema_version,
                    "deterministic": context.deterministic_ids,
                },
                step_id=step_id,
            )
        )
        return envelope

    def _error_response(
        self,
        *,
        code: str,
        message: str,
        context: RequestContext,
        payload: Mapping[str, Any],
        tool_id: str | None = None,
        prompt_id: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> Envelope:
        ids = self._request_ids(context)
        step_id = self._log_manager.next_step_id()
        duration_ms = _duration_ms(context.start_time)
        meta = EnvelopeMeta.from_ids(
            request_id=ids["request_id"],
            trace_id=ids["trace_id"],
            span_id=ids["span_id"],
            schema_version=self._schema_version,
            deterministic=context.deterministic_ids,
            transport=context.transport,
            route=context.route,
            method=context.method,
            duration_ms=duration_ms,
            status="error",
            attempt=context.attempt,
            input_bytes=ids["input_bytes"],
            output_bytes=0,
            tool_id=tool_id,
            prompt_id=prompt_id,
        )
        normalised_details = (
            {str(key): _normalise_detail_value(value) for key, value in details.items()}
            if details is not None
            else None
        )
        envelope = Envelope.failure(
            error=EnvelopeError(code=code, message=message, details=normalised_details),
            meta=meta,
        )
        self._schemas.validator("envelope.schema.json").validate(
            envelope.model_dump(by_alias=True)
        )
        error_payload: dict[str, Any] = {"code": code, "message": message}
        if normalised_details is not None:
            error_payload["details"] = normalised_details
        self._log_manager.emit(
            ServerLogEvent(
                ts=datetime.now(UTC),
                request_id=meta.request_id,
                trace_id=meta.trace_id,
                span_id=meta.span_id,
                transport=context.transport,
                route=context.route,
                method=context.method,
                status="error",
                duration_ms=duration_ms,
                attempt=context.attempt,
                input_bytes=ids["input_bytes"],
                output_bytes=0,
                error=error_payload,
                metadata={
                    "toolId": tool_id,
                    "promptId": prompt_id,
                    "schemaVersion": self._schema_version,
                    "deterministic": context.deterministic_ids,
                },
                step_id=step_id,
            )
        )
        return envelope

    def _normalise_context(
        self,
        context: RequestContext | None,
        route: str,
        method: str,
        payload: Mapping[str, Any],
    ) -> RequestContext:
        if context is None:
            context = RequestContext(
                transport="http",
                route=route,
                method=method,
                deterministic_ids=False,
            )
        return context.clone_with_payload(payload)

    def _tool_to_dict(self, toolpack: Toolpack) -> dict[str, Any]:
        return {
            "id": toolpack.id,
            "version": toolpack.version,
            "deterministic": toolpack.deterministic,
            "timeoutMs": toolpack.timeout_ms,
            "limits": dict(toolpack.limits),
            "caps": dict(toolpack.caps),
        }

    def _request_ids(self, context: RequestContext) -> dict[str, Any]:
        payload = context.request_payload or {}
        fingerprint = _fingerprint({
            "route": context.route,
            "method": context.method,
            "payload": payload,
        })
        if context.deterministic_ids:
            request_id = uuid5(_DETERMINISTIC_NAMESPACE, f"req:{fingerprint}")
            trace_id = uuid5(_DETERMINISTIC_NAMESPACE, f"trace:{fingerprint}")
            span_id = uuid5(_DETERMINISTIC_NAMESPACE, f"span:{fingerprint}")
        else:
            request_id = uuid4()
            trace_id = uuid4()
            span_id = uuid4()
        return {
            "request_id": str(request_id),
            "trace_id": str(trace_id),
            "span_id": str(span_id),
            "input_bytes": _payload_size(payload),
        }


def _fingerprint(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _payload_size(payload: Mapping[str, Any] | Any) -> int:
    if isinstance(payload, Mapping):
        return len(_fingerprint(payload).encode("utf-8"))
    return len(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _prompt_from_payload(data: Mapping[str, Any], path: Path) -> Prompt:
    try:
        identifier = str(data["id"])
        major = int(data.get("major", 1))
        name = str(data.get("name", identifier))
        description = str(data.get("description", ""))
        tags = [str(tag) for tag in data.get("tags", [])]
        messages_field = data.get("messages", [])
        if not isinstance(messages_field, Iterable):
            raise TypeError("messages must be an iterable")
        messages: list[dict[str, str]] = []
        for entry in messages_field:
            if not isinstance(entry, Mapping):
                raise TypeError("prompt message must be a mapping")
            role = str(entry.get("role"))
            content = str(entry.get("content", ""))
            messages.append({"role": role, "content": content})
    except Exception as exc:  # pragma: no cover - defensive parsing
        raise ValueError(f"Invalid prompt file {path}: {exc}") from exc
    prompt_id = f"{identifier}@{major}"
    return Prompt(
        prompt_id=prompt_id,
        name=name,
        description=description,
        major=major,
        tags=tags,
        messages=messages,
    )


def _duration_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _json_schema_validator_for(schema: Mapping[str, Any]) -> Draft202012Validator:
    validator_cls = validators.validator_for(schema)
    validator_cls.check_schema(schema)
    return validator_cls(schema)


def _normalise_detail_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalise_detail_value(val) for key, val in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_normalise_detail_value(item) for item in value]
    try:
        json.dumps(value)
    except TypeError:
        return repr(value)
    return value


def _schema_error_details(
    *,
    tool_id: str | None,
    stage: str,
    error: ValidationError,
) -> dict[str, Any]:
    return {
        "toolId": tool_id,
        "stage": stage,
        "message": error.message,
        "instancePath": [str(part) for part in error.path],
        "schemaPath": [str(part) for part in error.schema_path],
        "validator": error.validator,
        "validatorValue": _normalise_detail_value(error.validator_value),
    }
