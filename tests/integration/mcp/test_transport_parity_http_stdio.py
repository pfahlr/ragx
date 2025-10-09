from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator

from apps.mcp_server.http.main import create_app
from apps.mcp_server.service.mcp_service import McpService
from apps.mcp_server.stdio.server import JsonRpcStdioServer

SCHEMA_DIR = Path("apps/mcp_server/schemas/mcp")
PROMPTS_DIR = Path("apps/mcp_server/prompts")
TOOLPACKS_DIR = Path("tests/stubs/toolpacks")
TOOL_ID = "tests.toolpacks.deterministicsum"
GOLDEN_LOG_PATH = Path("tests/fixtures/mcp/logs/mcp_toolpacks_transport_golden.jsonl")


@pytest.fixture(scope="module")
def tool_response_validator() -> Draft202012Validator:
    schema_path = SCHEMA_DIR / "tool.response.schema.json"
    with schema_path.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    validator = Draft202012Validator(schema)
    validator.check_schema(schema)
    return validator


@pytest.fixture
def service_factory(tmp_path: Path) -> Callable[..., McpService]:
    def _build(**overrides: Any) -> McpService:
        return McpService.create(
            toolpacks_dir=TOOLPACKS_DIR,
            prompts_dir=PROMPTS_DIR,
            schema_dir=SCHEMA_DIR,
            log_dir=tmp_path / "runs",
            schema_version="0.1.0",
            deterministic_logs=True,
            deterministic_ids=overrides.pop("deterministic_ids", True),
            max_input_bytes=overrides.pop("max_input_bytes", 2048),
            max_output_bytes=overrides.pop("max_output_bytes", 4096),
            timeout_ms=overrides.pop("timeout_ms", 2000),
            **overrides,
        )

    return _build


def _invoke_http(client: TestClient, payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
    response = client.post(f"/mcp/tool/{TOOL_ID}", json={"arguments": payload})
    return response.json(), response.status_code


async def _invoke_stdio(server: JsonRpcStdioServer, payload: dict[str, Any]) -> dict[str, Any]:
    response = await server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": "tool-1",
            "method": "mcp.tool.invoke",
            "params": {"toolId": TOOL_ID, "arguments": payload},
        }
    )
    assert "result" in response or "error" in response
    return response.get("result") or response["error"]["data"]["envelope"]


def test_http_stdio_parity_and_schema_validation(
    service_factory: Callable[..., McpService], tool_response_validator: Draft202012Validator
) -> None:
    service = service_factory()
    client = TestClient(create_app(service, deterministic_ids=True))
    stdio_server = JsonRpcStdioServer(service, deterministic_ids=True)

    payload = {"numbers": [1, 2, 3]}
    http_envelope, http_status = _invoke_http(client, payload)
    stdio_envelope = asyncio.run(_invoke_stdio(stdio_server, payload))

    assert http_status == 200
    tool_response_validator.validate(http_envelope)
    tool_response_validator.validate(stdio_envelope)

    assert http_envelope["data"] == stdio_envelope["data"]
    assert http_envelope["meta"]["status"] == "ok"
    assert stdio_envelope["meta"]["status"] == "ok"

    http_execution = http_envelope["meta"]["execution"]
    stdio_execution = stdio_envelope["meta"]["execution"]
    assert http_execution["inputBytes"] == stdio_execution["inputBytes"]
    assert http_execution["outputBytes"] == stdio_execution["outputBytes"]
    assert isinstance(http_execution["durationMs"], int | float)
    assert isinstance(stdio_execution["durationMs"], int | float)

    http_idempotency = http_envelope["meta"].get("idempotency", {})
    stdio_idempotency = stdio_envelope["meta"].get("idempotency", {})
    assert isinstance(http_idempotency.get("cacheHit"), bool)
    assert isinstance(stdio_idempotency.get("cacheHit"), bool)

    assert http_envelope["meta"]["deterministic"] is True
    assert stdio_envelope["meta"]["deterministic"] is True

    assert http_envelope["meta"]["transport"] == "http"
    assert stdio_envelope["meta"]["transport"] == "stdio"


def test_service_enforces_max_input_bytes(
    service_factory: Callable[..., McpService], tool_response_validator: Draft202012Validator
) -> None:
    service = service_factory(max_input_bytes=32)
    client = TestClient(create_app(service, deterministic_ids=True))
    stdio_server = JsonRpcStdioServer(service, deterministic_ids=True)

    large_payload = {"numbers": [1, 2, 3], "output_bytes": 0, "padding": "x" * 64}

    http_envelope, http_status = _invoke_http(client, large_payload)
    stdio_envelope = asyncio.run(_invoke_stdio(stdio_server, large_payload))

    assert http_status == 400
    assert http_envelope["ok"] is False
    assert stdio_envelope["ok"] is False
    assert http_envelope["error"]["code"] == "INVALID_INPUT"
    assert stdio_envelope["error"]["code"] == "INVALID_INPUT"

    tool_response_validator.validate(http_envelope)
    tool_response_validator.validate(stdio_envelope)


def test_service_enforces_max_output_bytes(
    service_factory: Callable[..., McpService], tool_response_validator: Draft202012Validator
) -> None:
    service = service_factory(max_output_bytes=64)
    client = TestClient(create_app(service, deterministic_ids=True))
    stdio_server = JsonRpcStdioServer(service, deterministic_ids=True)

    payload = {"numbers": [1, 2, 3], "output_bytes": 128}

    http_envelope, http_status = _invoke_http(client, payload)
    stdio_envelope = asyncio.run(_invoke_stdio(stdio_server, payload))

    assert http_status == 502
    assert http_envelope["ok"] is False
    assert stdio_envelope["ok"] is False
    assert http_envelope["error"]["code"] == "INVALID_OUTPUT"
    assert stdio_envelope["error"]["code"] == "INVALID_OUTPUT"

    tool_response_validator.validate(http_envelope)
    tool_response_validator.validate(stdio_envelope)


def test_service_enforces_timeout(
    service_factory: Callable[..., McpService], tool_response_validator: Draft202012Validator
) -> None:
    service = service_factory(timeout_ms=10)
    client = TestClient(create_app(service, deterministic_ids=True))
    stdio_server = JsonRpcStdioServer(service, deterministic_ids=True)

    payload = {"numbers": [1, 2], "sleep_ms": 50}

    http_envelope, http_status = _invoke_http(client, payload)
    stdio_envelope = asyncio.run(_invoke_stdio(stdio_server, payload))

    assert http_status == 504
    assert http_envelope["ok"] is False
    assert stdio_envelope["ok"] is False
    assert http_envelope["error"]["code"] == "TIMEOUT"
    assert stdio_envelope["error"]["code"] == "TIMEOUT"

    tool_response_validator.validate(http_envelope)
    tool_response_validator.validate(stdio_envelope)


def _normalise_log_record(record: dict[str, Any]) -> dict[str, Any]:
    clone = json.loads(json.dumps(record))
    for field in ["ts", "traceId", "spanId", "requestId"]:
        clone.pop(field, None)
    execution = clone.setdefault("execution", {})
    execution["durationMs"] = 0.0
    metadata = clone.setdefault("metadata", {})
    for volatile in ["runId", "attemptId", "logPath"]:
        metadata.pop(volatile, None)
    idempotency = clone.setdefault("idempotency", {})
    idempotency.setdefault("cacheHit", False)
    if "cacheKey" in idempotency:
        idempotency["cacheKey"] = "<cache-key>"
    return clone


def test_transport_logs_match_golden(service_factory: Callable[..., McpService]) -> None:
    service = service_factory()
    client = TestClient(create_app(service, deterministic_ids=True))

    payload = {"numbers": [1, 2, 3]}
    _invoke_http(client, payload)
    _invoke_http(client, payload)

    log_path = service.log_manager.writer.path
    actual_records = [
        _normalise_log_record(json.loads(line))
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    expected_records = [
        json.loads(line)
        for line in GOLDEN_LOG_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert actual_records == expected_records
