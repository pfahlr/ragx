"""Integration contract for HTTP/STDIO parity and logging."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from deepdiff import DeepDiff
from fastapi.testclient import TestClient

from apps.mcp_server.http import create_app
from apps.mcp_server.service.mcp_service import McpService
from apps.mcp_server.stdio import JsonRpcStdioServer

INVALID_ENVELOPE_FIXTURE = Path("tests/fixtures/mcp/envelope/invalid_params_type.json")
GOLDEN_LOG = Path("tests/fixtures/mcp/envelope_validation_golden.jsonl")


@pytest.fixture()
def invalid_envelope_payload() -> dict[str, object]:
    return json.loads(INVALID_ENVELOPE_FIXTURE.read_text(encoding="utf-8"))


@pytest.mark.xfail(strict=True, reason="Transport parity validation not implemented yet")
def test_http_and_stdio_invalid_envelope_parity(
    tmp_path: Path,
    invalid_envelope_payload: dict[str, object],
) -> None:
    log_dir = tmp_path / "runs"
    service = McpService.create(
        toolpacks_dir=Path("apps/mcp_server/toolpacks"),
        prompts_dir=Path("apps/mcp_server/prompts"),
        schema_dir=Path("apps/mcp_server/schemas/mcp"),
        log_dir=log_dir,
        deterministic_logs=True,
    )

    app = create_app(service)
    client = TestClient(app)
    response = client.post(
        "/mcp/tool/mcp.tool:docs.load.fetch",
        json=invalid_envelope_payload,
    )
    assert response.status_code == 400
    http_payload = response.json()

    stdio_server = JsonRpcStdioServer(service, deterministic_ids=True)
    stdio_payload = asyncio.run(stdio_server.handle_request(invalid_envelope_payload))
    assert stdio_payload == http_payload

    latest_symlink = log_dir / "mcp_server" / "envelope_validation.latest.jsonl"
    produced_path = latest_symlink.resolve(strict=True)
    expected_records = [
        json.loads(line)
        for line in GOLDEN_LOG.read_text(encoding="utf-8").splitlines()
    ]
    produced_records = [
        json.loads(line)
        for line in produced_path.read_text(encoding="utf-8").splitlines()
    ]

    diff = DeepDiff(
        expected_records,
        produced_records,
        ignore_order=True,
        exclude_regex_paths=[
            r"root\[\d+\]\['ts'\]",
            r"root\[\d+\]\['traceId'\]",
            r"root\[\d+\]\['spanId'\]",
            r"root\[\d+\]\['durationMs'\]",
            r"root\[\d+\]\['metadata'\]\['requestId'\]",
        ]
    )
    assert diff == {}
