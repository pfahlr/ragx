from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("pydantic")

from fastapi.testclient import TestClient

from apps.mcp_server.http.main import create_app
from apps.mcp_server.service.mcp_service import McpService

SCHEMA_DIR = Path("apps/mcp_server/schemas/mcp")
TOOLPACKS_DIR = Path("apps/mcp_server/toolpacks")
PROMPTS_DIR = Path("apps/mcp_server/prompts")


def _read_log_records(path: Path) -> list[dict[str, object]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_python_toolpack_executes_via_http_with_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAGX_SEED", "7")

    service = McpService.create(
        toolpacks_dir=TOOLPACKS_DIR,
        prompts_dir=PROMPTS_DIR,
        schema_dir=SCHEMA_DIR,
        log_dir=tmp_path / "runs",
        schema_version="0.1.0",
        deterministic_logs=True,
    )

    app = create_app(service, deterministic_ids=True)
    client = TestClient(app)

    fixture_path = Path("tests/fixtures/mcp/docs/sample_article.md")
    metadata_path = Path("tests/fixtures/mcp/docs/sample_metadata.json")

    response_one = client.post(
        "/mcp/tool/mcp.tool:docs.load.fetch",
        json={"arguments": {"path": str(fixture_path), "metadataPath": str(metadata_path)}},
    )
    assert response_one.status_code == 200
    payload_one = response_one.json()
    assert payload_one["ok"] is True
    result_one = payload_one["data"]["result"]
    assert result_one["document"]["path"].endswith("sample_article.md")

    response_two = client.post(
        "/mcp/tool/mcp.tool:docs.load.fetch",
        json={"arguments": {"path": str(fixture_path), "metadataPath": str(metadata_path)}},
    )
    assert response_two.status_code == 200
    payload_two = response_two.json()
    assert payload_two["ok"] is True
    assert payload_two["data"] == payload_one["data"], (
        "cached invocation should return identical payload"
    )

    log_path = service.log_manager.writer.path
    records = _read_log_records(log_path)
    tool_records = [entry for entry in records if entry.get("method") == "mcp.tool.invoke"]
    assert len(tool_records) == 2, f"expected two tool invocations in logs, got {len(tool_records)}"

    first_meta = tool_records[0]["metadata"]
    second_meta = tool_records[1]["metadata"]
    assert isinstance(first_meta, dict) and isinstance(second_meta, dict)

    execution_one = first_meta.get("execution")
    execution_two = second_meta.get("execution")
    assert execution_one and execution_two, (
        "execution metadata must be present for tool invocations"
    )

    cache_one = first_meta.get("idempotency")
    cache_two = second_meta.get("idempotency")
    assert cache_one and cache_two, "idempotency metadata must be present"
    assert cache_one.get("cacheHit") is False
    assert cache_two.get("cacheHit") is True

    assert execution_one["outputBytes"] == execution_two["outputBytes"]
    assert execution_one["inputBytes"] == execution_two["inputBytes"]
    assert execution_two["durationMs"] <= execution_one["durationMs"] + 1.0

    meta_one = payload_one["meta"]
    meta_two = payload_two["meta"]
    assert meta_one["inputBytes"] == execution_one["inputBytes"]
    assert meta_two["inputBytes"] == execution_two["inputBytes"]
    assert meta_two["outputBytes"] == execution_two["outputBytes"]
    assert meta_two["durationMs"] == execution_two["durationMs"]
