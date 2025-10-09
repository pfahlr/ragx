from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from apps.mcp_server.http import create_app
from apps.mcp_server.service.mcp_service import McpService

SCHEMA_DIR = Path("apps/mcp_server/schemas/mcp")
TOOLPACKS_DIR = Path("apps/mcp_server/toolpacks")
PROMPTS_DIR = Path("apps/mcp_server/prompts")
DOC_FIXTURE = Path("tests/fixtures/mcp/docs/sample_article.md")


@pytest.fixture(autouse=True)
def _seed_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGX_SEED", "1234")


@pytest.fixture
def service(tmp_path: Path) -> McpService:
    return McpService.create(
        toolpacks_dir=TOOLPACKS_DIR,
        prompts_dir=PROMPTS_DIR,
        schema_dir=SCHEMA_DIR,
        log_dir=tmp_path / "runs",
        schema_version="0.1.0",
        deterministic_logs=True,
    )


def _load_log_lines(service: McpService) -> list[dict[str, object]]:
    log_path = service.log_manager.writer.path
    with log_path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def test_python_toolpack_invocation_and_idempotency_logging(service: McpService) -> None:
    app = create_app(service, deterministic_ids=True)
    client = TestClient(app)

    payload = {"arguments": {"path": str(DOC_FIXTURE)}}

    first = client.post("/mcp/tool/mcp.tool:docs.load.fetch", json=payload)
    assert first.status_code == 200, first.text
    second = client.post("/mcp/tool/mcp.tool:docs.load.fetch", json=payload)
    assert second.status_code == 200, second.text

    first_body = first.json()
    second_body = second.json()

    first_result = first_body["data"]["result"]
    second_result = second_body["data"]["result"]

    assert first_result["document"]["path"].endswith("sample_article.md")
    assert first_result == second_result, (
        "Toolpack execution should be deterministic for identical payloads"
    )

    logs = _load_log_lines(service)
    tool_events = [
        entry
        for entry in logs
        if entry.get("method") == "mcp.tool.invoke"
    ]
    assert len(tool_events) == 2, (
        "expected 2 tool events, got: "
        f"{[entry.get('metadata') for entry in tool_events]}"
    )

    cache_hit_first = tool_events[0]["metadata"].get("idempotencyCacheHit")
    cache_hit_second = tool_events[1]["metadata"].get("idempotencyCacheHit")

    assert cache_hit_first is False or cache_hit_first is None, (
        f"first call should not be cache hit: {cache_hit_first!r}"
    )
    assert cache_hit_second is True, (
        f"second call should be recorded as cache hit: {cache_hit_second!r}"
    )
