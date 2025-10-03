from __future__ import annotations

import json
from pathlib import Path

import pytest

from apps.mcp_server.logging import JsonLogWriter
from apps.mcp_server.runtime.core_tools import CoreToolsRuntime
from apps.toolpacks.executor import Executor, ToolpackExecutionError
from apps.toolpacks.loader import ToolpackLoader

TOOLPACK_DIR = Path("apps/mcp_server/toolpacks/core")
DOC_FIXTURE = Path("tests/fixtures/mcp/core_tools/docs/example.md")
METADATA_FIXTURE = Path("tests/fixtures/mcp/core_tools/docs/example_metadata.json")


@pytest.fixture()
def runtime(tmp_path: Path) -> tuple[CoreToolsRuntime, JsonLogWriter]:
    loader = ToolpackLoader()
    loader.load_dir(TOOLPACK_DIR)
    log_path = tmp_path / "core-tools.jsonl"
    logger = JsonLogWriter(
        log_path,
        agent_id="mcp_server",
        task_id="06ab_core_tools_minimal_subset",
    )
    runtime = CoreToolsRuntime(toolpacks=loader.list(), executor=Executor(), log_writer=logger)
    return runtime, logger


def _read_logs(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_exports_render_markdown_generates_expected_markdown(
    runtime: tuple[CoreToolsRuntime, JsonLogWriter],
) -> None:
    runtime_instance, logger = runtime
    payload = {
        "title": "Core Tools Demo",
        "template": "# {{ title }}\n\n{{ body }}",
        "body": "Hello from deterministic stub.",
        "front_matter": {
            "authors": ["RAGX"],
            "tags": ["demo"],
        },
    }

    result = runtime_instance.invoke("mcp.tool:exports.render.markdown", payload)

    assert "markdown" in result
    assert result["markdown"].startswith("---\n")
    assert "Core Tools Demo" in result["markdown"]
    assert result["metadata"]["content_type"] == "text/markdown"

    logs = _read_logs(logger.path)
    assert any(entry["event"] == "invoke.success" for entry in logs)


def test_docs_load_fetch_reads_fixture(runtime: tuple[CoreToolsRuntime, JsonLogWriter]) -> None:
    runtime_instance, _ = runtime
    payload = {
        "path": str(DOC_FIXTURE),
        "metadata_path": str(METADATA_FIXTURE),
    }

    result = runtime_instance.invoke("mcp.tool:docs.load.fetch", payload)

    assert result["document"]["path"].endswith("example.md")
    assert result["metadata"]["title"] == "Example Document"
    assert result["metadata"]["authors"] == ["Test Author"]


def test_vector_query_search_returns_deterministic_hits(
    runtime: tuple[CoreToolsRuntime, JsonLogWriter],
) -> None:
    runtime_instance, logger = runtime

    result = runtime_instance.invoke(
        "mcp.tool:vector.query.search",
        {"query": "retrieval", "top_k": 2},
    )

    assert result["hits"]
    assert len(result["hits"]) == 2
    assert all(hit["score"] <= 1.0 for hit in result["hits"])
    logs = _read_logs(logger.path)
    assert sum(entry["event"] == "invoke.success" for entry in logs) >= 1


def test_failed_invocation_logs_error(runtime: tuple[CoreToolsRuntime, JsonLogWriter]) -> None:
    runtime_instance, logger = runtime

    with pytest.raises(ToolpackExecutionError):
        runtime_instance.invoke("mcp.tool:exports.render.markdown", {"body": "missing fields"})

    error_events = [entry for entry in _read_logs(logger.path) if entry["status"] == "error"]
    assert error_events, "Error events must be recorded"
    assert error_events[0]["event"] == "invoke.failure"
    assert error_events[0]["error"]["code"] == "validation_error"
