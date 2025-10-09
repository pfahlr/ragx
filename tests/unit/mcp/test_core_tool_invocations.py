# tests/integration/mcp/test_core_tool_invocations.py
from __future__ import annotations

from pathlib import Path

import pytest

from apps.mcp_server.logging import JsonLogWriter
from apps.mcp_server.runtime.core_tools import CoreToolsRuntime
from apps.toolpacks.executor import Executor
from apps.toolpacks.loader import ToolpackLoader

TOOLPACK_DIR = Path("apps/mcp_server/toolpacks/core")
FIXTURE_DIR = Path("tests/fixtures/mcp/docs")


@pytest.fixture
def runtime(tmp_path: Path) -> tuple[CoreToolsRuntime, JsonLogWriter]:
    loader = ToolpackLoader()
    loader.load_dir(TOOLPACK_DIR)
    storage_prefix = tmp_path / "runs/core_tools/minimal"
    latest = tmp_path / "runs/core_tools/minimal.jsonl"
    logger = JsonLogWriter(
        agent_id="mcp_server",
        task_id="06a_core_tools_minimal_subset",
        storage_prefix=storage_prefix,
        latest_symlink=latest,
        schema_version="0.1.0",
        deterministic=True,
        root_dir=tmp_path,
    )
    runtime = CoreToolsRuntime(
        toolpacks=loader.list(),
        executor=Executor(),
        log_writer=logger,
        agent_id="mcp_server",
        task_id="06a_core_tools_minimal_subset",
    )
    return runtime, logger


def test_exports_render_markdown_invocation(
    runtime: tuple[CoreToolsRuntime, JsonLogWriter],
) -> None:
    runtime_instance, logger = runtime
    payload = {
        "title": "Core Tools Demo",
        "template": "# {{ title }}\n\n{{ body }}",
        "body": "Hello from deterministic stub.",
        "frontMatter": {"authors": ["RAGX"], "tags": ["demo"]},
    }

    result = runtime_instance.invoke("mcp.tool:exports.render.markdown", payload)
    assert "markdown" in result
    assert result["markdown"].startswith("---")
    assert "contentHash" in result
    assert logger.path.exists()


def test_docs_load_fetch_reads_fixture(
    runtime: tuple[CoreToolsRuntime, JsonLogWriter],
) -> None:
    runtime_instance, _ = runtime
    payload = {
        "path": str(FIXTURE_DIR / "sample_article.md"),
        "metadataPath": str(FIXTURE_DIR / "sample_metadata.json"),
    }
    result = runtime_instance.invoke("mcp.tool:docs.load.fetch", payload)
    assert "document" in result and "metadata" in result
    assert result["metadata"]["title"] == "Sample Article"


def test_vector_query_search_invocation(
    runtime: tuple[CoreToolsRuntime, JsonLogWriter],
) -> None:
    runtime_instance, logger = runtime
    result = runtime_instance.invoke(
        "mcp.tool:vector.query.search",
        {"query": "retrieval", "topK": 2},
    )
    assert "hits" in result and len(result["hits"]) == 2
    assert logger.path.read_text(encoding="utf-8").strip(), "Logs must not be empty"
