from __future__ import annotations

import hashlib
from pathlib import Path

from apps.mcp_server.logging import JsonLogWriter
from apps.mcp_server.runtime.core_tools import CoreToolsRuntime
from apps.toolpacks.executor import Executor
from apps.toolpacks.loader import ToolpackLoader

TOOLPACK_DIR = Path("apps/mcp_server/toolpacks/core")
DOC_FIXTURE_DIR = Path("tests/fixtures/mcp/docs")


def test_end_to_end_invocations(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RAGX_SEED", "9")
    loader = ToolpackLoader()
    loader.load_dir(TOOLPACK_DIR)

    storage_prefix = tmp_path / "runs/core_tools/minimal"
    latest = tmp_path / "runs/core_tools/minimal.jsonl"
    writer = JsonLogWriter(
        agent_id="mcp-e2e",
        task_id="core-tools",
        storage_prefix=storage_prefix,
        latest_symlink=latest,
        schema_version="0.1.0",
        deterministic=True,
        root_dir=tmp_path,
    )
    runtime = CoreToolsRuntime(
        toolpacks=loader.list(),
        executor=Executor(),
        log_writer=writer,
        agent_id="mcp-e2e",
        task_id="core-tools",
    )

    doc = runtime.invoke(
        "mcp.tool:docs.load.fetch",
        {
            "path": str(DOC_FIXTURE_DIR / "sample_article.md"),
            "metadataPath": str(DOC_FIXTURE_DIR / "sample_metadata.json"),
        },
    )

    hits = runtime.invoke(
        "mcp.tool:vector.query.search",
        {"query": doc["metadata"]["title"], "topK": 1},
    )
    render = runtime.invoke(
        "mcp.tool:exports.render.markdown",
        {
            "title": doc["metadata"]["title"],
            "template": "# {{ title }}\n\n{{ body }}",
            "body": doc["document"]["content"],
            "frontMatter": doc["metadata"],
        },
    )

    assert hits["hits"], "Vector search should return at least one hit"
    assert render["contentHash"] == hashlib.sha256(render["markdown"].encode("utf-8")).hexdigest()
    assert latest.is_symlink() and latest.resolve() == writer.path
