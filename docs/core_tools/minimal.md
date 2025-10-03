# Core Tools Minimal Subset

This document describes the deterministic core tool implementations delivered as
part of task `06ab_core_tools_minimal_subset`.

## Included tools

- **`mcp.tool:exports.render.markdown`** — Renders Markdown with YAML front
  matter using a Jinja2 template. The tool returns the rendered Markdown, a
  SHA-256 content hash, and metadata about the render operation.
- **`mcp.tool:vector.query.search`** — Provides a deterministic vector search
  stub over a static in-memory corpus. Scores are derived from a stable hash so
  tests can diff output precisely.
- **`mcp.tool:docs.load.fetch`** — Loads local Markdown documents and optional
  JSON metadata with deterministic hashing and line counts.

## Structured logging

All invocations are logged to `runs/core_tools/minimal.jsonl` using the
`JsonLogWriter`. Each event contains the fields enumerated in the structured
logging contract, including `run_id` and `attempt_id`. The writer buffers log
lines to handle transient filesystem errors and enforces a keep-last-five
retention policy for JSONL artefacts.

## Golden diff workflow

Use the helper script to compare a captured run against the golden fixture:

```bash
python scripts/diff_core_tool_logs.py \
  --actual runs/core_tools/minimal.jsonl \
  --golden tests/fixtures/mcp/logs/core_tools_minimal_golden.jsonl
```

The script removes volatile identifiers and fails if any material differences
are detected.
