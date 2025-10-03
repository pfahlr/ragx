# Core Tools Minimal Subset

This document describes the deterministic MCP core tools delivered in task
`06ab_core_tools_minimal_subset`.

## Implemented tools

- `mcp.tool:exports.render.markdown` — renders markdown using Jinja2 templates
  and emits YAML front matter.
- `mcp.tool:vector.query.search` — deterministic stub that returns repeatable
  vector hits for integration testing.
- `mcp.tool:docs.load.fetch` — loads local markdown documents and optional
  metadata from JSON sidecars.

## Observability

Tool invocations emit structured JSON lines to `runs/core_tools/minimal.jsonl`.
The payload contains `ts`, `agent_id`, `task_id`, `step_id`, and DeepDiff is
used to guard regressions against a golden fixture in
`tests/fixtures/mcp/core_tools/minimal_golden.jsonl`.

## Diff tooling

`scripts/diff_core_tool_logs.py` compares a captured log with the golden fixture
while ignoring volatile fields such as timestamps, trace IDs, and run IDs.
