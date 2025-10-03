# Core Tools Minimal Subset

This document summarises the deterministic stub implementation for the MCP core
tools used during the 06ab task. The subset covers three Python toolpacks:

- `mcp.tool:exports.render.markdown`
- `mcp.tool:vector.query.search`
- `mcp.tool:docs.load.fetch`

Each toolpack loads via the declarative runtime, validates input and output using
Draft 2020-12 JSON Schemas, and emits structured JSONL logs at
`runs/core_tools/minimal.jsonl`. Logs capture timestamps, identifiers, payload
sizes, and deterministic metadata so that DeepDiff comparisons can enforce
regressions against the golden fixture
(`tests/fixtures/mcp/core_tools/minimal_golden.jsonl`).

The runtime intentionally keeps the behaviour deterministic: the vector search
returns scripted hits and the markdown renderer always emits the same content
hash for the same payload. This determinism allows the log diff utility
(`scripts/diff_core_tool_logs.py`) to provide high-signal regression detection.
