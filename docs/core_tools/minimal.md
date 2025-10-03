# Minimal Core Tools Runtime

The minimal core tools runtime implements three deterministic MCP tools backed by
Python toolpacks:

- `mcp.tool:exports.render.markdown` renders Markdown using a Jinja2 template and
  produces reproducible front matter metadata.
- `mcp.tool:docs.load.fetch` reads Markdown documents from disk, merges optional
  metadata, and emits checksum-verified payloads.
- `mcp.tool:vector.query.search` performs a token overlap search against a
  deterministic in-memory corpus suitable for regression tests.

Each invocation emits structured JSONL logs to `runs/core_tools/minimal.jsonl`
through the `JsonLogWriter`. The writer enforces a `keep-last-5` retention policy
so older runs are rotated to `*.1` … `*.5`. Logs contain:

- deterministic metadata for toolpack id, version, and determinism flag,
- payload sizes and duration measurements, and
- error envelopes when retries are triggered.

Use `scripts/diff_core_tool_logs.py` to compare a produced log against the
`tests/fixtures/mcp/core_tools/minimal_golden.jsonl` fixture. The script ignores
volatile fields such as timestamps and span identifiers via `DeepDiff`.

## Development checklist

1. Update JSON Schemas under `apps/mcp_server/schemas/tools/` before changing
   payload contracts.
2. Add or update tests under `tests/unit/mcp/` and regenerate the golden log
   fixture when behaviour changes.
3. Run `pytest -k "core_tools_minimal"` followed by
   `python scripts/diff_core_tool_logs.py` to confirm parity.
