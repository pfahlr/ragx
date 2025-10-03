# Core Tools Minimal Suite

This document describes the deterministic MCP core tools implemented for task 06aV2.

## Available Tools

- **`mcp.tool:exports.render.markdown`** – renders Markdown using a fixed templating engine
  and front-matter normalisation. Outputs include a SHA-256 hash for regression tracking.
- **`mcp.tool:vector.query.search`** – deterministic scoring over fixture documents. Tie-breaking
  is seeded using the `RAGX_SEED` environment variable to guarantee repeatability.
- **`mcp.tool:docs.load.fetch`** – reads Markdown fixtures and optional JSON metadata from disk,
  returning canonical payloads used across integration tests.

## Logging Pipeline

All tools emit structured JSONL logs via `JsonLogWriter`. Each invocation yields `tool.invoke`
followed by `tool.ok` (or `tool.err` upon failure). Logs are written under `runs/core_tools/`
with run-level rotation (keep last five) and a `runs/core_tools/minimal.jsonl` symlink pointing
to the most recent run. Metadata embeds run identifiers, schema versions, and content digests
for DeepDiff regression guards.

## Golden Fixtures and Diffing

The golden log lives at `tests/fixtures/mcp/core_tools/minimal_golden.jsonl`. Regenerate and
compare using:

```bash
python scripts/diff_core_tool_logs.py --new runs/core_tools/minimal.jsonl --golden tests/fixtures/mcp/core_tools/minimal_golden.jsonl
```

Only whitelisted volatile fields (`ts`, `duration_ms`, `run_id`, `trace_id`, `span_id`,
`attempt_id`) are ignored during diffs. All other changes surface as test failures.

## Troubleshooting

- Ensure `RAGX_SEED` is set for reproducible tie-breaking in vector search tests.
- If logs drift, inspect the DeepDiff output for metadata changes before updating the golden.
- Rotation retains only five runs; copy log files elsewhere before running long test suites
  if historical inspection is required.
