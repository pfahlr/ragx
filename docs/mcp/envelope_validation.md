# MCP Envelope Validation

This document describes the production implementation of MCP envelope
validation that supersedes the executable specification delivered in
`06cV2A`.

## Overview

* Validation is backed by the canonical schemas defined in
  `codex/specs/schemas/envelope.schema.json` and
  `codex/specs/schemas/tool_io.schema.json`.
* The `SchemaRegistry` caches compiled Draft 2020-12 validators keyed by
  SHA256 fingerprints to avoid repeated compilation.
* HTTP responses are validated via `EnvelopeValidationMiddleware` before
  leaving the server; STDIO traffic is validated by
  `ValidationFilter`.
* MCP services emit canonical error codes (`INVALID_INPUT`,
  `INVALID_OUTPUT`, `NOT_FOUND`, `UNAUTHORIZED`, `INTERNAL_ERROR`) with
  transport-appropriate mappings and structured metadata.

## Structured Logging

* Logs are persisted under `runs/mcp_server/envelope_validation` with a
  rotating `envelope_validation.latest.jsonl` symlink.
* Each event records request/trace/span identifiers, transport, route,
  duration, attempt counter, canonical error information, and metadata
  (`schemaVersion`, `deterministic`, `toolId`, `promptId`).
* The `scripts/diff_envelope_validation_logs.py` script compares a new
  run against the golden fixture while ignoring volatile identifiers:

```bash
python scripts/diff_envelope_validation_logs.py \
  --baseline tests/fixtures/mcp/envelope_validation_golden.jsonl \
  --new runs/mcp_server/envelope_validation.latest.jsonl
```

## Shadow → Enforce Rollout

* Both HTTP middleware and STDIO filters default to **shadow** mode
  (validate + log). Enforcement can be toggled by instantiating the
  middleware/filter with `mode="enforce"` once SLOs stabilise.
* Runbooks:
  - **Clear validator cache:** restart the MCP server or re-instantiate
    `SchemaRegistry`. The cache is memory only.
  - **Golden diff failures:** run the diff script above to inspect the
    DeepDiff output and compare canonical codes and metadata with the
    fixture.

## Error Mapping

* HTTP status codes are derived via `CanonicalError.to_http_status()`
  (e.g. `INVALID_INPUT → 400`, `INVALID_OUTPUT → 502`).
* JSON-RPC errors are produced via
  `CanonicalError.to_jsonrpc_error()` and include the canonical code,
  mapped HTTP status, and structured `details` payloads.

## Metrics & Tracing Hooks

* Hooks are exposed via the middleware/filter logger for integration
  with observability backends. Metrics collection can wrap the logging
  calls to increment counters (`envelope_validation_success_total`,
  `envelope_validation_failure_total`) and record latencies using the
  request context start/end timestamps.
