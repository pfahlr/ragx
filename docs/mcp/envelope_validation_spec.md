# MCP Envelope Validation Spec (Executable Draft)

## Overview

This document captures the executable contract for MCP envelope
validation introduced in task `06cV2A`. The accompanying unit,
integration, and property-based tests exercise the rules defined here.
Implementation work is deferred to a follow-up task.

## Envelope Schema

* Schema path: `apps/mcp_server/schemas/envelope.schema.json`
* Draft: 2020-12
* Required fields: `ok`, `data`, `error`, `meta`
* The `meta` object must include deterministic identifiers,
  transport/method context, duration, attempt count, and IO byte sizes.
* Success envelopes (`ok: true`) must set `error` to `null`; error
  envelopes require a structured `error` payload and `data: null`.
* Additional properties remain disallowed to ensure a tight contract.

## Tool IO Schema

* Schema path: `codex/specs/schemas/tool_io.schema.json`
* Draft: 2020-12
* Required fields: `tool`, `input`
* Optional fields: `version`, `metadata`
* Tool identifiers must be non-empty strings using the canonical
  `mcp.tool:*` format.

## Schema Registry Runtime

* Module: `apps.mcp_server.validation.schema_registry`
* Exposes `SchemaRegistry` with `load_envelope()` and
  `load_tool_io(tool_id)` returning compiled validators.
* Validators are cached by schema fingerprint (SHA256) so repeated calls
  reuse compiled state even across different tools that share the same
  schema definition.
* `ToolIOValidators` enforce canonical tool identifiers and validate
  both shared envelope structure and tool-specific payloads. Output
  validation skips the shared `input` requirement while still applying
  the tool schema.
* Default roots: `codex/specs/schemas` and
  `apps/mcp_server/schemas/tools`.

## Canonical Errors

* Module: `apps.mcp_server.service.errors`
* Enumerates canonical codes aligning with the master spec
  (`INVALID_INPUT`, `INVALID_OUTPUT`, `NOT_FOUND`, `UNAUTHORIZED`,
  `INTERNAL_ERROR`).
* `CanonicalError.to_http_status()` maps canonical codes to deterministic
  HTTP status codes.
* `CanonicalError.to_jsonrpc_error()` produces JSON-RPC error payloads
  containing the canonical code, HTTP status, human-readable message,
  and retryability metadata.
* HTTP and STDIO transports use these helpers to keep error surfaces in
  lock-step.

## Structured Logging Golden

* Fixture: `tests/fixtures/mcp/envelope_validation_golden.jsonl`
* Provides reference log entries for envelope validation in shadow mode
  across the HTTP transport.
* Integration tests assert required fields, metadata payload, and parity
  of canonical error surfaces using
  `scripts/diff_envelope_validation_logs.py`.

## Runbooks

### Refreshing the Validator Cache

* Schema validators are cached in memory. Restart the MCP server after
  modifying schema files to rebuild the cache.

### Investigating Golden Diff Failures

1. Re-run the CLI once mode:
   `python -m apps.mcp_server.cli --once --deterministic-ids --log-dir runs`
2. Compare the latest log to the golden fixture:
   `python scripts/diff_envelope_validation_logs.py --new runs/mcp_server/envelope_validation.latest.jsonl --golden tests/fixtures/mcp/envelope_validation_golden.jsonl`
3. If the diff reflects an intentional change, update the golden and
   include the rationale in the changelog.

