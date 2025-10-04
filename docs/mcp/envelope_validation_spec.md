# MCP Envelope Validation Spec (Draft)

This document captures the executable contract for validating MCP envelopes and
per-tool schemas as introduced in task `06cV2A_mcp_envelope_and_schema_validation_A`.
It establishes the expectations that the automated tests enforce.

## Envelope Schema

* Schema location: `codex/specs/schemas/envelope.schema.json`.
* Draft: 2020-12.
* Required request fields: `id`, `jsonrpc`, `method`, `params`.
* `jsonrpc` must equal `"2.0"`.
* `params` is a JSON object containing at least:
  * `tool`: canonical MCP tool identifier (`mcp.tool:<domain>` form).
  * `input`: JSON object forwarded to tool validators.
* Any deviation (missing `method`, non-object `params`, etc.) triggers
  `INVALID_INPUT` once validation is enforced.

## Tool Input / Output Schema Contract

* Schema bundle descriptor: `codex/specs/schemas/tool_io.schema.json`.
* `SchemaRegistry.load_tool_io(tool_id)` must return compiled Draft 2020-12
  validators for the tool's input **and** output schemas.
* Validators are cached by schema fingerprint to guarantee deterministic
  behavior across transports.
* Input validators reject payloads missing mandatory keys (e.g. `query`).
* Output validators guard shape and metadata (e.g. `hits[]` structure).

## Canonical Errors

The error enum is intentionally small for clarity:

| Code             | HTTP Status | JSON-RPC Code | Message                 |
| ---------------- | ----------- | ------------- | ----------------------- |
| `INVALID_INPUT`  | 400         | -32602        | `"Invalid params"`      |
| `INVALID_OUTPUT` | 502         | -32002        | `"Invalid tool output"` |
| `NOT_FOUND`      | 404         | -32004        | `"Resource not found"`  |
| `UNAUTHORIZED`   | 401         | -32001        | `"Unauthorized"`        |
| `INTERNAL_ERROR` | 500         | -32603        | `"Internal error"`      |

`CanonicalError.to_jsonrpc_error(code)` wraps the mapping and includes a
`{"canonical_code": code}` data payload for downstream telemetry.

## Transport Parity & Logging

* Golden fixture: `tests/fixtures/mcp/envelope_validation_golden.jsonl`.
* After stripping volatile identifiers (`ts`, `traceId`, `spanId`,
  `requestId`, `durationMs`) and transport-local routing fields (`transport`,
  `route`), HTTP and STDIO envelope log entries must be identical.
* Structured logs are stored under `runs/mcp_server/envelope_validation/` with a
  `*.latest.jsonl` symlink for the most recent attempt.
* Golden logs ignore volatile fields using `DeepDiff` whitelist rules defined in
  the task file.

## Deterministic IDs

* `--deterministic-ids` flag (already available in CLI) will seed UUID5
  generation. Tests assert deterministic behavior once validators are active.

This spec is executable via the tests added in this task. Implementations must
satisfy the tests **without** mutating the contracts documented above.
