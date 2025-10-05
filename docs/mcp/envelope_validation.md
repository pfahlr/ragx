# MCP Envelope Validation – Implementation Notes

## Overview

This document captures the production-ready implementation of the MCP
envelope and tool IO validation stack introduced in task
`06cV2B_mcp_envelope_and_schema_validation_B`. It supersedes the draft
contract documented in `envelope_validation_spec.md` and should be read
alongside the executable unit, integration, and property-based tests.

## Schema Registry

* Module: `apps.mcp_server.validation.schema_registry`
* Responsibilities:
  * Load Draft 2020-12 JSON Schemas from `codex/specs/schemas/`.
  * Compute SHA256 fingerprints and cache compiled validators keyed by
    file path + fingerprint. This ensures hot reload safety while
    avoiding redundant compilation work.
  * Expose `load_envelope()` and `load_tool_io(tool_id)` helpers that
    return jsonschema validators satisfying the `ValidatorProtocol`
    interface.
* Tool IO validators are shared across tools because the canonical
  schema is transport-agnostic. Per-tool specialisation lives in the
  toolpack schemas and is enforced downstream.

## Canonical Errors

* Module: `apps.mcp_server.service.errors`
* Canonical codes: `INVALID_INPUT`, `INVALID_OUTPUT`, `NOT_FOUND`,
  `UNAUTHORIZED`, `INTERNAL_ERROR`.
* HTTP status mapping:

  | Code            | HTTP Status |
  |-----------------|-------------|
  | INVALID_INPUT   | 400         |
  | INVALID_OUTPUT  | 502         |
  | NOT_FOUND       | 404         |
  | UNAUTHORIZED    | 401         |
  | INTERNAL_ERROR  | 500         |

* JSON-RPC errors expose `{code, message, data}` where `data` includes
  the canonical code and mapped HTTP status. This surface is used by the
  HTTP and STDIO transports to present consistent error metadata.

## Structured Logging

* Log prefix: `runs/mcp_server/envelope_validation/`
* Latest symlink: `runs/mcp_server/envelope_validation.latest.jsonl`
* Volatile fields: `ts`, `traceId`, `spanId`, `durationMs`, `requestId`
* Validation diff script: `scripts/diff_envelope_validation_logs.py`
  compares produced logs against
  `tests/fixtures/mcp/envelope_validation_golden.jsonl` while ignoring
  the whitelisted volatile fields.

## Runbooks

### Validator Cache Refresh

1. Remove the relevant cache entry by restarting the MCP server (cache
   is in-memory and keyed by fingerprint).
2. Verify schema fingerprints by running
   `python -m apps.mcp_server.validation.schema_registry` (future CLIs
   may automate this step).
3. Execute `pytest tests/unit/test_envelope_schema_validation.py` to
   confirm the new schema compiles cleanly.

### Golden Drift Investigation

1. Run `python scripts/diff_envelope_validation_logs.py`.
2. If DeepDiff reports changes, inspect whether new metadata fields were
   introduced. Update the whitelist when adding sanctioned fields.
3. Regenerate the golden file once behaviour is confirmed as intended
   and append a changelog entry describing the drift.

### Shadow → Enforce Rollout

1. Enable shadow validation in the transport (log-only mode).
2. Monitor `envelope_validation_*` metrics for spikes (success/failure
   counters + latency histogram).
3. Flip the enforcement flag once failure rate remains within SLO for a
   full release cycle.
