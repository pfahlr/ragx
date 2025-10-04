# MCP Envelope Validation Contract (Part A)

This document captures the executable specification introduced in
`06cV2A_mcp_envelope_and_schema_validation_A`. The code remains stubbed, but
the behaviour is locked in via tests and fixtures.

## Scope

* **Schemas** – Draft 2020-12 JSON Schemas for the transport envelope and
  per-tool IO. These live under `codex/specs/schemas/` and are consumed via
  the forthcoming `SchemaRegistry` implementation.
* **Canonical Errors** – Enumeration of `INVALID_INPUT`, `INVALID_OUTPUT`,
  `NOT_FOUND`, `UNAUTHORIZED`, and `INTERNAL_ERROR`. Transport-specific
  mappings will be provided in Part B.
* **Transports** – HTTP and STDIO must reject invalid envelopes with
  identical canonical error payloads. Structured logs are written to
  `runs/mcp_server/envelope_validation*.jsonl` with camelCase fields.

## Test Inventory

| Test | Contract |
| ---- | -------- |
| `tests/unit/test_envelope_schema_validation.py` | Envelope + tool IO schema contracts and caching invariants. |
| `tests/unit/test_canonical_error_mapping.py` | Canonical error to HTTP / JSON-RPC mapping. |
| `tests/property/test_envelope_fuzz.py` | Property-based guardrails around envelope required fields. |
| `tests/integration/test_transport_parity_http_stdio.py` | HTTP vs STDIO parity and structured log diff against golden. |

## Golden Log

The golden fixture `tests/fixtures/mcp/envelope_validation_golden.jsonl`
encodes the required logging fields. Volatile identifiers (`ts`, `traceId`,
`spanId`, `requestId`, `durationMs`) are ignored when diffing.

## Next Steps

* Replace stubs with the real `SchemaRegistry` and `CanonicalError` helpers.
* Wire validation into HTTP/STDIO transports and ensure structured logs are
  emitted using `JsonLogWriter`.
* Add rollout controls (`--once`, `--deterministic-ids`) when enabling the
  validators in production.
