# MCP Envelope Validation Spec (Executable Draft)

## Overview

This document captures the executable contract for MCP envelope
validation introduced in task `06cV2A`. The accompanying unit,
integration, and property-based tests exercise the rules defined here.
Implementation work is deferred to a follow-up task.

## Envelope Schema

* Schema path: `codex/specs/schemas/envelope.schema.json`
* Draft: 2020-12
* Required fields: `id`, `jsonrpc`, `method`, `params`
* Optional fields: `meta`, `idempotencyKey`
* `jsonrpc` is constrained to the literal `"2.0"`
* Additional properties are currently disallowed to ensure a tight
  contract. Future extensions should update the schema and tests.

## Tool IO Schema

* Schema path: `codex/specs/schemas/tool_io.schema.json`
* Draft: 2020-12
* Required fields: `tool`, `input`
* Optional fields: `version`, `metadata`
* Tool identifiers must be non-empty strings using the canonical
  `mcp.tool:*` format.

## Schema Registry Stub

* Module: `apps.mcp_server.validation.schema_registry`
* Exposes `SchemaRegistry` with `load_envelope()` and
  `load_tool_io(tool_id)` returning compiled jsonschema validators.
* Implementation caches validators keyed by schema fingerprint (SHA256)
  to avoid redundant compilation work.

## Canonical Errors

* Module: `apps.mcp_server.service.errors`
* Enumerates canonical codes aligning with the master spec.
* HTTP status and JSON-RPC mapping helpers return deterministic
  translations for each canonical error.
* Tests exercise the mapping contract to guard against regressions.

## Structured Logging Golden

* Fixture: `tests/fixtures/mcp/envelope_validation_golden.jsonl`
* Provides reference log entries for envelope validation failures across
  HTTP and STDIO transports.
* Integration tests assert required fields, metadata payload, and parity
  of canonical error surfaces.

## Next Steps

* Implement jsonschema-backed validators and caching in the schema
  registry.
* Populate canonical error mappings for HTTP and JSON-RPC.
* Wire the MCP service transports to use the new registry and emit
  parity-checked error envelopes.
