# Toolpack Loader

The Toolpack loader discovers declarative tool definitions (`*.tool.yaml`),
materialises them into strongly-typed `Toolpack` objects, and validates their
schemas up front. Implementations follow the contracts in
[`codex/specs/ragx_master_spec.yaml`](../codex/specs/ragx_master_spec.yaml):
camelCase field names, spec-listed execution kinds, and deterministic
configuration snapshots.

## Usage

```python
from pathlib import Path

from apps.toolpacks.loader import ToolpackLoader, ToolpackValidationError

loader = ToolpackLoader()
try:
    loader.load_dir(Path("apps/mcp_server/toolpacks"))
except ToolpackValidationError as exc:
    raise SystemExit(f"invalid toolpack: {exc}")

for pack in loader.list():
    print(pack.id, pack.execution["kind"], pack.timeout_ms)
```

* `load_dir(path)` walks the directory recursively, resolves `$ref` entries in
  `inputSchema` / `outputSchema`, and validates each schema via
  `jsonschema.validator_for(...).check_schema`.
* `list()` returns the loaded toolpacks sorted by `id` to keep downstream cache
  keys deterministic.
* `get(tool_id)` returns the matching toolpack or raises `KeyError` if not
  present.

## Validation Rules

The loader enforces the spec-defined invariants:

- Required fields: `id`, `version`, `deterministic`, `timeoutMs`, `limits`,
  `inputSchema`, `outputSchema`, `execution`.
- `id` must be dotted lowercase segments (e.g. `pkg.tool`).
- `version` must follow semantic version rules (validated via `packaging.Version`).
- `timeoutMs` must be a positive integer.
- `limits.maxInputBytes` and `limits.maxOutputBytes` must exist and be positive
  integers.
- `execution.kind` must be one of `python`, `node`, `php`, `cli`, `http`.
- Execution payloads must satisfy per-kind contracts: python requires
  `module:callable` (or a non-empty `script`), CLI `cmd` must be a list of
  strings, HTTP requires a `url` plus optional string headers/method, Node needs
  a `script`/`node`/`module` entry, and PHP requires either `php` or `script`.
- Duplicate tool identifiers are rejected (the loader guarantees a single
  toolpack per id).
- Optional blocks (`caps`, `env`, `templating`) must follow spec contracts:
  `caps.network` accepts `http`/`https`, `caps.filesystem` allows `read`/`write`
  path lists, `caps.subprocess` is boolean; `env.passthrough`/`env.set` entries
  must use uppercase variable names; `templating.engine` currently supports
  `jinja2`, and `templating.context` must be JSON-serialisable.
- `$ref` schema targets must exist and contain valid JSON Schema documents.
- Nested `$ref` chains are resolved relative to the referencing file before
  validation.

## Failure semantics

Any violation of the above contracts raises `ToolpackValidationError` with a
contextual message (malformed YAML, missing fields, schema issues, duplicate
ids, unsupported execution kinds, etc.). Consumers should treat the loader as a
strict gatekeeper: if `load_dir` succeeds the resulting `Toolpack` instances are
guaranteed to match the spec, and downstream systems can rely on their
structure without revalidating.

## Testing

Regression coverage lives in `tests/unit/test_toolpacks_loader.py`, exercising:

- Happy-path loading of spec-compliant toolpacks with `$ref` schemas.
- Rejection of snake_case metadata (missing camelCase spec fields).
- Detection of duplicate tool ids and missing required fields.
- Enforcement of id patterns and semantic version strings.
- Validation of `caps`/`env`/`templating` semantics, including rejection of
  unsupported protocols, env names, or templating engines.
- Execution kind whitelisting.
- JSON Schema structural validation failures (including property-based checks
  for invalid `type` values when Hypothesis is available).

The suite runs as part of `./scripts/ensure_green.sh` and must stay green before
shipping changes to the loader.
