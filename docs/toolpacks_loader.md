# Toolpack Loader

The Toolpack loader discovers declarative tool definitions (`*.tool.yaml`),
materialises them into strongly-typed `Toolpack` objects, and validates their
schemas up front. Implementations follow the contracts in
[`codex/specs/ragx_master_spec.yaml`](../codex/specs/ragx_master_spec.yaml):
camelCase field names, validated execution kinds, and deterministic configuration
snapshots.

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

* `load_dir(path)` walks the directory recursively, resolving `$ref` entries in
  `inputSchema` / `outputSchema`, and validating each schema via
  `jsonschema.validator_for(...).check_schema`.
* `list()` returns the loaded toolpacks sorted by `id` to keep downstream cache
  keys deterministic.
* `get(tool_id)` returns the matching toolpack or raises `KeyError` if not
  present.

## Validation Rules

The loader enforces the spec-defined invariants:

- Required fields: `id`, `version`, `deterministic`, `timeoutMs`, `limits`,
  `inputSchema`, `outputSchema`, `execution`.
- `timeoutMs` must be a positive integer.
- `limits.maxInputBytes` and `limits.maxOutputBytes` must exist and be positive
  integers.
- `execution.kind` must be one of `python`, `node`, `php`, `cli`, `http`.
- Optional blocks (`caps`, `env`, `templating`) must be mappings when present.
- `$ref` schema targets must exist and contain valid JSON Schema documents.

Violations raise `ToolpackValidationError`, keeping problems discoverable before
any runtime execution.

## Testing

Regression coverage lives in `tests/unit/test_toolpacks_loader.py`, exercising:

- Happy-path loading of spec-compliant toolpacks with `$ref` schemas.
- Rejection of snake_case metadata (missing camelCase spec fields).
- Execution kind whitelisting.
- JSON Schema structural validation failures.

The suite runs as part of `./scripts/ensure_green.sh` and must stay green before
shipping changes to the loader.
