# Toolpacks Loader

The Toolpacks loader is responsible for discovering declarative tool definitions
(`*.tool.yaml`) and preparing them for execution. Its implementation follows the
contracts in [`codex/specs/ragx_master_spec.yaml`](../codex/specs/ragx_master_spec.yaml)
and currently focuses on schema resolution and validation.

## Responsibilities

* Recursively walk a directory to find every `*.tool.yaml` file.
* Parse the YAML definition and materialize it as an in-memory `Toolpack` object.
* Resolve `$ref` entries under `inputSchema` and `outputSchema` to concrete JSON
  Schemas relative to the toolpack file.
* Validate JSON Schemas using `jsonschema`'s `validator_for(...).check_schema` so
  malformed schemas fail fast.
* Enforce structural contracts for each toolpack (`id`, `version`,
  `deterministic`, `timeoutMs`, `limits`, `execution.kind`, etc.).
* Reject duplicate tool identifiers and expose deterministic ordering via
  `ToolpackLoader.list()`.

## Public API

```python
from apps.toolpacks.loader import ToolpackLoader

loader = ToolpackLoader()
loader.load_dir("apps/mcp_server/toolpacks")
for pack in loader.list():
    print(pack.id, pack.execution["kind"])
```

* `ToolpackLoader.load_dir(path)` resets the loader state and imports the full
  directory tree (raising `ToolpackValidationError` on any invalid file).
* `ToolpackLoader.list()` returns the loaded `Toolpack` instances sorted by id.
* `ToolpackLoader.get(tool_id)` returns the `Toolpack` for a specific id or
  raises `KeyError` if it is unknown.

## Validation Rules

The loader enforces the following invariants:

* `id` – lowercase dotted identifier (e.g. `domain.tool.action`).
* `version` – [SemVer](https://semver.org) with explicit `major.minor.patch`.
* `deterministic` – boolean flag.
* `timeoutMs` – positive integer.
* `limits.maxInputBytes` / `limits.maxOutputBytes` – positive integers.
* `execution.kind` – one of `{python, node, php, cli, http}`.
* `execution` – requires a concrete entry point per kind (`module`/`script` for
  python, `cmd` list for CLI, `url` for HTTP, etc.).
* `caps` – merged with declared timeout/size limits and validated network list
  (defaults to network-off).
* `env` / `templating` – mappings with string keys and validated value types.
* `$ref` schema paths must exist and contain valid JSON Schema documents.

These checks keep downstream components deterministic and guard against loading
invalid or malformed toolpacks.

## Testing

Unit coverage lives in `tests/unit/test_toolpacks_loader.py` and exercises:

* Success path with multiple toolpacks and nested directories.
* `$ref` schema resolution.
* Duplicate id detection and missing field validation.
* `get()` failure semantics for unknown tool ids.

The suite runs as part of the global `pytest` invocation and must remain green
before any pull request is considered complete.
