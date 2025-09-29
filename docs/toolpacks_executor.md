# Toolpack Executor

The Toolpack executor runs declarative toolpacks produced by the loader. The
initial implementation focuses on the `python` execution kind and enforces
contract validation before and after calling the handler (input schemas,
output schemas, and deterministic caching).

## Usage

```python
from apps.toolpacks.executor import Executor
from apps.toolpacks.loader import ToolpackLoader

loader = ToolpackLoader()
loader.load_dir("apps/mcp_server/toolpacks")
executor = Executor()

pack = loader.get("tool.echo")
result = executor.run_toolpack(pack, {"text": "hello"})
print(result["text"])
```

## Behaviour

- Only `execution.kind: python` toolpacks are supported. Other kinds raise
  `ToolpackExecutionError` immediately.
- Input payloads are normalised to mappings and validated against the
  `Toolpack.input_schema`. Output payloads are validated against
  `Toolpack.output_schema` before returning.
- Deterministic toolpacks (`deterministic: true`) are cached by a SHA-256 hash
  of `id`, `version`, and the normalised input payload. Each cache hit returns a
  fresh deep copy, keeping results immutable to the caller.
- Non-deterministic toolpacks bypass the cache entirely.
- Handlers are resolved via the `execution.module` field using the
  `module:callable` convention. Invalid formats, missing modules, missing
  callables, or non-callable attributes raise `ToolpackExecutionError`.
- Async handlers (`async def`) are awaited automatically via `asyncio.run`.
- Schema-level problems raised by the loader (invalid `$ref` targets or schema
  definitions) surface as `ToolpackValidationError`, while runtime or validation
  problems surface as `ToolpackExecutionError`.

## Testing

Unit coverage in `tests/unit/test_toolpacks_exec_python.py` verifies handler
execution, schema validation failures, deterministic cache behaviour, cache
bypass for non-deterministic packs, entrypoint validation, and async handler
support. An end-to-end example using on-disk YAML lives in
`tests/e2e/test_mcp_core_tools_python_only.py`.

Both suites run via `./scripts/ensure_green.sh`.
