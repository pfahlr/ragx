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

Async transports can await toolpacks directly using the asynchronous helpers:

```python
async def invoke():
    result = await executor.run_toolpack_async(pack, {"text": "hello"})
    print(result["text"])
```

To capture execution metrics alongside the payload, use
`run_toolpack_with_stats`:

```python
result, stats = executor.run_toolpack_with_stats(pack, {"text": "hello"})
print(result["text"], stats.duration_ms)
```

Pass ``use_cache=False`` to `run_toolpack_with_stats` when you need to bypass
deterministic caching (for example, to isolate transport-specific invocations)
without flushing previously cached entries. The most recent metrics can always
be retrieved via `executor.last_run_stats()`.

## Behaviour

- Only `execution.kind: python` toolpacks are supported. Other kinds raise
  `ToolpackExecutionError` immediately.
- Input payloads are normalised to mappings and validated against the
  `Toolpack.input_schema`. Output payloads are validated against
  `Toolpack.output_schema` before returning.
- Deterministic toolpacks (`deterministic: true`) are cached by a SHA-256 hash
  of `id`, `version`, and the normalised input payload. Each cache hit returns a
  fresh deep copy, keeping results immutable to the caller.
- `run_toolpack_with_stats` returns an `ExecutionStats` instance describing the
  duration, payload sizes, and cache-hit status for the invocation. It exposes a
  ``use_cache`` flag to bypass caching when required.
- `Executor.last_run_stats()` returns the metrics collected for the most recent
  invocation, allowing transports to preserve observability when they temporarily
  disable caching.
- Non-deterministic toolpacks bypass the cache entirely.
- Handlers are resolved via the `execution.module` field using the
  `module:callable` convention. Invalid formats, missing modules, missing
  callables, or non-callable attributes raise `ToolpackExecutionError`.
- Async handlers (`async def`) are awaited automatically. Use the `run_toolpack_async`
  or `run_toolpack_with_stats_async` helpers when invoking toolpacks from within an
  active event loop.
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
