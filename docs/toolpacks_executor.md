# Toolpack Executor

The Toolpack executor runs declarative toolpacks produced by the loader. The
initial implementation focuses on the `python` execution kind and enforces
contract validation before and after calling the handler.

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
  `ToolpackExecutionError`.
- Input and output payloads are validated with the schemas bundled in the
  `Toolpack` definition (`jsonschema.validator_for` ensures both the schema and
  the payload are valid).
- Deterministic toolpacks (`deterministic: true`) are cached by a hash of
  `id`, `version`, and the input payload. Cache hits return deep copies so
  callers can mutate outputs safely.
- Handlers are resolved via the `execution.module` field using the
  `module:callable` convention. Async handlers are supported transparently.

## Testing

Unit coverage in `tests/unit/test_toolpacks_exec_python.py` verifies handler
execution, schema validation failures, and deterministic caching. An end-to-end
example using on-disk YAML lives in
`tests/e2e/test_mcp_core_tools_python_only.py`.

Both suites run via `./scripts/ensure_green.sh`.
