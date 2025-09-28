# Toolpack Executor

The toolpack executor runs declarative toolpacks produced by the loader. The
current implementation focuses on the `python` kind and provides:

* JSON Schema validation for both input and output payloads.
* Environment assembly combining executor-level defaults and per-toolpack
  overrides.
* A simple idempotency cache for deterministic toolpacks.

## Usage

```python
from pathlib import Path
from apps.toolpacks.loader import ToolpackLoader
from apps.toolpacks.executor import ToolpackExecutor

loader = ToolpackLoader.load_dir(Path("apps/mcp_server/toolpacks"))
executor = ToolpackExecutor(loader=loader)

result = executor.run("tool.echo", {"text": "hello"})
print(result)
```

## Caching Behaviour

Toolpacks that declare `deterministic: true` are cached in-memory based on the
combination of tool id, input payload, and resolved environment variables. Each
cache hit returns a deep copy, so callers can safely mutate the returned
structures without affecting future executions.

## Handler Signature

Python handlers may expose one of the following signatures:

* `handler(payload)`
* `handler(payload, context)`
* `handler(payload, *, context)`

The executor injects an `ExecutionContext` that currently exposes a single
field, `env`, containing the merged environment mapping.
