# Toolpack Loader

The toolpack loader provides read-only access to declarative tool definitions
stored as YAML files. Each `*.tool.yaml` document describes a single toolpack
and may reference JSON/YAML schemas via `$ref`. The loader resolves those
references, validates required fields, and exposes an in-memory catalogue of
ready-to-execute tool metadata.

## Usage

```python
from pathlib import Path
from apps.toolpacks.loader import ToolpackLoader

loader = ToolpackLoader.load_dir(Path("apps/mcp_server/toolpacks"))
for toolpack in loader.list():
    print(toolpack.id, toolpack.version)

markdown = loader.get("tool.echo")
print(markdown.input_schema)
```

## Responsibilities

* Discover every `*.tool.yaml` within the supplied directory (recursively).
* Resolve `$ref` nodes to local JSON/YAML schemas, supporting optional JSON
  Pointer fragments.
* Enforce required metadata (`id`, `version`) while leaving execution-specific
  fields intact for later stages (executors, transports).
* Provide deterministic ordering from `list()` so downstream systems may cache
  results.

The loader intentionally avoids importing runtimes or executing code; it only
normalises declarative metadata.
