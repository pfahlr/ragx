# 05x Feature Branch - features/05-toolpacks(-local)


## New CLI Operations, Class interfaces, 

These operations cover the capabilities delivered across tasks 05a–05j on the current feature branch.

### Toolpack Executor

- **Execute a python toolpack end-to-end:**

    # Confirms schema validation, async support, deterministic caching (mutate result and re-run), and error handling

    loader = ToolpackLoader(); loader.load_dir(Path("apps/mcp_server/toolpacks"))
    executor = Executor()
    result = executor.run_toolpack(loader.get("tool.echo"), {"text": "hello"})

- **Trigger execution errors:**

   run the executor on a **non-python toolpack** or supply **invalid input/output payloads** to observe `ToolpackExecutionError`

### Docs & Diagnostics

  - Review docs/toolpacks_loader.md and docs/toolpacks_executor.md for the consolidated invariants, failure semantics, and usage snippets—all synced with the spec.
  - Check logs for legacy shim warnings (Toolpack ... used legacy snake_case keys) or validation errors while loading malformed toolpacks.

### CLI / Scripts

  - Use `./scripts/ensure_green.sh` for the full lint/type/test gate; focus on unit suites covering loader (tests/unit/test_toolpacks_loader.py) and executor (tests/unit/
  test_toolpacks_exec_python.py).
  - Run targeted tests interactively, e.g.
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 venv/bin/pytest tests/unit/test_toolpacks_loader.py -k execution.

  These operations cover the capabilities delivered across tasks `05a–05j` on the current feature branch.
  
