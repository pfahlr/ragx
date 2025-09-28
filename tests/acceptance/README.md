# Acceptance Test Suite (Skeletons)

These tests define **Done Criteria** for core components per the RAGX Master Spec.
They are initially marked `xfail` or `skip` and should be enabled as implementations land.

Components covered:
- DSL Runner (task-runner)
- MCP Server (HTTP/STDIO envelope + toolpacks)
- Vector DB Core: FAISS backend (C++ + pybind11)

Run locally:
```bash
pytest -q tests/acceptance -m "not slow"
```
