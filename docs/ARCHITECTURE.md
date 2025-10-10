# RAGX Architecture (Snapshot)

- **Master spec**: `codex/specs/ragx_master_spec.yaml`
- **Component contracts**: `codex/specs/components/*.component.yaml` (source of truth per component)
- **DSL Runner**: executes YAML flows with policies/budgets; linter + run; emits JSONL traces.
- **MCP Server**: deterministic envelope over HTTP/STDIO; loads Toolpacks; validates IO.
- **Toolpacks**: YAML-defined tools (python/node/php/cli/http); loader validates; executor runs python-kind P0.
- **Vector DB Core**: backend registry + shard/merge; CPU-serialized index; PDF + Markdown ingestion.
- **Research pipeline**: planner → executor → publisher (P1).
- **Observability/CI**: lint/type/unit/integration/e2e; coverage gate.

> Codex rule of engagement: implement tests from each component’s `acceptance.tests` first, then code to pass them.
