# Codex Run Policy (Zero-Touch)

- Source of truth: `codex/specs/ragx_master_spec.yaml`. Do not invent flags or paths.
- For each task:
  1. Produce a single **unified diff** patch (no prose).
  2. Include tests and docs as needed.
  3. Make `ruff`, `mypy`, and `pytest` pass locally.
- If a spec gap blocks the task:
  - Create a *separate* spec-updating patch first (new branch), then proceed with implementation.
- Respect style_guide and arg_spec throughout.
- Never access vector DB directly from Task Runner — only via MCP tools.
- Research Collector is out-of-repo (mention as TODO link only).
