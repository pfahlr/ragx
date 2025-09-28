# AGENTS.md

This repository is set up so autonomous agents (e.g., Codex) can implement large parts of **RAGX** using a single source of truth: `codex/specs/ragx_master_spec.yaml`.

---

## Golden Rules

1. **Spec-first**: All interfaces, flags, and schemas come from `codex/specs/ragx_master_spec.yaml`. Never invent flags or paths.
2. **Readability-first**: Follow naming/style rules in the spec's `style_guide` section.
3. **Determinism**: Avoid nondeterministic behavior in unit tests (seed randomness, offline fixtures).
4. **Contracts over code**: Implement acceptance tests before filling implementations. CI must pass.
5. **No backdoors**: Tools must respect policy/budget guards in the DSL.
6. **Tests First (absolute gate)**: Do nothing until the test harness runs cleanly. All tasks stop if `./scripts/ensure_green.sh` is failing. Check it often.
7. **Test Driven Development: your environment**:  Consider your development environment broken if tests are not running, stop everything and focus on getting them working before doing anything else.
8. **Test Driven Development: your code**: Consider your code not working if tests are failing, use their output to determine what is wrong with your code




---

## Quick Start for Agents

* Parse `codex/specs/ragx_master_spec.yaml` and load `arg_spec`, `components`, and `tool_registry`.
* Pick tasks from `codex/agents/TASKS/*.yaml` (start with the lowest number).
* For each task, produce a PR with:

  * Code under the designated path (see task file).
  * Tests under `tests/` as specified.
  * Updates to docs under `docs/` if required.
  * Ensure **`./scripts/ensure_green.sh`** passes locally before proposing code changes.
* If a flag or schema is missing, add it to the **spec** first in a dedicated PR.

---

## Commands

```bash
make codex-bootstrap   # sanity checks + print next tasks
make lint              # ruff + yamllint
make typecheck         # mypy
make test              # pytest (unit + e2e)
./scripts/ensure_green.sh  # full gate (must pass before anything else)
```

---

## Stop Conditions

Agents must **abort work and restore the harness** if:

1. `pytest` cannot run (import errors, broken config).
2. Linters/type checks (`ruff`, `mypy`, `yamllint`) fail to execute.
3. Native C++/GPU toolchain is required but unavailable and no skip markers exist.
4. `/eval/verification` corpus is missing/unreadable.
5. `./scripts/ensure_green.sh` fails for any reason.

---

## CI Gates

* Lint + typecheck + unit + integration + e2e.
* Coverage ≥ spec `tests.ci.coverage_minimum` (default 85).
* CI steps mirror `./scripts/ensure_green.sh`.

---

## Project Overview

**RAGX** is a clean-slate architecture for **DSL-driven retrieval-augmented generation (RAG)**, built on:

* A declarative **multi-shot DSL** (flows, policies, budgets, transforms).
* A deterministic **MCP server** exposing tools & prompt packs over HTTP/STDIO.
* **Vector database core** with modular backends (FAISS, HNSW, GPU, adapters).
* **Declarative Toolpacks** (YAML) for Python/Node/CLI/HTTP tools.
* **Research pipeline** (planner → executor → publisher).
* **Observability & CI**: structured logs, traces, schema validation, type/lint gates.

All canonical specs are stored under `codex/specs/`.

---

## Environment Bootstrap

Agents must ensure:

```bash
python -V
pip install -r requirements.txt
pip install ruff mypy pytest coverage yamllint
```

Optional native build:

```bash
sudo apt-get install -y cmake ninja-build ccache build-essential
pip install pybind11
```

If not available, fall back to dummy backend and skip native/gpu tests.

---

## One Command Gate

**File:** `scripts/ensure_green.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

ruff check .
mypy .
yamllint .

pytest --maxfail=1 --disable-warnings ${PYTEST_ADDOPTS:-}

echo "[ensure_green] OK"
```

Executable:

```bash
chmod +x scripts/ensure_green.sh
```

---

## TDD Protocol (MANDATORY)

> Codex and other agents must follow **Test-Driven Development**.

### Red → Green → Refactor

1. **Red**: write failing tests first.
2. **Green**: implement minimal code to pass.
3. **Refactor**: improve code structure, keep tests green.

### Test rules

* Mirror source paths under `tests/`.
* Coverage ≥ 80% for changed files.
* Use `hypothesis` where helpful.
* Mock network/external providers.
* Add regression tests before fixes.

---

## Components & Entrypoints

| Component          | Entrypoint           | CLI Command        |
| ------------------ | -------------------- | ------------------ |
| DSL Runner         | `pkgs/dsl/`          | `task-runner run`  |
| MCP Server         | `apps/mcp_server/`   | `mcp-server`       |
| Toolpacks Runtime  | `apps/toolpacks/`    | loaded by server   |
| Vector DB Core     | `ragcore/`           | `vectordb-builder` |
| Retrieval + Rerank | `pkgs/retrieval/`    | `rw search`        |
| Planner            | `pkgs/planner/`      | `rw plan`          |
| Research Pipeline  | `pkgs/research/`     | `rw research`      |
| Observability & CI | `.github/workflows/` | `make lint test`   |

---

## Human-in-the-Loop

* Agents may request approval for outlines/questions via MCP `hitl.*` tools.
* Default `--approve-outline` is `noninteractive`.

---

## Contributing

1. Fork + clone.
2. Feature branch.
3. Update specs if needed.
4. Run `./scripts/ensure_green.sh`.
5. Submit PR with spec references.

---

### TL;DR for Agents

* **Run `./scripts/ensure_green.sh` first.**
* If red, fix harness before coding.
* Add failing test → implement → pass tests → refactor.
* No commit/PR is valid unless tests are green.



