# AGENTS.md

This repository is set up so autonomous agents (e.g., Codex) can implement large parts of **RAGX** using a single source of truth: `codex/specs/ragx_master_spec.yaml`.

## Golden Rules
1. **Spec-first**: All interfaces, flags, and schemas come from `codex/specs/ragx_master_spec.yaml`. Never invent flags or paths.
2. **Readability-first**: Follow naming/style rules in the spec's `style_guide` section.
3. **Determinism**: Avoid nondeterministic behavior in unit tests (seed randomness, offline fixtures).
4. **Contracts over code**: Implement acceptance tests before filling implementations. CI must pass.
5. **No backdoors**: Tools must respect policy/budget guards in the DSL.

## Quick Start for Agents
- Parse `codex/specs/ragx_master_spec.yaml` and load `arg_spec`, `components`, and `tool_registry`.
- Pick tasks from `codex/agents/TASKS/*.yaml` (start with the lowest number).
- For each task, produce a PR with:
  - Code under the designated path (see task file).
  - Tests under `tests/` as specified.
  - Updates to docs under `docs/` if required.
  - Ensure `make lint typecheck test` passes locally.
- If a flag or schema is missing, add it to the **spec** first in a dedicated PR.

## Commands
```bash
make codex-bootstrap   # sanity checks + print next tasks
make lint              # ruff + yamllint
make typecheck         # mypy
make test              # pytest (unit + e2e)
```

## CI Gates
- Lint + typecheck + unit + integration + e2e.
- Coverage ≥ spec `tests.ci.coverage_minimum` (default 85).

See `codex/agents/CODEX_BOOTSTRAP.md` for more details.

---

## Project Overview

**RAGX** is a clean-slate architecture for **DSL-driven retrieval-augmented generation (RAG)**, built on:

* A declarative **multi-shot DSL** (for flows, policies, budgets, transforms).
* A deterministic **MCP server** exposing tools & prompt packs over HTTP/STDIO.
* **Vector database core** with modular backends (FAISS, HNSW, GPU, adapters).
* **Declarative Toolpacks** (YAML) for Python/Node/CLI/HTTP tools.
* **Research pipeline** (planner → executor → publisher).
* **Observability & CI**: structured logs, traces, schema validation, type/lint gates.

All canonical specs are stored under `codex/specs/`.

---

## Setup Commands

### Clone and bootstrap

```bash
git clone https://github.com/your-org/ragx.git
cd ragx
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```



### Build C++ FAISS backend (optional)

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

This produces `_ragcore_cpp.*.so` for Python bindings.

### Run checks

```bash
make lint
make typecheck
make test
```

---

## Code Conventions

See **style guide** in [`codex/specs/ragx_master_spec.yaml`](codex/specs/ragx_master_spec.yaml#style_guide).

* Python: `PascalCase` for classes, `snake_case` for functions/vars.
* C++: `PascalCase` for types, `snake_case` for methods/vars.
* YAML/JSON: `snake_case` keys.
* CLI flags: `--kebab-case`.
* Booleans: prefixes `is_`, `has_`, `can_`, `should_`.

---

## TDD Protocol (MANDATORY)

> Codex and other coding agents **must follow Test‑Driven Development**. Write/modify tests **before** implementing or changing code. Treat failing tests as your guidance loop.

### Red → Green → Refactor

1. **Red:**

   * For a *new feature*: write failing tests in `tests/` that specify the behavior.
   * For a *bug*: reproduce with a failing **regression test** (name like `test_issue_<id>_regression`).
   * Run `pytest -q` and confirm failures.
2. **Green:**

   * Implement the minimal code to pass the tests.
   * Run `make test` (or `pytest -q`) until green.
3. **Refactor:**

   * Improve code structure, keep tests green.
   * Run `make fmt lint test` to ensure style and static checks.

### Test authoring rules

* **Mirroring:** tests mirror source paths (e.g., `src/services/retrieval/foo.py` → `tests/services/retrieval/test_foo.py`).
* **Coverage gate:** target ≥ **80%** for changed files. If under target, add tests. Command:

  ```bash
  pytest -q --cov=src --cov-report=term-missing
  ```
* **Property tests (where useful):** use `hypothesis` for pure functions and text transforms.
* **Snapshot tests:** for deterministic RAG outputs (store under `tests/snapshots/`).
* **Network isolation:** mock providers and HTTP; mark external calls with `-m external`.
* **Fixtures:** place shared fixtures in `tests/conftest.py` and `tests/fixtures/`.
* **Regression first:** any discovered bug must first land as a failing test.

### Self‑feedback loop for agents

* If tests fail, **read the traceback**, locate the function, and iterate.
* If `ruff`/`mypy` fail, fix style/types **before** pushing.
* Commit tests and implementation together; PR description must note new/changed tests.

### Optional but recommended

* **Mutation tests:** (e.g., `mutmut`) for critical modules.
* **Pre‑commit hooks:** add `pre-commit` with `ruff`, `black`, and `pytest -q` (fast subset) on commit.

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

## Codex & Agent Usage

### 1. Codex Prompts

* All specs in `codex/specs/` are **machine-readable**.
* Codex agents should always load the canonical flag definitions from `arg_spec` rather than inventing new ones.
* When creating or updating components, agents must:

  1. Reference component `id` in `components:`.
  2. Ensure CLI flags match `arg_spec`.
  3. Preserve style rules from `style_guide`.

### 2. Development Workflow

* **One spec → many outputs.**
  Codex may be asked to:

  * Generate documentation (`/docs`).
  * Produce test fixtures (`/tests`).
  * Update implementation code (`/pkgs`).
* Patches should be applied with `git apply` (agents: generate unified diff).

### 3. CI Rules

* Unit tests must pass locally (`pytest`).
* CI runs lint/type/unit/integration/e2e defined in spec.
* External-network variability should be mocked in fixtures.

---

## Human-in-the-Loop

* Agents may request user approval for **outlines** and **question merges** via MCP `hitl.*` tools.
* Default `--approve-outline` is `noninteractive`.
* Editors can switch to `--approve-outline=editor` for manual review.

---

## Contributing

1. Fork and clone the repo.
2. Make changes in feature branch.
3. Ensure specs are updated (`codex/specs/`).
4. Run `make test` and `make lint`.
5. Submit PR with link to relevant spec section.
