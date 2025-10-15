# RAGX

> A deterministic, test-first RAG stack with a modular vector DB core, declarative Toolpacks, and a multi-transport MCP server.

---

## TL;DR

```bash
# 1) Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) (Optional) Build native backends
./scripts/build_cpp.sh

# 3) Verify everything (lint → types → yaml → tests)
./scripts/ensure_green.sh

# 4) Run MCP server (HTTP + STDIO, deterministic IDs, logs to ./runs)
python -m apps.mcp_server.cli --http --stdio --deterministic-ids --log-dir runs

# 5) Build a toy index (see CLI reference below)
python -m ragcore.cli build --backend py_flat --index-kind flat --metric ip \
  --dim 384 --corpus-dir ./docs --out ./artifacts/index
````

---

## 1) Project Overview

RAGX is a retrieval-augmented generation platform built around:

* **`ragcore/`** — a modular vector-database toolkit (backends, ingestion, registry) and a **builder CLI** for listing, building, and merging indexes.
* **`apps/mcp_server/`** — an MCP (Model Context Protocol) service exposing **discover / prompt / tool** over **HTTP** and **STDIO**, with **structured logging**, **schema validation**, **guardrails** (timeouts/byte limits), **deterministic IDs**, and **idempotency hooks**.
* **`apps/toolpacks/`** — **declarative Toolpacks** (YAML + JSON Schemas) and a deterministic **Executor** (python kind) with **input/output schema enforcement**, **result caching**, and **execution telemetry**.
* **`codex/specs/`** — authoritative YAML specs (tasks, component contracts, schemas).
* **`scripts/`** — CI guardrails, **DeepDiff** log comparators with volatile-field masking, example ops, and Codex task helpers.
* **`flows/`** — example DSL flows for orchestration experiments.

**Core principles:** contract-driven design, deterministic pipelines, transport parity (HTTP↔STDIO), snapshot testing with stable goldens, and small reversible changes.

---

## 2) Developer Setup

### 2.1 Prerequisites

* **Python**: CPython **3.11** (virtualenv recommended)
* **Build tools** (optional, for native backends): `cmake`, a C/C++ toolchain, `pybind11`
* **Optional ingestion**: `pypdf` (for PDF parsing)

### 2.2 Environment Variables

* `RAGCORE_DISABLE_CPP=1` — force Python-only vector backends (ignore native shim)
* `RAGX_MCP_ENVELOPE_VALIDATION` — `off | shadow | enforce` (default: shadow)
* `RAGX_SEED` — deterministic seed for tie-breakers in sample vector tools
* `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` — set by `ensure_green.sh` for reproducible tests
* (Server guardrails are also exposed via CLI/env; see CLI reference)

### 2.3 Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# optional native build:
./scripts/build_cpp.sh
# verify:
./scripts/ensure_green.sh
```

---

## 3) CLI Reference (Quick)

> Use `make` targets or call modules directly.

### Make targets

* `make lint` — Ruff + YAML lint
* `make typecheck` — mypy
* `make test` — pytest (fast-fail)
* `make check` — lint + typecheck + test
* `make codex-bootstrap [TASK_LIMIT=N]` — next Codex tasks
* `make unit | integration | e2e | acceptance` — scoped pytest suites

### MCP server

```bash
python -m apps.mcp_server.cli \
  [--http] [--stdio] [--host 0.0.0.0] [--port 3333] \
  [--max-connections N] [--shutdown-grace S] \
  [--log-level LEVEL] [--once] [--deterministic-ids] \
  [--log-dir PATH] [--max-input-bytes BYTES] \
  [--max-output-bytes BYTES] [--timeout-ms MS]
```

* **`--once`**: deterministic smoke cycle (discover → prompt → tool → exit)
* **Guardrails**: `--max-input-bytes`, `--max-output-bytes`, `--timeout-ms`
* **Determinism**: `--deterministic-ids` for reproducible run/trace IDs
* **Transports**: enable one or both of `--http`, `--stdio` (defaults to HTTP)

### Vector DB builder

```bash
# list backends/capabilities
python -m ragcore.cli list

# build an index from a corpus dir
python -m ragcore.cli build \
  --backend py_flat --index-kind flat --metric ip --dim 384 \
  --corpus-dir ./data/corpus --out ./artifacts/index \
  --accept-format md --accept-format pdf \
  [IVF/HNSW/PQ params: --nlist --nprobe --pq-m --pq-bits --M --ef-construction --ef-search] \
  [--train-vectors FILE] [--add-vectors FILE]

# merge multiple shard dirs into one index
python -m ragcore.cli merge \
  --merge ./shards/000 --merge ./shards/001 --out ./artifacts/merged
```

---

## 4) Budget Guards + FlowRunner Integration (Phase 3)

The Phase 3 sandbox reconciles the budget guard branches into production-ready modules under `pkgs/dsl/`:

* **`pkgs/dsl/budget_models.py`** — immutable `CostSnapshot`, `BudgetSpec`, and `BudgetDecision` helpers that export mapping-proxy trace payloads.
* **`pkgs/dsl/budget_manager.py`** — manages scope lifecycle, preview/commit/record flows, and emits `budget_charge`/`budget_breach` traces.
* **`pkgs/dsl/flow_runner.py`** — orchestrates ToolAdapters, BudgetManager, and PolicyStack with deterministic trace ordering (`policy_resolved` → `budget_breach` → `loop_stop`) and nested-loop execution support.
* **`pkgs/dsl/trace.py`** — `TraceEventEmitter` producing deeply immutable `TraceEvent` payloads with optional sinks/validators.

### 4.1 Quick validation

```bash
# Phase 6 regression coverage (nested loops, trace immutability)
pytest codex/code/07b_budget_guards_and_runner_integration.yaml/tests -q

# Legacy unit coverage (imports will be updated in future phases)
pytest tests/unit/dsl/test_budget_manager.py -q
```

### 4.2 Execution flow

1. `FlowRunner.run()` enters the run scope, emits `run_start`, and iterates nodes/loops.
2. For each node, the runner calls `PolicyStack.effective_allowlist()` to emit `policy_resolved`, then `PolicyStack.enforce()` to raise `PolicyViolationError` when needed.
3. `BudgetManager.preview_charge()` returns a `BudgetDecision`; if `decision.breached`, `record_breach()` emits immutable payloads and `BudgetBreachError` propagates when `should_stop`.
4. Loop execution honours `breach_action` semantics: `stop` halts the loop (`loop_stop`), while `warn` keeps iterating after emitting `budget_breach`.

### 4.3 Extension hooks

* Inject custom trace sinks by passing `TraceEventEmitter` instances (or setting `runner.trace_sink` via config flags defined in the Phase 3 task plan).
* Policy instrumentation can attach a `PolicyTraceRecorder` or sink to observe `policy_push/pop/resolved/violation` events.
* Downstream adapters should honour the `ToolAdapter` protocol (`estimate_cost`, `execute`) to integrate with the budget manager seamlessly.

### 4.4 Acceptance targets

* Loop budget stops, soft warnings, and nested-loop recursion (`test_flow_runner_auto.py`).
* Policy/budget interleaving and run-level hard stops.
* Trace payload immutability (deep freezing) and validator error surfacing (`test_trace_auto.py`).

### Log diff & verification

```bash
# DeepDiff comparisons with volatile-field whitelisting
python -m scripts.diff_core_tool_logs --new runs/core_tools/latest.jsonl --golden tests/fixtures/... --whitelist ts runId traceId
python -m scripts.diff_mcp_server_logs  --new runs/mcp_server/...latest.jsonl --golden tests/fixtures/... --whitelist ts runId traceId
python -m scripts.diff_envelope_validation_logs --new runs/mcp_server/envelope_validation.latest.jsonl --golden tests/fixtures/... --whitelist ts runId traceId
# Curated end-to-end example scripts
./scripts/verify_example_operations.sh
```

---

## 4) Key Modules & Responsibilities

### 4.1 `ragcore/` (vector core)

* **`interfaces.py`** — `Backend`/`Handle` protocols, `IndexSpec` validation, `SerializedIndex` DTOs
* **`backends/`** — simulated **FAISS/HNSW/cuVS/PyFlat/Dummy** backends sharing `VectorIndexHandle` (train/add/search/serialize/GPU/merge)
* **`ingest/`** — `scan_corpus`, `parse_markdown`, `parse_pdf` (optional `pypdf`)
* **`registry.py`** — register/get/list backends, test reset helper
* **`cli.py`** — `list | build | merge` commands orchestrating ingestion, embeddings, artifacts

### 4.2 `apps/toolpacks/` (tools)

* **`loader.py`** — loads `*.tool.yaml` (camelCase), resolves `$ref`, validates schema/env/limits/caps, enforces unique IDs
* **`executor.py`** — python-kind execution with **input/output schema** checks, **deterministic caching**, and **ExecutionStats** (duration/inputBytes/outputBytes/cacheHit)
* **`python/core/*`** — reference tools: `docs.load.fetch`, `exports.render.markdown`, `vector.query.search` (deterministic ranking via `RAGX_SEED`)

### 4.3 `apps/mcp_server/` (service & transports)

* **Service**: `McpService` implements **discover / get_prompt / invoke_tool / health**

  * **Guardrails**: global + per-tool **timeouts/byte limits**
  * **Validation**: tool IO & envelope validation (`off | shadow | enforce`)
  * **Determinism**: UUID5 request/trace IDs (opt-in), cache-aware telemetry
  * **Idempotency hooks**: key/plumbing for safe retries (short-circuit on duplicate)
  * **Errors**: canonical error mapping (HTTP & JSON-RPC)
* **Transports**:

  * `http/` — FastAPI router for MCP endpoints + `/healthz`
  * `stdio/` — JSON-RPC 2.0 over stdin/stdout (newline-delimited frames)
* **Logging**: `logging.py` JSONL **structured logs** (run/attempt IDs, retention, `.latest` symlink)
* **Validation logs**: schema audit logs emitted even in **shadow** mode

---

## 5) Defining a Tool (Two paths)

### A) **Toolpack (recommended)**

```
apps/toolpacks/word_count/
├─ toolpack.yaml           # module:function, limits, schemas, idempotency key template
├─ input.schema.json
├─ output.schema.json
└─ tool.py                 # def run(params: dict, *, ctx=None) -> dict
```

* **Why**: test-first, schema-driven, centrally enforced guardrails, deterministic caching/telemetry
* **Executor** attaches `meta.execution` and (if enabled) `meta.idempotency`

### B) **Raw MCP tool (direct registration)**

* Register a handler with input schema directly on the server (bypasses Toolpack manifest).
* Useful for prototypes; prefer Toolpacks for consistency and CI contracts.

---

## 6) Testing & Determinism

* **Suites**: `tests/unit`, `tests/integration`, `tests/e2e`, `tests/acceptance`
* **Transport parity**: run identical scenarios over **HTTP** and **STDIO**; assert same outcomes and telemetry (after masking volatile fields).
* **Snapshots**: compare JSONL logs using **DeepDiff** with a **whitelist** of stable fields; mask `ts`, `runId`, `traceId`, `spanId`, `attemptId`, `requestId`.
* **Deterministic fixtures**: fixed seeds + stable payload sizes; exact telemetry expected where meaningful.

---

## 7) DevOps & Automation

* **One-command gate**: `./scripts/ensure_green.sh` (fails fast; honors `PYTEST_ADDOPTS`)
* **Example operations**: `./scripts/verify_example_operations.sh` runs curated shell flows end-to-end
* **Codex helpers**: `make codex-bootstrap` or `python -m scripts.codex_next_tasks --format json`

---

## 8) Known Limitations / Future Work

* Backends are **simulations** unless the native extension is built; real FAISS/HNSW/cuVS integrations require native bindings/GPU.
* Toolpack **executor** currently implements **python** kind; other kinds validate config but are not executed.
* Envelope validation defaults to **shadow**; set `RAGX_MCP_ENVELOPE_VALIDATION=enforce` to hard-fail schema violations.
* Example operations assume a local verification corpus; adjust paths for your environment.

---

## 9) Glossary

* **MCP** — transport-agnostic protocol for tools/prompts over HTTP/STDIO
* **Toolpack** — declarative tool definition (YAML + schemas + limits + exec entrypoint)
* **Envelope** — canonical response (`ok/data/error/meta`) with execution/idempotency metadata
* **DeepDiff** — JSON structure diff used for golden/snapshot comparisons
* **Shard/Docmap** — build artifacts (index spec/bin + document map) produced by `ragcore.cli`

---

## 10) Contributing

* Keep PRs **small & reversible**; update/commit goldens with volatile-field masking.
* Preserve **camelCase** in manifest/schema keys (spec alignment).
* Avoid global state (e.g., “last_run_stats”); keep metrics **request-scoped**.
* Prefer adding tests **before** or alongside implementation; aim for **transport parity** for all tool flows.
