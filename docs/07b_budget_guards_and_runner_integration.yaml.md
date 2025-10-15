# Phase 3 – Budget Guards and Runner Integration

## Overview

This document summarises the Phase 3 consolidation of the budget guard branches. The deliverable restores a unified FlowRunner
that cooperates with `PolicyStack`, enforces immutable budget models, and emits deterministic traces aligned with the DSL
contract. Phase 6 finalisation introduced nested-loop recursion and refreshed regression coverage.

Key modules:

| Module | Description |
| ------ | ----------- |
| `pkgs/dsl/budget_models.py` | Defines immutable `ScopeKey`, `CostSnapshot`, `BudgetSpec`, `BudgetChargeOutcome`, and `BudgetDecision` helpers that emit mapping-proxy payloads. |
| `pkgs/dsl/budget_manager.py` | Coordinates scope entry/exit, preview vs commit lifecycles, and breach recording; exposes inspection helpers such as `spent(scope, spec_name)`. |
| `pkgs/dsl/flow_runner.py` | Executes flow nodes through ToolAdapters, invoking policy allowlists before enforcement and charging budgets prior to adapter execution. |
| `pkgs/dsl/trace.py` | Produces immutable `TraceEvent` records and supports optional sinks/validators for schema enforcement. |

## Lifecycle & Control Flow

1. **Runner bootstrap** – `FlowRunner.run()` enters the run scope, emits `run_start`, and prepares node/loop execution (including nested loops handled recursively).
2. **Policy resolution** – Each node invokes `PolicyStack.effective_allowlist([tool])`, emitting `policy_resolved`, before calling
   `PolicyStack.enforce(tool)` to raise `PolicyViolationError` when the allowlist denies the tool.
3. **Budget preview** – `BudgetManager.preview_charge(scope, cost)` computes `BudgetDecision` objects per scope. Breaches trigger
   `BudgetManager.record_breach(decision)` which emits immutable payloads before any stop decisions propagate.
4. **Commit vs stop** – When `decision.should_stop` is true, `BudgetBreachError` is raised; otherwise `BudgetManager.commit_charge`
   updates spend maps and emits `budget_charge` records.
5. **Loop semantics** – `_run_loop()` attaches loop scopes, emits `loop_start`/`loop_iteration_*`, and handles soft vs hard budgets:
   * `breach_action: stop` → emit `loop_stop` with reason `budget_stop`.
   * `breach_action: warn` → emit `budget_breach` but continue iterating.
   * Nested loop bodies call `_run_loop()` recursively so inner loops honour the same guards without leaking parent scopes.
6. **Cleanup** – All scopes exit in `finally` blocks to prevent state leakage. `run_complete` is emitted when execution ends without
   a hard stop.

## Trace Contract

* **Policy** – `policy_push`, `policy_resolved`, `policy_violation`, and `policy_pop` remain available via `PolicyTraceRecorder`.
* **Budgets** – `budget_charge` and `budget_breach` payloads expose `scope_type`, `scope_id`, `spec_name`, `mode`, `breach_action`,
  `breached`, `should_stop`, and nested cost totals. Nested dictionaries are wrapped in `MappingProxyType` to guarantee deep
  immutability.
* **Loops** – `loop_start`, `loop_iteration_start/complete`, `loop_stop`, and `loop_complete` events document loop progress or
  stop reasons.

Attach a validator via `TraceEventEmitter.attach_validator` to enforce field-level invariants. See
`codex/code/07b_budget_guards_and_runner_integration.yaml/tests/test_trace_auto_phase6.py` for examples of validator integration and error
surfacing.

## Configuration Hooks

* **Trace sink** – Supply a callable to `TraceEventEmitter.attach_sink`. The Phase 3 task plan also defines a `runner.trace_sink`
  config flag for dependency injection.
* **Policy sink** – Pass a recorder or sink to `PolicyStack` via its constructor to persist `policy_*` events.
* **Budget specs** – Initialise `BudgetManager` with declarative `BudgetSpec` objects for `run`, `loop`, `node`, and optional custom
  scopes (e.g. `spec`). Specs without certain metrics treat the missing metrics as unbounded while still recording totals.

## CLI & Testing

```bash
# Dedicated regression suite shipped with the task
pytest codex/code/07b_budget_guards_and_runner_integration.yaml/tests -q

# Legacy unit coverage (imports will be normalised in a follow-up)
pytest tests/unit/dsl/test_flow_runner_budget_integration.py -q
```

Regression tests cover:

* Loop hard-stop, soft-warn semantics, nested loop propagation, and run-level stop handling (`test_flow_runner_auto_phase6.py`).
* Nested scope accounting, spec-level budgets, and property-based arithmetic invariants (`test_budget_manager_auto_phase6.py`).
* Trace payload schema validation, sink failure propagation, and validator context (`test_trace_auto_phase6.py`).

## Invariants & Future Work

* **Immutability** – All emitted trace payloads use mapping proxies; mutate state only through new dataclass instances.
* **Policy-first** – `policy_resolved` always precedes budget charging for a node; violations prevent any budget commits.
* **Scope hygiene** – `BudgetManager` refuses duplicate `enter_scope` calls and preserves history for post-run inspection.

Future enhancements:

1. Normalise legacy test imports (`codex.code.work.dsl` → `pkgs.dsl`) to remove aliasing.
2. Expand schema validation to load a published JSON Schema once it lands in `codex/specs/schemas/`.
3. Add async-aware adapters once the runner grows concurrency support.
