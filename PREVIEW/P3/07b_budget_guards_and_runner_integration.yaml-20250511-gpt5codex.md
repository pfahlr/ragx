# Phase 3 Preview — Task 07b_budget_guards_and_runner_integration

## Overview
- Introduced immutable budget domain objects (`BudgetSpec`, `CostSnapshot`, `BudgetBreach`, `BudgetCheck`, `BudgetChargeOutcome`).
- Added a trace-aware `BudgetManager` that performs preflight and commit bookkeeping while emitting schema-aligned events.
- Implemented a lightweight FlowRunner that coordinates policy enforcement, adapter execution, and budget outcomes via a shared `TraceEventEmitter`.

## Key Design Decisions
1. **Trace Normalisation** — Centralised trace emission through `TraceEventEmitter` to guarantee mapping-proxy payloads and optional sink fan-out.
2. **Deterministic Budget Arithmetic** — Normalised all metrics to floats and stored remaining/overages in immutable mappings to avoid mutable-state leaks noted in Phase 2 reviews.
3. **Stop Semantics** — Encoded soft vs. hard behaviour through `BudgetMode` and `breach_action`, with preflight and commit stages both capable of emitting `budget_breach` events.

## Test Coverage
- `tests/test_budget_manager.py`: Validates soft/hard semantics, trace immutability, and unknown scope handling.
- `tests/test_flow_runner.py`: Exercises runner stop reasons, combined policy/budget tracing, and policy violation handling.

## Next Checks
- Confirm integration with broader DSL policy stack once upstream modules adopt the shared emitter.
- Extend property-based tests for metric normalisation edge cases.
