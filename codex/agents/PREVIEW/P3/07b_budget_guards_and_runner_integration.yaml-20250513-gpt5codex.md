# Phase 3 Preview — 07b_budget_guards_and_runner_integration

## Overview
- Implemented canonical budget value objects (`BudgetSpec`, `CostSnapshot`, `BudgetDecision`) with strict millisecond normalization and immutable payload helpers.
- Built a scope-aware `BudgetManager` coordinating preflight/commit enforcement with warning aggregation and immutable snapshots.
- Added a schema-validated `TraceEventEmitter` to unify policy and budget telemetry.
- Wired a simplified, adapter-driven `FlowRunner` that exercises run/spec/loop/node budgets, orchestrates policy traces, and propagates breach diagnostics to stop reasons.

## Planned Test Coverage
- Unit tests target budget models, manager semantics, and FlowRunner end-to-end budget behaviour with mocked adapters.
- Trace schema validation is performed inline via emitter enforcement, ensuring payload parity across events.

## Dependencies & Assumptions
- Tests rely on lightweight loader helpers to import modules from the task-specific directory without polluting global packages.
- Flow specifications in tests model sequential node execution; parallelism and asynchronous adapters remain out of scope for this phase.
