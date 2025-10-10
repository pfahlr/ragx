# Phase 3 Preview – Budget guards and runner integration

## Objective
Implement a self-contained FlowRunner prototype that unifies structured budget models, a scope-aware BudgetManager, and TraceWriter-backed telemetry while keeping adapter-driven execution and policy enforcement deterministic for unit testing.

## Key Components
- Canonical cost utilities that normalise seconds to milliseconds and guard metric names.
- Immutable budget value objects and exceptions backing a `BudgetManager` orchestrator with scope registration and warning aggregation.
- A `TraceEventEmitter` façade plus minimal `PolicyStack` to keep runner concerns loosely coupled.
- FlowRunner implementation exercising adapter estimate/execute lifecycle, run/node/loop scopes, and trace emission.

## Test Strategy
Two pytest modules cover:
1. Cost normalisation and budget manager semantics (soft warn, hard breach, loop stop).
2. End-to-end runner behaviour for run halts, node warnings, and loop stop budgets with trace assertions.

