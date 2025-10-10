# Phase 3 Preview — Budget Guards and Runner Integration

## Intent
- Stabilise unified budget models (`BudgetSpec`, `BudgetScope`, `BudgetManager`) with preview/commit lifecycle.
- Introduce shared `TraceEventEmitter` for immutable budget and policy traces.
- Retrofit FlowRunner to orchestrate adapters, policies, and budgets with loop stop semantics.

## Design Highlights
- Budget APIs favour mapping proxies to avoid trace mutation.
- Trace emitter bridges budget charges/breaches and policy resolutions with run/loop metadata.
- FlowRunner keeps sequential adapter execution but enforces scopes (run, loop, node, spec) via `BudgetManager` outcomes.

## Test Plan
- `test_budget_manager.py`: preview vs commit, soft/hard breach handling, immutability checks.
- `test_flow_runner_budget_integration.py`: loop halt on budget breach, warnings for soft node budgets, trace fidelity.
