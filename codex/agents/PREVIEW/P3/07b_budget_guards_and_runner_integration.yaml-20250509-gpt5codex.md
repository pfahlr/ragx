# Phase 3 Preview — Budget Guards & Runner Integration

## Summary
- Establish immutable budget value objects (`BudgetSpec`, `CostSnapshot`, `BudgetChargeResult`) with normalization helpers and mapping-proxy payloads.
- Deliver a hierarchical `BudgetManager` that coordinates scope registration, preflight checks, commit accounting, and trace emission.
- Extend `FlowRunner` to drive adapters through estimate/execute hooks, respect budget preflight stop signals, and emit structured loop stop and breach events via a shared `TraceEventEmitter`.

## Planned Scope
- Code lives under `codex/code/07b_budget_guards_and_runner_integration.yaml/` with task-local namespace packaging.
- Tests cover models, manager orchestration, and runner behaviour with ≥85% coverage targets per plan.
- Documentation + run artefacts produced in PREVIEW/REVIEW/POSTEXECUTION directories for downstream agents.

## Key Considerations
- Preflight breaches on hard budgets raise immediately while still emitting `budget_breach` traces for auditability.
- Stop-on-soft budgets return structured decisions so loops can terminate gracefully without exceptions.
- Trace payloads are recursively frozen (mapping proxies + tuples) to meet observability contracts and avoid mutation in tests.
