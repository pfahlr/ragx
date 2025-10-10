# Phase 3 Review — 07b_budget_guards_and_runner_integration

## Summary
- Budget models normalise all limits/costs to milliseconds and expose immutable payload projections for trace usage.
- BudgetManager tracks run/spec/loop/node scopes, differentiating warn vs stop behaviour and capturing warnings without duplicates.
- TraceEventEmitter validates schema compliance for policy and budget events, ensuring deterministic payload shapes.
- FlowRunner integrates adapters, policy stack, and BudgetManager to surface stop reasons, warnings, and trace emissions.

## Verification Checklist
- [x] Budget normalization matches milliseconds-only contract (unit tests in `test_budget_models.py`).
- [x] Hard vs soft breach actions propagate correctly to run results and traces (`test_budget_manager.py`, `test_flow_runner.py`).
- [x] Trace payloads satisfy schema validation for policy and budget events (emitter enforces schema; exercised in integration tests).
- [x] FlowRunner preserves adapter execution order when budgets permit (integration tests ensure node sequencing and outputs).
- [x] Warnings accumulate deterministically without duplicate emissions (manager warning aggregation verified in tests).

## Known Issues / Follow-ups
- Current FlowRunner assumes sequential node execution and synchronous adapters; concurrency handling remains future work.
- Trace emitter schema is local to this task directory; upstream integration should reconcile with global observability schemas.
