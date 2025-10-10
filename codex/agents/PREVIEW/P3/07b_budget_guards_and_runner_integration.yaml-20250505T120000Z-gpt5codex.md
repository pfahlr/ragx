# Phase 3 Preview — Budget Guards and Runner Integration

## Highlights
- Added immutable budget domain models (`BudgetSpec`, `CostSnapshot`, `BudgetDecision`) with normalization helpers and stop semantics that differentiate hard vs. soft behaviour.
- Introduced a `BudgetManager` orchestrator with scope-aware meters, trace emission, and loop stack management.
- Implemented a minimal FlowRunner capable of executing unit and loop nodes while enforcing policy allowlists and budget guards.
- Shared `TraceEventEmitter` ensures schema-friendly, immutable payloads reused by budgets and runner traces.

## Test Plan
- `pytest codex/code/work/tests/unit/test_budget_models.py`
- `pytest codex/code/work/tests/unit/test_budget_manager.py`
- `pytest codex/code/work/tests/unit/test_flow_runner_budgets.py`

## Open Considerations
- Transform/decision node execution is deferred; follow-up tasks should expand FlowRunner coverage.
- Trace payload schema validation is stubbed; future work should integrate JSON schema checks.
- Runner currently propagates first policy/budget violation; multi-error aggregation may be desirable later.
