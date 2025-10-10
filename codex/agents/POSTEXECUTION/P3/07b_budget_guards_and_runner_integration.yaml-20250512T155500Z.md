# Phase 3 Post-Execution – 07b_budget_guards_and_runner_integration

## Test & Coverage Summary
- `pytest codex/code/work/tests -q`
  - Manager suite validates normalization, hard-stop detection, soft warnings, and trace ordering.
  - Runner suite validates run halting, node soft warnings, loop stop signalling, and policy trace emission.
- All phase-specific tests pass; global suite not executed in this phase.

## Implementation Notes
- `CostSnapshot` treats token/call limits as opt-in to avoid noisy warnings when specs omit them.
- BudgetManager preflight emits breaches/warnings with phase markers (`preflight`, `commit`) for observability parity.
- FlowRunner reuses shared `TraceEventEmitter` for budgets, policies, and loop stop events, returning deterministic `RunResult` objects.
