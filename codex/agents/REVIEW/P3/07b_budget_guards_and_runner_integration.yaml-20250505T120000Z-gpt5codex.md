# Phase 3 Review — Budget Guards & Runner Integration

## Summary
- Domain models ensure consistent metric conversion, immutable snapshots, and BudgetMode semantics.
- BudgetManager previews/commits across parent-child scopes with deterministic event emission.
- FlowRunner coordinates adapters and PolicyStack enforcement while emitting budget/policy/loop-stop traces.

## Review Checklist
- [x] Cost normalization handles seconds→milliseconds conversion deterministically.
- [x] BudgetManager enforces warn vs stop semantics with immutable snapshots.
- [x] FlowRunner stops on hard breaches and records stop reasons in trace events.
- [x] PolicyStack integration emits policy_resolved and policy_violation events.
- [x] Tests cover simultaneous policy violation and soft budget warnings.

## Verification
- Unit suite: `pytest codex/code/07b_budget_guards_and_runner_integration.yaml/tests -q`
- Dynamic loader ensures namespace safety for task-scoped modules.

## Known Issues / Follow-ups
- No integration with global logging directories yet; emitter currently writes to provided sink only.
- Additional coverage tooling (e.g., coverage.py) can be layered later if CI requires explicit percentage reports.

