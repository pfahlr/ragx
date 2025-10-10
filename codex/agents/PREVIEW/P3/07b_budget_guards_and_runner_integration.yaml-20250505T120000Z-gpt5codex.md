# Phase 3 Preview — Budget Guards & Runner Integration

## Scope
- Canonicalised budget domain objects (`BudgetSpec`, `CostSnapshot`, `BudgetCharge`, `BudgetPreview`).
- Centralised cost normalisation for seconds→milliseconds conversion.
- Implemented `BudgetManager` with preview/commit lifecycle and hierarchical scope tracking.
- Added `TraceEventEmitter` to emit immutable budget/policy/loop-stop traces.
- Wired `FlowRunner` to adapters, `PolicyStack`, budget manager, and emitter with preflight checks.

## Goals for Implementation
1. Enforce consistent budget semantics across run/node scopes with warn vs stop modes.
2. Produce structured `budget_charge`, `budget_breach`, `policy_resolved`, `policy_violation`, and `loop_stop` events.
3. Guard execution paths via TDD: cost utilities → manager → runner integration.

## Key Decisions
- Use dynamic loader wrappers so feature modules live under task-specific path without altering global packages.
- Emit one `budget_breach` event per breached scope with explicit `breach_action` metadata.
- Maintain deterministic order for scope teardown to simplify fixture cleanup.

## Risk & Mitigations
- **Dynamic import quirks** → Provide helper loader for tests and package entry point.
- **Trace schema drift** → Centralised emitter ensures uniform payload shapes.
- **Double charging** → Preview returns immutable snapshot reused by commit; state updates only during commit.

