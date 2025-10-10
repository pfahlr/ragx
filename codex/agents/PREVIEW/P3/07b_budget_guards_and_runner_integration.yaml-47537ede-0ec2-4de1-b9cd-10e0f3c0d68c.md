# Phase 3 Preview — Task 07b Budget Guards and Runner Integration

## Summary
- Build immutable budget models plus a BudgetManager handling preflight/commit with scope registration.
- Implement a TraceEventEmitter that emits immutable policy and budget events through a TraceWriter.
- Retrofit a FlowRunner that orchestrates adapters, budgets, and policy stack interactions with deterministic trace ordering.

## Key Considerations
- Maintain millisecond-normalised `CostSnapshot` arithmetic to prevent drift between estimate and commit phases.
- Distinguish soft-warn and hard-stop semantics via `BudgetMode` and `BreachAction` without reintroducing bespoke exception hierarchies.
- Ensure policy push → resolve → pop ordering is traceable even when budgets halt execution mid-flow.

## Test Strategy
- Unit tests for `BudgetManager` covering soft/hard breaches and scope lifecycle.
- Trace emitter tests validating immutable payloads and correct event routing.
- FlowRunner integration tests simulating adapter execution, warnings, and hard stops while asserting trace emissions.
