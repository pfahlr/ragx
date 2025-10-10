# Phase 3 Review Notes – 07b_budget_guards_and_runner_integration

## Checklist
- [x] Immutable budget models avoid shared mutation; dataclasses frozen and mapping proxies applied.
- [x] BudgetManager enforces breach_action semantics by halting on hard-stop preflight breaches and emitting warnings for soft budgets.
- [x] FlowRunner emits `policy_resolved` prior to budget events via `PolicyStack.effective_allowlist` calls.
- [x] Loop stop reasons recorded as `budget_exhausted` with iteration counts through `TraceEventEmitter.emit_loop_stop`.
- [x] Tests span hard-stop runs, soft warnings, and loop exhaustion across manager and runner suites.

## Notes
- Tokens/calls defaults treated as unlimited to match DSL expectations when limits omitted.
- Fake adapters provide deterministic estimate/execute separation enabling reliable budget projections.
- Further coverage opportunity: simultaneous policy violation + budget breach interplay (deferred to missing tests list).
