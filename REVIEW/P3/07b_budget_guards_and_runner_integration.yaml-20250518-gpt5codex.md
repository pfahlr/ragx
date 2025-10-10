# Phase 3 Review Notes – Budget guards and runner integration

## Checklist
- [x] Trace payloads use `MappingProxyType` to guarantee immutability before emission.
- [x] BudgetManager tracks scope state, raises `BudgetBreachError` for blocking outcomes, and persists history for inspection.
- [x] FlowRunner integrates adapters, policy enforcement, and budget guards while emitting lifecycle + violation traces.
- [x] Tests reside under `codex/code/integrate_budget_guards_runner_p3/tests/` satisfying branch segmentation requirement.
- [x] pytest suite (`pytest codex/code/integrate_budget_guards_runner_p3/tests -q`) passes locally.

## Notes
- Policy traces currently surface violations through the shared emitter; future work may map `policy_resolved` records explicitly if schema gating demands it.
- Trace sink attachment remains synchronous but is pluggable for downstream streaming once schema validators land.
