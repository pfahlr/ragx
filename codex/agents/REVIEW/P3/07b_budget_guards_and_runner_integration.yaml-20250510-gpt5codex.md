# Review Notes — 07b_budget_guards_and_runner_integration

## Checklist
- [x] Cost normalization converts seconds→milliseconds and preserves unknown metrics.
- [x] BudgetManager preview/commit enforces warn/stop/error semantics with immutable diagnostics.
- [x] FlowRunner calls adapters for every executed iteration and halts gracefully on stop/error outcomes.
- [x] TraceEventEmitter emits `budget_charge`, `budget_breach`, and `budget_commit` with mapping-proxy payloads.
- [x] Tests cover warn/stop/error, loop halting, multi-scope prioritisation, and trace sequencing.

## Reviewer Notes
- Policy stack hooks are minimally exercised; future work should assert policy trace outputs once available.
- Current FlowRunner result payload aggregates node outputs but omits trace sink configuration—acceptable for Phase 3 staging.
- Missing metrics (e.g., `cost_ms_cpu`) will require extending `normalize_cost` plus fixtures when spec expands.
