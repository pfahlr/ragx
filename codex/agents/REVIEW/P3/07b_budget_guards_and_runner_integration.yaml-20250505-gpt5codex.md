# Phase 3 Review Notes – Budget guards and runner integration

## Checklist
- [x] Cost normalisation handles seconds → milliseconds and preserves tokens.
- [x] BudgetManager distinguishes warn vs stop and logs breaches before raising.
- [x] FlowRunner enforces PolicyStack prior to adapter execution and releases scopes on errors.
- [x] Trace payloads flow through `TraceEventEmitter` and remain immutable mapping proxies.
- [x] Tests cover hard stop, soft warn, scope exit validation, and integration pathways.

## Reviewer Guidance
- Verify history snapshots allow inspection after scope exit.
- Confirm FlowRunner surfaces `BudgetBreachError` with node scope metadata for hard stops.
- Ensure adapters without specs still execute (decisions with zero outcomes) without errors.
