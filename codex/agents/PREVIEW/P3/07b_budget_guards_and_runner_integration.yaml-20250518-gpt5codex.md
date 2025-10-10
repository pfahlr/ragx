# Phase 3 Preview – Budget guards and runner integration

## Scope
- Stand up canonical trace emitter and budget value objects under `codex/code/phase3_budget_runner/dsl`.
- Provide scope-aware `BudgetManager` with preview/commit/record flows feeding immutable traces.
- Integrate FlowRunner with adapters, policies, and budgets while emitting schema-aligned events.

## Test Strategy
- `test_budget_models.py`: Cost normalisation, arithmetic clamping, and trace payload immutability.
- `test_budget_manager.py`: Scope lifecycle, warn vs stop semantics, and breach/charge emission ordering.
- `test_flow_runner_budget_integration.py`: Adapter-driven execution with BudgetBreachError propagation and policy violation tracing.

## Risks & Mitigations
- **Trace schema drift** → Centralise emission via `TraceEventEmitter` with mapping proxies.
- **Policy/Budget ordering confusion** → Tests assert policy violation traces coexist with budget stop flow.
- **Path isolation** → Branch-specific tests live under `codex/code/phase3_budget_runner/tests` for downstream automation.
