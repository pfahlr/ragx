# Phase 3 Review Checklist — Budget Guards & Runner Integration

- [x] BudgetMode and breach_action semantics reflect hard-stop vs soft-warn behavior.
- [x] Cost normalization converts `*_s` metrics to milliseconds exactly once and all payloads are immutable.
- [x] Trace emission flows through `TraceEventEmitter` using mapping proxies with deterministic ordering.
- [x] FlowRunner enforces PolicyStack decisions before adapter execution and surfaces stop reasons from budgets.
- [x] Loop summaries collect iterations executed, aggregated spend, and propagate warnings via traces.
- [x] Unit tests cover manager arithmetic, trace immutability, policy-budget interplay, and loop warnings.
