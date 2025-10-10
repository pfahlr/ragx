# Phase 3 Preview — 07b_budget_guards_and_runner_integration

## Overview
- Stage sandbox FlowRunner integrating immutable budget models, a shared trace emitter, and policy stack plumbing.
- Preserve adapter-driven execution with deterministic cost normalization and loop stop semantics.
- Emit policy and budget traces through a single emitter to validate ordering and payload immutability.

## Key Objectives
1. Finalize `BudgetManager` with scope-aware preflight/commit lifecycle.
2. Ensure `TraceEventEmitter` bridges policy and budget events for observability.
3. Execute loops through FlowRunner, enforcing breach actions and summarizing stop reasons.

## Anticipated Risks
- Divergent breach semantics between scopes.
- Maintaining event ordering amidst early loop exits.
- Keeping dataclasses frozen while exposing useful diagnostics.
