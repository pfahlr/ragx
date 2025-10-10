# Phase 3 Preview — Budget guards and runner integration

## Overview
- Built sandbox modules under `codex/code/phase3-budget-runner-4f72/` that compose
  the adapter-driven runner from `codex/implement-budget-guards-with-test-first-approach-fa0vm9`
  with immutable budget models and tracing patterns credited to
  `codex/integrate-budget-guards-with-runner-zwi2ny` and
  `codex/integrate-budget-guards-with-runner-pbdel9`.
- Added a canonical cost normaliser to eliminate seconds→milliseconds drift,
  honouring arithmetic decisions highlighted in branch `8wxk32`.
- Layered PolicyStack hooks (from task 07a decisions) into the FlowRunner so
  `policy_resolved` traces always accompany successful adapter execution while
  budget warnings/stops emit shared trace payloads.

## Test Strategy
- Unit tests validate cost normalisation immutability, BudgetManager soft/hard
  semantics, and trace emissions (charges + breaches).
- Integration tests exercise FlowRunner control flow for
  * soft budgets continuing execution with warnings,
  * hard budgets halting before execution of the second node, and
  * policy violations short-circuiting prior to budget checks.

## Pending Risks / Questions
- Current sandbox treats adapter execution costs as matching estimates; future
  phases should decide how to reconcile estimate vs actual before committing.
- Loop summaries are minimal; richer aggregation (per-iteration metrics) may be
  required once we integrate with real flows.
