# SpO2 Modelling — TODO

## In Progress

- [ ] Project scaffold and core model (Phase 1)
- [ ] Collect session #2 data (same subject, same device, different day) for multi-session validation

## Planned

- [ ] Multi-session cross-validation (freeze session #1 globals, test on session #2)
- [ ] CSV upload and hold detection API
- [ ] Hold tagging and fit preview/save endpoints
- [ ] Model versioning and analysis endpoints
- [ ] React frontend with dark-themed Plotly charts
- [ ] Upload flow with hold detection and tagging UI
- [ ] Fit workflow with preview and approval
- [ ] Model management with version history and rollback
- [ ] Analysis tools (threshold, sensitivity, desaturation rate)
- [ ] Railway deployment

## Completed

_(newest first)_

- [x] v7 model shipped (v7.08) - 8/8 success criteria PASS, formal methods spec written (2026-03-13)
  - Two-stage sensor + physiology model for breath-hold apnea SpO2 prediction
  - k_co2 identified (interior minimum at 0.13), all global params interior
  - Formal spec: `backend/scripts/experiments/v7_methods_spec.txt`
  - Scope: single subject/session, descriptive model, awaiting session #2 for validation
- [x] Remove lag parameter — not needed, absorbed by tau_washout and ODC plateau (2026-02-25)
  - 8 → 7 fitted parameters, removes identifiability issue (lag vs tau_washout confounding)
  - DB migration: auto-drops `model_versions.lag` column on startup
  - Updated model equations, frontend charts/descriptions, tests
- [x] Replace Hill equation ODC with Severinghaus (1979) + gamma steepness exponent (2026-02-25)
  - Standalone comparison script: Hill vs Kelman vs Severinghaus (+gamma variants)
  - Severinghaus+gamma wins: R²=0.9956 vs Hill R²=0.9915 (same 8 params, 3/8 at bounds vs 6/8)
  - Backend: new `severinghaus_spo2()`, virtual PO2 Bohr shift, gamma power transform
  - Renamed parameter `n` (Hill coefficient) → `gamma` (steepness exponent)
  - Updated bounds, DB schema, API, frontend labels/charts/descriptions
  - DB migration: auto-renames `model_versions.n` → `gamma` on startup
