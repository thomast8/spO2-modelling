# SpO2 Modelling — TODO

## In Progress

- [ ] Project scaffold and core model (Phase 1)
- [ ] Collect session #2 data (same subject, same device, different day) for multi-session validation
  - Sessions now exist (Nadir, September 2026) but on a different oximeter; see v8.00 below

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

- [x] v8.01: what the two oximeters do to a hold, without ground truth (2026-10-17)
  - Plateau-and-step inside holds is the same on Masimo and PM100; PM100 recovery onset ~10 s later within RV; sensor low-pass under 8 s
  - Transit vs device delay not separable without a dual-oximeter session: `backend/scripts/experiments/exp_v8_01/`
- [x] v8.00 baseline: v7.08 refitted per session on the Nadir recordings (2026-10-17)
  - Ten sessions, two oximeters: tau_0 6.6-39.1 s, k_co2 0.02-0.15 mmHg/s, gamma 1.16-2.51 across sessions
  - Freezing all four globals at February's values costs under 0.025 R2a on the app-recorded sessions with real desaturation; the shallow 8 August and 8 September sessions lose 0.059 and 0.018. The per-hold parameters do the work
  - These are the 17 October numbers from the old loader; the loader fixes of 28 October need a full rerun (7 of 10 free fits done, transfers not yet)
  - Loader for Nadir exports + STAmina CSVs, transfer test, drift table and data request: `backend/scripts/experiments/exp_v8_00/`
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
