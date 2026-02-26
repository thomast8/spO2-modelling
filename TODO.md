# SpO2 Modelling — TODO

## In Progress

- [ ] Project scaffold and core model (Phase 1)

## Planned

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

- [x] Replace saturating Bohr with CO2-driven Bohr effect (2026-02-26)
  - Ablation study: 10 model variants tested, CO2 Bohr wins (4.5% lower RMSE, same 7 params)
  - `bohr_max` + `tau_bohr` → `paco2_0` + `k_co2` with fixed Bohr coefficient (β=0.48)
  - Handles hypocapnic starts (post-hyperventilation), better late-phase residuals
  - Widened gamma bounds (2.0 → 3.0), updated all bounds per hold type
  - DB migration: auto-drops `bohr_max`/`tau_bohr`, adds `paco2_0`/`k_co2` on startup
  - Updated model equations, frontend charts/descriptions, tests
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
