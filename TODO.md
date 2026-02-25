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

- [x] Replace Hill equation ODC with Severinghaus (1979) + gamma steepness exponent (2026-02-25)
  - Standalone comparison script: Hill vs Kelman vs Severinghaus (+gamma variants)
  - Severinghaus+gamma wins: R²=0.9956 vs Hill R²=0.9915 (same 8 params, 3/8 at bounds vs 6/8)
  - Backend: new `severinghaus_spo2()`, virtual PO2 Bohr shift, gamma power transform
  - Renamed parameter `n` (Hill coefficient) → `gamma` (steepness exponent)
  - Updated bounds, DB schema, API, frontend labels/charts/descriptions
  - DB migration: auto-renames `model_versions.n` → `gamma` on startup
