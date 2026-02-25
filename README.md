# SpO2 Desaturation Modelling

A webapp for modelling SpO2 desaturation during breath-hold apnea sessions using the Severinghaus (1979) oxygen-haemoglobin dissociation curve.

## Features

- **CSV Upload**: Import pulse oximeter session data, auto-detect apnea holds
- **Hold Tagging**: Classify holds as FRC (Functional Residual Capacity), RV (Residual Volume), or FL (Full Lungs)
- **Model Fitting**: Fit Severinghaus ODC parameters per hold type with interactive preview
- **Model Versioning**: Track fit versions with rollback capability
- **Analysis Tools**: Threshold prediction, VO2 sensitivity analysis, desaturation rate calculation
- **Dark Theme**: Polished data visualization with Plotly charts

## Tech Stack

- **Backend**: FastAPI + SQLAlchemy (async) + SQLite
- **Frontend**: React + TypeScript + MUI + Plotly
- **Model**: Severinghaus ODC with gamma steepness exponent, fitted via differential evolution

## Development

### Backend

```bash
cd backend
uv sync
uv run uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Running Tests

```bash
cd backend
uv run pytest
```

## Deployment (Railway)

1. Create a Railway project
2. Attach a persistent volume at `/app/data`
3. Deploy from this repo (auto-detects Dockerfile)

## Model

The model uses exponential alveolar O2 washout, a saturating Bohr effect, and the Severinghaus (1979) ODC with adjustable steepness:

```
t_eff        = max(t - lag, 0)
PAO2(t)      = pvo2 + (pao2_0 - pvo2) * exp(-t_eff / tau_washout)
P50_eff(t)   = P50_BASE + bohr_max * (1 - exp(-t_eff / tau_bohr))
PAO2_virtual = PAO2 * (P50_BASE / P50_eff)          [Bohr shift]
PAO2_adj     = P50_BASE * (PAO2_virtual / P50_BASE)^gamma  [steepness]
SpO2(t)      = r_offset + 100 / (1 + 23400/(PAO2_adj^3 + 150*PAO2_adj))
```

Parameters are fitted per hold type (FRC/RV/FL) with different bounds reflecting the physiological differences in lung volume.
