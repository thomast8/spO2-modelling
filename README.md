# SpO2 Desaturation Modelling

A webapp for modelling SpO2 desaturation during breath-hold apnea sessions using the Hill equation oxygen-haemoglobin dissociation curve.

## Features

- **CSV Upload**: Import pulse oximeter session data, auto-detect apnea holds
- **Hold Tagging**: Classify holds as FRC (Functional Residual Capacity), RV (Residual Volume), or FL (Full Lungs)
- **Model Fitting**: Fit Hill equation parameters per hold type with interactive preview
- **Model Versioning**: Track fit versions with rollback capability
- **Analysis Tools**: Threshold prediction, VO2 sensitivity analysis, desaturation rate calculation
- **Dark Theme**: Polished data visualization with Plotly charts

## Tech Stack

- **Backend**: FastAPI + SQLAlchemy (async) + SQLite
- **Frontend**: React + TypeScript + MUI + Plotly
- **Model**: Hill equation ODC with residual correction, fitted via differential evolution

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

The Hill equation models O2-haemoglobin dissociation:

```
O2(t)     = O2_start - (VO2 / 60) * max(t - lag, 0)
PaO2_eff  = O2(t) / scale
SpO2_base = 100 * PaO2_eff^n / (PaO2_eff^n + P50^n)
SpO2(t)   = SpO2_base + r_offset + r_decay * exp(-t / tau_decay)
```

Parameters are fitted per hold type (FRC/RV/FL) with different bounds reflecting the physiological differences in lung volume.
