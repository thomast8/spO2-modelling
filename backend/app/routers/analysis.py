"""Analysis API endpoints for v7 models."""

from typing import Callable

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.db_models import ModelVersionV7
from app.models.schemas import (
    DesatRatePointResponse,
    SensitivityPointResponse,
    ThresholdResponse,
)
from app.services.v7_model import predict_spo2_v7

router = APIRouter(prefix="/analysis", tags=["analysis"])


def _format_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}:{s:02d}"


def _build_predict_fn(
    model: ModelVersionV7, hold_id: int
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a prediction function for a specific hold from a v7 model."""
    params = model.to_v7_params()

    def predict_fn(t: np.ndarray) -> np.ndarray:
        return predict_spo2_v7(t, hold_id, params)

    return predict_fn


@router.get("/threshold", response_model=ThresholdResponse)
async def threshold_analysis_endpoint(
    model_id: int = Query(..., description="Model version ID"),
    hold_id: int = Query(..., description="Hold ID for prediction"),
    threshold: float = Query(40.0, ge=0, le=100, description="SpO2 threshold (%)"),
    t_max: float = Query(800.0, ge=60, le=1800, description="Max simulation time (s)"),
    db: AsyncSession = Depends(get_db),
):
    """Find time at which SpO2 drops below a threshold for a specific hold."""
    model = await db.get(ModelVersionV7, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model version not found")

    try:
        predict_fn = _build_predict_fn(model, hold_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None

    t = np.arange(0, t_max, 0.5)
    spo2 = predict_fn(t)

    idx = np.where(spo2 <= threshold)[0]
    if len(idx) > 0:
        crossing = float(t[idx[0]])
        return ThresholdResponse(
            threshold=threshold,
            crossing_time_s=crossing,
            crossing_time_fmt=_format_time(crossing),
            spo2_at_end=None,
        )
    else:
        spo2_end = float(spo2[-1]) if len(spo2) > 0 else None
        return ThresholdResponse(
            threshold=threshold,
            crossing_time_s=None,
            crossing_time_fmt=None,
            spo2_at_end=spo2_end,
        )


@router.get("/sensitivity", response_model=list[SensitivityPointResponse])
async def sensitivity_analysis_endpoint(
    model_id: int = Query(..., description="Model version ID"),
    hold_id: int = Query(..., description="Hold ID for prediction"),
    param_name: str = Query("tau_washout", description="Parameter to vary"),
    reference_time_s: float = Query(372.0, ge=0, description="Reference time (s)"),
    threshold: float = Query(40.0, ge=0, le=100, description="SpO2 threshold (%)"),
    t_max: float = Query(800.0, ge=60, le=1800, description="Max simulation time (s)"),
    db: AsyncSession = Depends(get_db),
):
    """Parameter sensitivity analysis for a v7 model on a specific hold."""
    model = await db.get(ModelVersionV7, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model version not found")

    params = model.to_v7_params()

    # Find the parameter to vary
    stage_b_ph = next(
        (ph for ph in params.stage_b_per_hold if ph.hold_id == hold_id), None
    )
    if stage_b_ph is None:
        raise HTTPException(status_code=400, detail=f"No params for hold_id={hold_id}")

    # Map param names to (object, attr) for modification
    param_map = {
        "tau_washout": (stage_b_ph, "tau_washout"),
        "paco2_0": (stage_b_ph, "paco2_0"),
        "k_co2": (params.stage_b_global, "k_co2"),
        "gamma": (params.stage_b_global, "gamma"),
    }

    if param_name not in param_map:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown parameter: {param_name}. Valid: {list(param_map)}",
        )

    obj, attr = param_map[param_name]
    base_value = getattr(obj, attr)
    dt = 0.5
    t = np.arange(0, t_max, dt)
    results = []

    for pct in range(-15, 16, 5):
        test_value = base_value * (1 + pct / 100)
        original = getattr(obj, attr)
        setattr(obj, attr, test_value)

        spo2 = predict_spo2_v7(t, hold_id, params)

        ref_idx = int(reference_time_s / dt)
        spo2_at_ref = float(spo2[ref_idx]) if ref_idx < len(spo2) else 0.0

        cross_idx = np.where(spo2 <= threshold)[0]
        crossing = float(t[cross_idx[0]]) if len(cross_idx) > 0 else None
        margin = crossing - reference_time_s if crossing else None

        results.append(SensitivityPointResponse(
            param_value=test_value,
            pct_change=float(pct),
            crossing_time_s=crossing,
            margin_s=margin,
            spo2_at_ref=spo2_at_ref,
        ))

        setattr(obj, attr, original)

    return results


@router.get("/desat-rate", response_model=list[DesatRatePointResponse])
async def desat_rate_analysis(
    model_id: int = Query(..., description="Model version ID"),
    hold_id: int = Query(..., description="Hold ID for prediction"),
    time_points: str = Query(
        "60,120,180,240,300",
        description="Comma-separated time points in seconds",
    ),
    t_max: float = Query(800.0, ge=60, le=1800, description="Max simulation time (s)"),
    db: AsyncSession = Depends(get_db),
):
    """Compute desaturation rate at specified time points for a specific hold."""
    model = await db.get(ModelVersionV7, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model version not found")

    try:
        predict_fn = _build_predict_fn(model, hold_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None

    try:
        tp_list = [float(t.strip()) for t in time_points.split(",")]
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="time_points must be comma-separated numbers",
        ) from None

    dt = 0.5
    t = np.arange(0, t_max, dt)
    spo2 = predict_fn(t)
    gradient = np.gradient(spo2, dt) * 60  # %/min

    results = []
    for tp in tp_list:
        idx = int(tp / dt)
        if idx < len(gradient):
            results.append(DesatRatePointResponse(
                time_s=tp,
                rate_per_min=float(gradient[idx]),
                spo2=float(spo2[idx]),
            ))

    return results
