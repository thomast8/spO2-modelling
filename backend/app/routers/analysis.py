"""Analysis API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.db_models import ModelVersion
from app.models.schemas import (
    DesatRatePointResponse,
    SensitivityPointResponse,
    ThresholdResponse,
)
from app.services.analysis import desaturation_rate, find_threshold_time, sensitivity_vo2

router = APIRouter(prefix="/analysis", tags=["analysis"])


async def _get_model_params(db: AsyncSession, model_id: int):
    """Load model and return HillParams, raising 404 if not found."""
    model = await db.get(ModelVersion, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model version not found")
    return model.to_hill_params()


@router.get("/threshold", response_model=ThresholdResponse)
async def threshold_analysis(
    model_id: int = Query(..., description="Model version ID"),
    threshold: float = Query(40.0, ge=0, le=100, description="SpO2 threshold (%)"),
    t_max: float = Query(800.0, ge=60, le=1800, description="Max simulation time (s)"),
    db: AsyncSession = Depends(get_db),
):
    """Find time at which SpO2 drops below a threshold."""
    params = await _get_model_params(db, model_id)
    result = find_threshold_time(params, threshold=threshold, t_max=t_max)
    return ThresholdResponse(
        threshold=result.threshold,
        crossing_time_s=result.crossing_time_s,
        crossing_time_fmt=result.crossing_time_fmt,
        spo2_at_end=result.spo2_at_end,
    )


@router.get("/sensitivity", response_model=list[SensitivityPointResponse])
async def sensitivity_analysis(
    model_id: int = Query(..., description="Model version ID"),
    reference_time_s: float = Query(372.0, ge=0, description="Reference time (s)"),
    threshold: float = Query(40.0, ge=0, le=100, description="SpO2 threshold (%)"),
    t_max: float = Query(800.0, ge=60, le=1800, description="Max simulation time (s)"),
    db: AsyncSession = Depends(get_db),
):
    """VO2 sensitivity analysis: how crossing time changes with VO2 variation."""
    params = await _get_model_params(db, model_id)
    results = sensitivity_vo2(
        params,
        reference_time_s=reference_time_s,
        threshold=threshold,
        t_max=t_max,
    )
    return [
        SensitivityPointResponse(
            vo2=r.vo2,
            pct_change=r.pct_change,
            crossing_time_s=r.crossing_time_s,
            margin_s=r.margin_s,
            spo2_at_ref=r.spo2_at_ref,
        )
        for r in results
    ]


@router.get("/desat-rate", response_model=list[DesatRatePointResponse])
async def desat_rate_analysis(
    model_id: int = Query(..., description="Model version ID"),
    time_points: str = Query(
        "60,120,180,240,300",
        description="Comma-separated time points in seconds",
    ),
    t_max: float = Query(800.0, ge=60, le=1800, description="Max simulation time (s)"),
    db: AsyncSession = Depends(get_db),
):
    """Compute desaturation rate at specified time points."""
    params = await _get_model_params(db, model_id)

    try:
        tp_list = [float(t.strip()) for t in time_points.split(",")]
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="time_points must be comma-separated numbers",
        ) from None

    results = desaturation_rate(params, time_points=tp_list, t_max=t_max)
    return [
        DesatRatePointResponse(
            time_s=r.time_s,
            rate_per_min=r.rate_per_min,
            spo2=r.spo2,
        )
        for r in results
    ]
