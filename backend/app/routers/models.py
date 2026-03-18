"""v7 model versioning API endpoints."""

import json

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.models.db_models import Hold, ModelVersionV7
from app.models.schemas import (
    V7FitPrediction,
    V7ModelVersionListResponse,
    V7ModelVersionResponse,
    V7PredictionCurveResponse,
)
from app.routers.fit import _model_to_response
from app.services.model_manager import (
    activate_model_v7,
    get_active_model_v7,
    get_model_versions_v7,
)
from app.services.v7_model import (
    compute_r2,
    predict_spo2_v7,
    predict_spo2_v7_components,
)

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=V7ModelVersionListResponse)
async def list_all_models(db: AsyncSession = Depends(get_db)):
    """Get all v7 model versions."""
    versions = await get_model_versions_v7(db)
    active = next((v.version for v in versions if v.is_active), None)
    return V7ModelVersionListResponse(
        versions=[_model_to_response(v) for v in versions],
        active_version=active,
    )


@router.get("/active", response_model=V7ModelVersionResponse)
async def get_active(db: AsyncSession = Depends(get_db)):
    """Get the currently active v7 model."""
    model = await get_active_model_v7(db)
    if not model:
        raise HTTPException(status_code=404, detail="No active v7 model")
    return _model_to_response(model)


@router.post("/{model_id}/activate", response_model=V7ModelVersionResponse)
async def activate(
    model_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Set a v7 model version as active."""
    try:
        model = await activate_model_v7(db, model_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from None
    return _model_to_response(model)


@router.get("/{model_id}/predict", response_model=V7PredictionCurveResponse)
async def predict_curve(
    model_id: int,
    hold_id: int = Query(..., description="Hold ID for prediction"),
    t_max: float = Query(600.0, ge=10, le=1200, description="Max time in seconds"),
    dt: float = Query(1.0, ge=0.1, le=10, description="Time step in seconds"),
    db: AsyncSession = Depends(get_db),
):
    """Generate a prediction curve for a specific hold from a saved v7 model."""
    model = await db.get(ModelVersionV7, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model version not found")

    params = model.to_v7_params()
    t = np.arange(0, t_max, dt)

    try:
        components = predict_spo2_v7_components(t, hold_id, params)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None

    return V7PredictionCurveResponse(
        hold_id=hold_id,
        t=t.tolist(),
        spo2=components["spo2"].tolist(),
        pao2=components["pao2"].tolist(),
        p50_eff=components["p50_eff"].tolist(),
        sao2_raw=components["sao2_raw"].tolist(),
    )


@router.get("/{model_id}/hold-predictions", response_model=list[V7FitPrediction])
async def get_hold_predictions(
    model_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Regenerate per-hold predictions from a saved v7 model's parameters."""
    model = await db.get(ModelVersionV7, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model version not found")

    hold_ids = model.get_hold_ids()
    params = model.to_v7_params()

    result = await db.execute(
        select(Hold).where(Hold.id.in_(hold_ids)).options(selectinload(Hold.data_points))
    )
    holds = result.scalars().all()

    predictions = []
    for hold in holds:
        elapsed_s = np.array([dp.elapsed_s for dp in hold.data_points], dtype=float)
        observed = np.array([dp.spo2 for dp in hold.data_points], dtype=float)

        try:
            predicted = predict_spo2_v7(elapsed_s, hold.id, params)
        except ValueError:
            continue

        r2 = compute_r2(observed, predicted)
        predictions.append(
            V7FitPrediction(
                hold_id=hold.id,
                elapsed_s=elapsed_s.tolist(),
                observed=observed.tolist(),
                predicted=predicted.tolist(),
                r_squared=float(r2),
            )
        )

    return predictions
