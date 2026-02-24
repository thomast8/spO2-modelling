"""Model fitting API endpoints."""

import json

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.models.db_models import Hold
from app.models.schemas import (
    FitPreviewRequest,
    FitPreviewResponse,
    FitSaveRequest,
    HillParamsResponse,
    ModelVersionResponse,
)
from app.services.fitter import fit_holds
from app.services.hill_model import HillParams
from app.services.model_manager import save_model_version

router = APIRouter(prefix="/fit", tags=["fit"])


@router.post("/preview", response_model=FitPreviewResponse)
async def preview_fit(
    request: FitPreviewRequest,
    db: AsyncSession = Depends(get_db),
):
    """Run a model fit without saving. Returns preview data for approval."""
    # Load hold data
    result = await db.execute(
        select(Hold)
        .where(Hold.id.in_(request.hold_ids))
        .options(selectinload(Hold.data_points))
    )
    holds = result.scalars().all()

    if not holds:
        raise HTTPException(status_code=404, detail="No holds found with the given IDs")

    if len(holds) != len(request.hold_ids):
        found_ids = {h.id for h in holds}
        missing = set(request.hold_ids) - found_ids
        raise HTTPException(status_code=404, detail=f"Holds not found: {missing}")

    # Prepare data for fitting
    hold_data_list = []
    for hold in holds:
        if not hold.data_points:
            raise HTTPException(
                status_code=400,
                detail=f"Hold {hold.id} has no data points",
            )
        hold_data_list.append({
            "id": hold.id,
            "elapsed_s": np.array([dp.elapsed_s for dp in hold.data_points], dtype=float),
            "spo2": np.array([dp.spo2 for dp in hold.data_points], dtype=float),
        })

    # Convert bounds override
    bounds_override = None
    if request.bounds_override:
        bounds_override = {
            name: (b.lower, b.upper) for name, b in request.bounds_override.items()
        }

    # Run fit
    logger.info(f"Running {request.hold_type} fit preview on {len(holds)} holds")
    try:
        fit_result = fit_holds(
            hold_data=hold_data_list,
            hold_type=request.hold_type,
            bounds_override=bounds_override,
            seed=request.seed,
        )
    except Exception as e:
        logger.error(f"Fit failed: {e}")
        raise HTTPException(status_code=500, detail=f"Fitting failed: {e}") from None

    return FitPreviewResponse(
        params=HillParamsResponse(**fit_result.params.to_dict()),
        r_squared=fit_result.r_squared,
        r_squared_per_hold=fit_result.r_squared_per_hold,
        objective_val=fit_result.objective_val,
        converged=fit_result.converged,
        n_holds=fit_result.n_holds,
        n_data_points=fit_result.n_data_points,
        predictions=fit_result.predictions,
    )


@router.post("/save", response_model=ModelVersionResponse)
async def save_fit(
    request: FitSaveRequest,
    db: AsyncSession = Depends(get_db),
):
    """Save a previewed fit as a new model version."""
    params = HillParams(
        o2_start=request.params.o2_start,
        vo2=request.params.vo2,
        scale=request.params.scale,
        p50=request.params.p50,
        n=request.params.n,
        r_offset=request.params.r_offset,
        r_decay=request.params.r_decay,
        tau_decay=request.params.tau_decay,
        lag=request.params.lag,
    )

    model = await save_model_version(
        db=db,
        hold_type=request.hold_type,
        params=params,
        r_squared=request.r_squared,
        objective_val=request.objective_val,
        converged=request.converged,
        hold_ids=request.hold_ids,
        notes=request.notes,
        set_active=request.set_active,
    )

    return ModelVersionResponse(
        id=model.id,
        hold_type=model.hold_type,
        version=model.version,
        is_active=model.is_active,
        params=HillParamsResponse(**params.to_dict()),
        r_squared=model.r_squared,
        objective_val=model.objective_val,
        converged=model.converged,
        n_holds_used=model.n_holds_used,
        hold_ids=json.loads(model.hold_ids_json),
        notes=model.notes,
        created_at=model.created_at,
    )
