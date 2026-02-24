"""Model versioning API endpoints."""

import json

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.db_models import ModelVersion
from app.models.schemas import (
    AllModelsResponse,
    ApneaModelParamsResponse,
    ModelVersionListResponse,
    ModelVersionResponse,
    PredictionCurveResponse,
)
from app.services.analysis import generate_prediction_curve
from app.services.model_manager import activate_model, get_active_model, get_model_versions

router = APIRouter(prefix="/models", tags=["models"])

VALID_HOLD_TYPES = ("FRC", "RV", "FL")


def _model_to_response(model: ModelVersion) -> ModelVersionResponse:
    """Convert ORM ModelVersion to response schema."""
    params = model.to_model_params()
    return ModelVersionResponse(
        id=model.id,
        hold_type=model.hold_type,
        version=model.version,
        is_active=model.is_active,
        params=ApneaModelParamsResponse(**params.to_dict()),
        r_squared=model.r_squared,
        objective_val=model.objective_val,
        converged=model.converged,
        n_holds_used=model.n_holds_used,
        hold_ids=json.loads(model.hold_ids_json),
        notes=model.notes,
        created_at=model.created_at,
    )


@router.get("", response_model=AllModelsResponse)
async def list_all_models(db: AsyncSession = Depends(get_db)):
    """Get all model versions grouped by hold type."""
    result = {}
    for ht in VALID_HOLD_TYPES:
        versions = await get_model_versions(db, ht)
        active = next((v.version for v in versions if v.is_active), None)
        result[ht] = ModelVersionListResponse(
            hold_type=ht,
            versions=[_model_to_response(v) for v in versions],
            active_version=active,
        )
    return AllModelsResponse(**result)


@router.get("/{hold_type}", response_model=ModelVersionListResponse)
async def list_models_for_type(
    hold_type: str,
    db: AsyncSession = Depends(get_db),
):
    """Get all versions for a specific hold type."""
    if hold_type not in VALID_HOLD_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid hold_type: {hold_type}. Must be FRC, RV, or FL",
        )

    versions = await get_model_versions(db, hold_type)
    active = next((v.version for v in versions if v.is_active), None)
    return ModelVersionListResponse(
        hold_type=hold_type,
        versions=[_model_to_response(v) for v in versions],
        active_version=active,
    )


@router.get("/{hold_type}/active", response_model=ModelVersionResponse)
async def get_active(
    hold_type: str,
    db: AsyncSession = Depends(get_db),
):
    """Get the currently active model version for a hold type."""
    if hold_type not in VALID_HOLD_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid hold_type: {hold_type}. Must be FRC, RV, or FL",
        )

    model = await get_active_model(db, hold_type)
    if not model:
        raise HTTPException(
            status_code=404,
            detail=f"No active model for {hold_type}",
        )
    return _model_to_response(model)


@router.post("/{model_id}/activate", response_model=ModelVersionResponse)
async def activate(
    model_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Set a model version as active (deactivates previous active version)."""
    try:
        model = await activate_model(db, model_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from None
    return _model_to_response(model)


@router.get("/{model_id}/predict", response_model=PredictionCurveResponse)
async def predict_curve(
    model_id: int,
    t_max: float = Query(600.0, ge=10, le=1200, description="Max time in seconds"),
    dt: float = Query(1.0, ge=0.1, le=10, description="Time step in seconds"),
    db: AsyncSession = Depends(get_db),
):
    """Generate a prediction curve from a saved model."""
    model = await db.get(ModelVersion, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model version not found")

    params = model.to_model_params()
    curve = generate_prediction_curve(params, t_max=t_max, dt=dt)

    return PredictionCurveResponse(
        t=curve["t"],
        spo2=curve["spo2"],
        spo2_base=curve["spo2_base"],
        pao2=curve["pao2"],
        p50_eff=curve["p50_eff"],
    )
