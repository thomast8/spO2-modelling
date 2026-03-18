"""v7 model fitting API endpoints with background task support."""

import json

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.models.db_models import Hold, ModelVersionV7, Session
from app.models.schemas import (
    TaskStatusResponse,
    V7FitPrediction,
    V7FitPreviewRequest,
    V7FitPreviewResponse,
    V7FitSaveRequest,
    V7ModelVersionResponse,
    V7StageAParamsResponse,
    V7StageAPerHoldResponse,
    V7StageBParamsResponse,
    V7StageBPerHoldResponse,
)
from app.services.recovery_data import extract_holds_with_recovery
from app.services.task_runner import TaskStatus, get_task, submit_task
from app.services.v7_fitter import fit_v7
from app.services.v7_model import V7FitResult

router = APIRouter(prefix="/fit", tags=["fit"])


def _fit_result_to_response(fit_result: V7FitResult) -> V7FitPreviewResponse:
    """Convert V7FitResult to API response."""
    params = fit_result.params
    sa = fit_result.stage_a_result
    sb = fit_result.stage_b_result

    stage_a = V7StageAParamsResponse(
        tau_0=params.stage_a_global.tau_0,
        p=params.stage_a_global.p,
        per_hold=[
            V7StageAPerHoldResponse(
                hold_id=ph.hold_id,
                hold_type=ph.hold_type,
                delta=ph.delta,
                s_min=ph.s_min,
                v_up=ph.v_up,
                eff_lag=ph.eff_lag,
                r_squared=sa.per_hold_r2.get(ph.hold_id, 0.0),
            )
            for ph in params.stage_a_per_hold
        ],
        converged=sa.converged,
        objective_val=sa.objective_val,
    )

    stage_b = V7StageBParamsResponse(
        k_co2=params.stage_b_global.k_co2,
        gamma=params.stage_b_global.gamma,
        per_hold=[
            V7StageBPerHoldResponse(
                hold_id=ph.hold_id,
                hold_type=ph.hold_type,
                tau_washout=ph.tau_washout,
                paco2_0=ph.paco2_0,
                r_squared=sb.per_hold_r2.get(ph.hold_id, 0.0),
            )
            for ph in params.stage_b_per_hold
        ],
        converged=sb.converged,
        objective_val=sb.objective_val,
    )

    predictions = []
    for hold_id, pred_data in sb.per_hold_predictions.items():
        predictions.append(V7FitPrediction(
            hold_id=hold_id,
            elapsed_s=pred_data["elapsed_s"],
            observed=pred_data["observed"],
            predicted=pred_data["predicted"],
            r_squared=sb.per_hold_r2.get(hold_id, 0.0),
        ))

    return V7FitPreviewResponse(
        stage_a=stage_a,
        stage_b=stage_b,
        predictions=predictions,
        overall_r2=fit_result.overall_r2,
        hold_ids=fit_result.hold_ids,
    )


def _run_fit_sync(
    hold_data_list: list[dict],
    s_base_values: dict[int, float],
    seed: int,
) -> dict:
    """Synchronous fit wrapper for background execution.

    Returns a serializable dict (V7FitResult can't be directly JSON-serialized
    because it contains numpy arrays in frozen_sensor).
    """
    result = fit_v7(hold_data_list, s_base_values, seed)
    # Store the full result for save, plus the response for display
    return {
        "fit_result": result,
        "response": _fit_result_to_response(result),
    }


@router.post("/preview")
async def preview_fit(
    request: V7FitPreviewRequest,
    db: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    """Start a v7 model fit as a background task. Returns {task_id} for polling."""
    # Load holds
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

    # Filter out excluded holds
    active_holds = [h for h in holds if h.id not in set(request.excluded_ids)]
    if not active_holds:
        raise HTTPException(status_code=400, detail="All holds are excluded")

    # Get CSV blob for recovery data
    session_ids = {h.session_id for h in active_holds}
    if len(session_ids) != 1:
        raise HTTPException(status_code=400, detail="All holds must belong to the same session")

    session_id = session_ids.pop()
    session = await db.get(Session, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Prepare hold data
    hold_data_list = []
    for hold in active_holds:
        if not hold.data_points:
            raise HTTPException(status_code=400, detail=f"Hold {hold.id} has no data points")
        hold_data_list.append({
            "id": hold.id,
            "type": hold.hold_type,
            "t": np.array([dp.elapsed_s for dp in hold.data_points], dtype=float),
            "spo2": np.array([dp.spo2 for dp in hold.data_points], dtype=float),
            "hr": np.array([dp.hr for dp in hold.data_points], dtype=float),
        })

    # Extract recovery data from CSV
    enriched_holds = extract_holds_with_recovery(session.csv_blob, hold_data_list)

    # Compute s_base (baseline SpO2) for each hold
    s_base_values = {}
    for h in enriched_holds:
        early = h["spo2"][h["t"] <= 5] if len(h["t"]) > 0 else h["spo2"][:5]
        s_base_values[h["id"]] = float(np.mean(early)) if len(early) > 0 else 98.0

    logger.info(f"Submitting v7 fit for {len(enriched_holds)} holds")
    task_id = await submit_task(_run_fit_sync, enriched_holds, s_base_values, request.seed)

    return {"task_id": task_id}


@router.get("/task/{task_id}", response_model=TaskStatusResponse)
async def poll_fit_task(task_id: str):
    """Poll a background fit task for status and result."""
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    result = None
    if task.status == TaskStatus.COMPLETED and task.result:
        result = task.result["response"]

    return TaskStatusResponse(
        id=task.id,
        status=task.status.value,
        result=result,
        error=task.error,
    )


@router.post("/save", response_model=V7ModelVersionResponse)
async def save_fit(
    request: V7FitSaveRequest,
    db: AsyncSession = Depends(get_db),
):
    """Save a completed v7 fit as a new model version."""
    from app.services.v7_model import (
        BS_FIXED,
        CV_FIXED,
        EFF_LAG_MIN,
        FrozenSensor,
        StageAGlobalParams,
        StageAPerHold,
        StageBGlobalParams,
        StageBPerHold,
        V7ModelParams,
    )

    # Reconstruct V7ModelParams from request
    stage_a_global = StageAGlobalParams(
        tau_0=request.stage_a.tau_0,
        p=request.stage_a.p,
    )
    stage_a_per_hold = [
        StageAPerHold(
            hold_id=ph.hold_id,
            hold_type=ph.hold_type,
            delta=ph.delta,
            s_min=ph.s_min,
            v_up=ph.v_up,
            eff_lag=ph.eff_lag,
        )
        for ph in request.stage_a.per_hold
    ]
    stage_b_global = StageBGlobalParams(
        k_co2=request.stage_b.k_co2,
        gamma=request.stage_b.gamma,
    )
    stage_b_per_hold = [
        StageBPerHold(
            hold_id=ph.hold_id,
            hold_type=ph.hold_type,
            tau_washout=ph.tau_washout,
            paco2_0=ph.paco2_0,
        )
        for ph in request.stage_b.per_hold
    ]

    s_base = {int(k): v for k, v in request.s_base_values.items()}

    params = V7ModelParams(
        stage_a_global=stage_a_global,
        stage_a_per_hold=stage_a_per_hold,
        stage_b_global=stage_b_global,
        stage_b_per_hold=stage_b_per_hold,
        s_base_values=s_base,
    )

    stage_a_diag = {
        "converged": request.stage_a.converged,
        "objective_val": request.stage_a.objective_val,
        "per_hold_r2": {str(ph.hold_id): ph.r_squared for ph in request.stage_a.per_hold},
    }
    stage_b_diag = {
        "converged": request.stage_b.converged,
        "objective_val": request.stage_b.objective_val,
        "per_hold_r2": {str(ph.hold_id): ph.r_squared for ph in request.stage_b.per_hold},
    }

    from app.services.model_manager import save_model_version_v7

    model = await save_model_version_v7(
        db=db,
        params=params,
        r_squared=request.overall_r2,
        objective_val=request.stage_b.objective_val,
        converged=request.stage_a.converged and request.stage_b.converged,
        hold_ids=request.hold_ids,
        stage_a_diagnostics=stage_a_diag,
        stage_b_diagnostics=stage_b_diag,
        notes=request.notes,
        set_active=request.set_active,
    )

    return _model_to_response(model)


def _model_to_response(model: ModelVersionV7) -> V7ModelVersionResponse:
    """Convert ORM ModelVersionV7 to response schema."""
    stage_a_diag = model.get_stage_a_diagnostics()
    stage_b_diag = model.get_stage_b_diagnostics()
    stage_a_params = json.loads(model.stage_a_params_json)
    stage_b_params = json.loads(model.stage_b_params_json)

    stage_a_r2 = stage_a_diag.get("per_hold_r2", {})
    stage_b_r2 = stage_b_diag.get("per_hold_r2", {})

    return V7ModelVersionResponse(
        id=model.id,
        version=model.version,
        is_active=model.is_active,
        stage_a=V7StageAParamsResponse(
            tau_0=stage_a_params["stage_a_global"]["tau_0"],
            p=stage_a_params["stage_a_global"]["p"],
            per_hold=[
                V7StageAPerHoldResponse(
                    hold_id=ph["hold_id"],
                    hold_type=ph["hold_type"],
                    delta=ph["delta"],
                    s_min=ph["s_min"],
                    v_up=ph["v_up"],
                    eff_lag=ph["eff_lag"],
                    r_squared=stage_a_r2.get(str(ph["hold_id"]), 0.0),
                )
                for ph in stage_a_params["stage_a_per_hold"]
            ],
            converged=stage_a_diag.get("converged", True),
            objective_val=stage_a_diag.get("objective_val", 0.0),
        ),
        stage_b=V7StageBParamsResponse(
            k_co2=stage_b_params["stage_b_global"]["k_co2"],
            gamma=stage_b_params["stage_b_global"]["gamma"],
            per_hold=[
                V7StageBPerHoldResponse(
                    hold_id=ph["hold_id"],
                    hold_type=ph["hold_type"],
                    tau_washout=ph["tau_washout"],
                    paco2_0=ph["paco2_0"],
                    r_squared=stage_b_r2.get(str(ph["hold_id"]), 0.0),
                )
                for ph in stage_b_params["stage_b_per_hold"]
            ],
            converged=stage_b_diag.get("converged", True),
            objective_val=stage_b_diag.get("objective_val", 0.0),
        ),
        r_squared=model.r_squared,
        objective_val=model.objective_val,
        converged=model.converged,
        hold_ids=model.get_hold_ids(),
        notes=model.notes,
        created_at=model.created_at,
    )
