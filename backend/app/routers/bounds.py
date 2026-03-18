"""v7 fit bounds management API endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.db_models import FitBounds
from app.models.schemas import BoundsOverride, BoundsResponse, BoundsUpdateRequest
from app.services.model_manager import get_bounds_for_type
from app.services.v7_model import (
    DELTA_BOUNDS,
    GAMMA_BOUNDS,
    K_CO2_BOUNDS,
    P_BOUNDS,
    S_MIN_BOUNDS,
    TAU0_BOUNDS,
    V_UP_BOUNDS,
)

router = APIRouter(prefix="/bounds", tags=["bounds"])

VALID_HOLD_TYPES = ("FRC", "RV", "FL")

# Stage A global bounds (not per-hold-type)
STAGE_A_GLOBAL_BOUNDS = {
    "tau_0": TAU0_BOUNDS,
    "p": P_BOUNDS,
    "delta": DELTA_BOUNDS,
    "s_min": S_MIN_BOUNDS,
    "v_up": V_UP_BOUNDS,
}

STAGE_B_GLOBAL_BOUNDS = {
    "k_co2": K_CO2_BOUNDS,
    "gamma": GAMMA_BOUNDS,
}


@router.get("/global", response_model=dict)
async def get_global_bounds():
    """Get stage A and B global bounds (not per-hold-type)."""
    return {
        "stage_a": {
            name: {"lower": lo, "upper": hi}
            for name, (lo, hi) in STAGE_A_GLOBAL_BOUNDS.items()
        },
        "stage_b": {
            name: {"lower": lo, "upper": hi}
            for name, (lo, hi) in STAGE_B_GLOBAL_BOUNDS.items()
        },
    }


@router.get("/{hold_type}", response_model=BoundsResponse)
async def get_bounds(
    hold_type: str,
    db: AsyncSession = Depends(get_db),
):
    """Get per-hold-type parameter bounds (tau_washout, paco2_0)."""
    if hold_type not in VALID_HOLD_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid hold_type: {hold_type}. Must be FRC, RV, or FL",
        )

    bounds = await get_bounds_for_type(db, hold_type)
    return BoundsResponse(
        hold_type=hold_type,
        bounds={
            name: BoundsOverride(lower=lo, upper=hi)
            for name, (lo, hi) in bounds.items()
        },
    )


@router.put("/{hold_type}", response_model=BoundsResponse)
async def update_bounds(
    hold_type: str,
    request: BoundsUpdateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Update per-hold-type parameter bounds."""
    if hold_type not in VALID_HOLD_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid hold_type: {hold_type}. Must be FRC, RV, or FL",
        )

    for name, bound in request.bounds.items():
        if bound.lower >= bound.upper:
            raise HTTPException(
                status_code=400,
                detail=f"Lower bound must be less than upper bound for {name}",
            )

        result = await db.execute(
            select(FitBounds).where(
                FitBounds.hold_type == hold_type,
                FitBounds.param_name == name,
            )
        )
        existing = result.scalar_one_or_none()

        if existing:
            existing.lower = bound.lower
            existing.upper = bound.upper
        else:
            db.add(FitBounds(
                hold_type=hold_type,
                param_name=name,
                lower=bound.lower,
                upper=bound.upper,
            ))

    await db.flush()

    bounds = await get_bounds_for_type(db, hold_type)
    return BoundsResponse(
        hold_type=hold_type,
        bounds={
            name: BoundsOverride(lower=lo, upper=hi)
            for name, (lo, hi) in bounds.items()
        },
    )
