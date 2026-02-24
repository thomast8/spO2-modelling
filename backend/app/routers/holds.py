"""Hold management API endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.models.db_models import Hold
from app.models.schemas import HoldDetailResponse, HoldUpdateRequest

router = APIRouter(prefix="/holds", tags=["holds"])


@router.get("/{hold_id}", response_model=HoldDetailResponse)
async def get_hold(hold_id: int, db: AsyncSession = Depends(get_db)):
    """Get hold details including all data points."""
    result = await db.execute(
        select(Hold).where(Hold.id == hold_id).options(selectinload(Hold.data_points))
    )
    hold = result.scalar_one_or_none()
    if not hold:
        raise HTTPException(status_code=404, detail="Hold not found")

    return HoldDetailResponse(
        id=hold.id,
        hold_number=hold.hold_number,
        hold_type=hold.hold_type,
        duration_s=hold.duration_s,
        min_spo2=hold.min_spo2,
        min_hr=hold.min_hr,
        include_in_fit=hold.include_in_fit,
        start_time_s=hold.start_time_s,
        end_time_s=hold.end_time_s,
        session_id=hold.session_id,
        data_points=[
            {"elapsed_s": dp.elapsed_s, "spo2": dp.spo2, "hr": dp.hr}
            for dp in hold.data_points
        ],
    )


@router.patch("/{hold_id}", response_model=HoldDetailResponse)
async def update_hold(
    hold_id: int,
    update: HoldUpdateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Update hold metadata (type, include_in_fit)."""
    result = await db.execute(
        select(Hold).where(Hold.id == hold_id).options(selectinload(Hold.data_points))
    )
    hold = result.scalar_one_or_none()
    if not hold:
        raise HTTPException(status_code=404, detail="Hold not found")

    if update.hold_type is not None:
        if update.hold_type not in ("FRC", "RV", "FL", "untagged"):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid hold_type: {update.hold_type}. Must be FRC, RV, FL, or untagged",
            )
        hold.hold_type = update.hold_type

    if update.include_in_fit is not None:
        hold.include_in_fit = update.include_in_fit

    await db.flush()

    return HoldDetailResponse(
        id=hold.id,
        hold_number=hold.hold_number,
        hold_type=hold.hold_type,
        duration_s=hold.duration_s,
        min_spo2=hold.min_spo2,
        min_hr=hold.min_hr,
        include_in_fit=hold.include_in_fit,
        start_time_s=hold.start_time_s,
        end_time_s=hold.end_time_s,
        session_id=hold.session_id,
        data_points=[
            {"elapsed_s": dp.elapsed_s, "spo2": dp.spo2, "hr": dp.hr}
            for dp in hold.data_points
        ],
    )
