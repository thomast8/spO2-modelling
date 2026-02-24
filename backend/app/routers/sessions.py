"""Session management API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.models.db_models import Hold, HoldData, Session
from app.models.schemas import SessionListItem, SessionResponse
from app.services.csv_parser import parse_csv

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("/upload", response_model=SessionResponse)
async def upload_session(file: UploadFile, db: AsyncSession = Depends(get_db)):
    """Upload a CSV file, parse it, and create a session with detected holds."""
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a .csv")

    content = await file.read()
    logger.info(f"Uploading session from {file.filename} ({len(content)} bytes)")

    try:
        session_data = parse_csv(content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"CSV parsing error: {e}") from None

    # Create session record
    db_session = Session(
        name=session_data.name,
        session_date=session_data.session_date,
        csv_filename=file.filename,
        csv_blob=content,
    )
    db.add(db_session)
    await db.flush()

    # Create holds and data points
    for hold_data in session_data.holds:
        db_hold = Hold(
            session_id=db_session.id,
            hold_number=hold_data.hold_number,
            hold_type="untagged",
            start_time_s=hold_data.start_time_abs_s,
            end_time_s=hold_data.end_time_abs_s,
            duration_s=hold_data.duration_s,
            min_spo2=hold_data.min_spo2,
            min_hr=hold_data.min_hr,
        )
        db.add(db_hold)
        await db.flush()

        # Add per-second data points
        for i in range(len(hold_data.elapsed_s)):
            db.add(HoldData(
                hold_id=db_hold.id,
                elapsed_s=int(hold_data.elapsed_s[i]),
                spo2=float(hold_data.spo2[i]),
                hr=float(hold_data.hr[i]),
            ))

    await db.flush()

    # Reload with relationships
    result = await db.execute(
        select(Session).where(Session.id == db_session.id).options(selectinload(Session.holds))
    )
    session = result.scalar_one()

    logger.info(f"Session {session.id} created: {len(session.holds)} holds")
    return _session_to_response(session)


@router.get("", response_model=list[SessionListItem])
async def list_sessions(db: AsyncSession = Depends(get_db)):
    """List all sessions with summary info."""
    result = await db.execute(
        select(Session).options(selectinload(Session.holds)).order_by(Session.created_at.desc())
    )
    sessions = result.scalars().all()
    return [
        SessionListItem(
            id=s.id,
            name=s.name,
            session_date=s.session_date,
            csv_filename=s.csv_filename,
            n_holds=len(s.holds),
            created_at=s.created_at,
        )
        for s in sessions
    ]


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(session_id: int, db: AsyncSession = Depends(get_db)):
    """Get session details with all holds."""
    result = await db.execute(
        select(Session).where(Session.id == session_id).options(selectinload(Session.holds))
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return _session_to_response(session)


@router.delete("/{session_id}")
async def delete_session(session_id: int, db: AsyncSession = Depends(get_db)):
    """Delete a session and all its holds."""
    session = await db.get(Session, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    await db.delete(session)
    return {"deleted": session_id}


def _session_to_response(session: Session) -> SessionResponse:
    """Convert ORM Session to response schema."""
    return SessionResponse(
        id=session.id,
        name=session.name,
        session_date=session.session_date,
        csv_filename=session.csv_filename,
        notes=session.notes,
        created_at=session.created_at,
        holds=[
            {
                "id": h.id,
                "hold_number": h.hold_number,
                "hold_type": h.hold_type,
                "duration_s": h.duration_s,
                "min_spo2": h.min_spo2,
                "min_hr": h.min_hr,
                "include_in_fit": h.include_in_fit,
                "start_time_s": h.start_time_s,
                "end_time_s": h.end_time_s,
            }
            for h in session.holds
        ],
    )
