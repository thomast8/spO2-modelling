"""Model version management and database seeding."""

import json

from loguru import logger
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.db_models import FitBounds, ModelVersion
from app.services.fitter import DEFAULT_BOUNDS
from app.services.hill_model import ApneaModelParams


async def seed_default_bounds(db: AsyncSession) -> None:
    """Seed default fit bounds, replacing any stale entries from old schema."""
    from sqlalchemy import delete

    # Bulk delete all existing bounds first (handles stale param names)
    await db.execute(delete(FitBounds))
    await db.flush()

    for hold_type, bounds in DEFAULT_BOUNDS.items():
        for param_name, (lower, upper) in bounds.items():
            db.add(FitBounds(
                hold_type=hold_type,
                param_name=param_name,
                lower=lower,
                upper=upper,
            ))

    await db.flush()
    logger.info(f"Seeded default fit bounds for {list(DEFAULT_BOUNDS.keys())}")


async def get_bounds_for_type(db: AsyncSession, hold_type: str) -> dict[str, tuple[float, float]]:
    """Get parameter bounds for a hold type from the database."""
    result = await db.execute(
        select(FitBounds).where(FitBounds.hold_type == hold_type)
    )
    rows = result.scalars().all()

    if not rows:
        # Fall back to defaults
        return DEFAULT_BOUNDS.get(hold_type, DEFAULT_BOUNDS["FL"])

    return {row.param_name: (row.lower, row.upper) for row in rows}


async def get_next_version(db: AsyncSession, hold_type: str) -> int:
    """Get the next version number for a hold type."""
    max_version = await db.scalar(
        select(func.max(ModelVersion.version)).where(ModelVersion.hold_type == hold_type)
    )
    return (max_version or 0) + 1


async def save_model_version(
    db: AsyncSession,
    hold_type: str,
    params: ApneaModelParams,
    r_squared: float,
    objective_val: float,
    converged: bool,
    hold_ids: list[int],
    notes: str | None = None,
    set_active: bool = True,
) -> ModelVersion:
    """Save a new model version to the database.

    Args:
        db: Database session
        hold_type: FRC, RV, or FL
        params: Fitted model parameters
        r_squared: Overall R-squared
        objective_val: Optimization objective value
        converged: Whether optimizer converged
        hold_ids: IDs of holds used in fit
        notes: Optional notes
        set_active: Whether to set this as the active version

    Returns:
        The created ModelVersion record
    """
    version = await get_next_version(db, hold_type)

    # Deactivate current active version if setting new one
    if set_active:
        result = await db.execute(
            select(ModelVersion).where(
                ModelVersion.hold_type == hold_type,
                ModelVersion.is_active == True,  # noqa: E712
            )
        )
        for existing in result.scalars().all():
            existing.is_active = False

    model = ModelVersion(
        hold_type=hold_type,
        version=version,
        is_active=set_active,
        pao2_0=params.pao2_0,
        pvo2=params.pvo2,
        tau_washout=params.tau_washout,
        n=params.n,
        bohr_max=params.bohr_max,
        tau_bohr=params.tau_bohr,
        lag=params.lag,
        r_offset=params.r_offset,
        r_squared=r_squared,
        objective_val=objective_val,
        converged=converged,
        n_holds_used=len(hold_ids),
        hold_ids_json=json.dumps(hold_ids),
        notes=notes,
    )
    db.add(model)
    await db.flush()

    logger.info(
        f"Saved {hold_type} model v{version} "
        f"(R²={r_squared:.4f}, active={set_active})"
    )
    return model


async def activate_model(db: AsyncSession, model_id: int) -> ModelVersion:
    """Set a model version as active, deactivating others of the same type."""
    model = await db.get(ModelVersion, model_id)
    if not model:
        raise ValueError(f"Model version {model_id} not found")

    # Deactivate others
    result = await db.execute(
        select(ModelVersion).where(
            ModelVersion.hold_type == model.hold_type,
            ModelVersion.is_active == True,  # noqa: E712
        )
    )
    for existing in result.scalars().all():
        existing.is_active = False

    model.is_active = True
    await db.flush()

    logger.info(f"Activated {model.hold_type} model v{model.version}")
    return model


async def get_active_model(db: AsyncSession, hold_type: str) -> ModelVersion | None:
    """Get the active model version for a hold type."""
    result = await db.execute(
        select(ModelVersion).where(
            ModelVersion.hold_type == hold_type,
            ModelVersion.is_active == True,  # noqa: E712
        )
    )
    return result.scalar_one_or_none()


async def get_model_versions(db: AsyncSession, hold_type: str) -> list[ModelVersion]:
    """Get all model versions for a hold type, ordered by version desc."""
    result = await db.execute(
        select(ModelVersion)
        .where(ModelVersion.hold_type == hold_type)
        .order_by(ModelVersion.version.desc())
    )
    return list(result.scalars().all())
