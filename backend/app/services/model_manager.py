"""Model version management for v7 two-stage models."""

import json

from loguru import logger
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.db_models import FitBounds, ModelVersionV7
from app.services.v7_model import PERHOLD_BOUNDS, V7ModelParams


async def seed_default_bounds(db: AsyncSession) -> None:
    """Seed default v7 fit bounds, replacing any stale entries."""
    from sqlalchemy import delete

    await db.execute(delete(FitBounds))
    await db.flush()

    # v7 uses per-hold-type bounds for tau_washout and paco2_0
    param_names = ["tau_washout", "paco2_0"]
    for hold_type, bounds_list in PERHOLD_BOUNDS.items():
        for param_name, (lower, upper) in zip(param_names, bounds_list):
            db.add(FitBounds(
                hold_type=hold_type,
                param_name=param_name,
                lower=lower,
                upper=upper,
            ))

    await db.flush()
    logger.info(f"Seeded v7 fit bounds for {list(PERHOLD_BOUNDS.keys())}")


async def get_bounds_for_type(db: AsyncSession, hold_type: str) -> dict[str, tuple[float, float]]:
    """Get parameter bounds for a hold type from the database."""
    result = await db.execute(
        select(FitBounds).where(FitBounds.hold_type == hold_type)
    )
    rows = result.scalars().all()

    if not rows:
        bounds_list = PERHOLD_BOUNDS.get(hold_type, PERHOLD_BOUNDS["FL"])
        return dict(zip(["tau_washout", "paco2_0"], bounds_list))

    return {row.param_name: (row.lower, row.upper) for row in rows}


async def get_next_version_v7(db: AsyncSession) -> int:
    """Get the next global version number for v7 models."""
    max_version = await db.scalar(
        select(func.max(ModelVersionV7.version))
    )
    return (max_version or 0) + 1


async def save_model_version_v7(
    db: AsyncSession,
    params: V7ModelParams,
    r_squared: float,
    objective_val: float,
    converged: bool,
    hold_ids: list[int],
    stage_a_diagnostics: dict,
    stage_b_diagnostics: dict,
    notes: str | None = None,
    set_active: bool = True,
) -> ModelVersionV7:
    """Save a new v7 model version to the database."""
    version = await get_next_version_v7(db)

    if set_active:
        result = await db.execute(
            select(ModelVersionV7).where(
                ModelVersionV7.is_active == True,  # noqa: E712
            )
        )
        for existing in result.scalars().all():
            existing.is_active = False

    params_dict = params.to_dict()

    model = ModelVersionV7(
        version=version,
        is_active=set_active,
        stage_a_params_json=json.dumps({
            "stage_a_global": params_dict["stage_a_global"],
            "stage_a_per_hold": params_dict["stage_a_per_hold"],
            "s_base_values": params_dict["s_base_values"],
        }),
        stage_b_params_json=json.dumps({
            "stage_b_global": params_dict["stage_b_global"],
            "stage_b_per_hold": params_dict["stage_b_per_hold"],
            "s_base_values": params_dict["s_base_values"],
        }),
        stage_a_diagnostics_json=json.dumps(stage_a_diagnostics),
        stage_b_diagnostics_json=json.dumps(stage_b_diagnostics),
        hold_ids_json=json.dumps(hold_ids),
        r_squared=r_squared,
        objective_val=objective_val,
        converged=converged,
        notes=notes,
    )
    db.add(model)
    await db.flush()

    logger.info(f"Saved v7 model v{version} (R2={r_squared:.4f}, active={set_active})")
    return model


async def activate_model_v7(db: AsyncSession, model_id: int) -> ModelVersionV7:
    """Set a v7 model version as active, deactivating all others."""
    model = await db.get(ModelVersionV7, model_id)
    if not model:
        raise ValueError(f"Model version {model_id} not found")

    result = await db.execute(
        select(ModelVersionV7).where(
            ModelVersionV7.is_active == True,  # noqa: E712
        )
    )
    for existing in result.scalars().all():
        existing.is_active = False

    model.is_active = True
    await db.flush()

    logger.info(f"Activated v7 model v{model.version}")
    return model


async def get_active_model_v7(db: AsyncSession) -> ModelVersionV7 | None:
    """Get the active v7 model version."""
    result = await db.execute(
        select(ModelVersionV7).where(
            ModelVersionV7.is_active == True,  # noqa: E712
        )
    )
    return result.scalar_one_or_none()


async def get_model_versions_v7(db: AsyncSession) -> list[ModelVersionV7]:
    """Get all v7 model versions, ordered by version desc."""
    result = await db.execute(
        select(ModelVersionV7).order_by(ModelVersionV7.version.desc())
    )
    return list(result.scalars().all())
