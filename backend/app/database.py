"""SQLAlchemy async database setup."""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import settings

engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    connect_args={"check_same_thread": False},
)

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """Base class for all ORM models."""


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency that provides an async database session."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """Create all tables and run schema migrations."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_migrate_n_to_gamma)
        await conn.run_sync(_migrate_drop_lag)
        await conn.run_sync(_migrate_bohr_to_co2)


def _migrate_n_to_gamma(conn) -> None:
    """Rename model_versions.n → gamma (Severinghaus steepness exponent)."""
    from sqlalchemy import inspect, text

    inspector = inspect(conn)
    if "model_versions" not in inspector.get_table_names():
        return
    columns = {c["name"] for c in inspector.get_columns("model_versions")}
    if "n" in columns and "gamma" not in columns:
        conn.execute(text("ALTER TABLE model_versions RENAME COLUMN n TO gamma"))
        from loguru import logger
        logger.info("Migrated model_versions: renamed column 'n' → 'gamma'")


def _migrate_drop_lag(conn) -> None:
    """Drop model_versions.lag column (no longer a fitted parameter)."""
    from sqlalchemy import inspect, text

    inspector = inspect(conn)
    if "model_versions" not in inspector.get_table_names():
        return
    columns = {c["name"] for c in inspector.get_columns("model_versions")}
    if "lag" in columns:
        conn.execute(text("ALTER TABLE model_versions DROP COLUMN lag"))
        from loguru import logger
        logger.info("Migrated model_versions: dropped column 'lag'")


def _migrate_bohr_to_co2(conn) -> None:
    """Replace bohr_max/tau_bohr with paco2_0/k_co2 (CO2-driven Bohr effect).

    Old model versions are incompatible; drop bohr_max and tau_bohr columns.
    New columns (paco2_0, k_co2) are created by metadata.create_all.
    """
    from sqlalchemy import inspect, text

    inspector = inspect(conn)
    if "model_versions" not in inspector.get_table_names():
        return
    columns = {c["name"] for c in inspector.get_columns("model_versions")}
    migrated = False
    for old_col in ("bohr_max", "tau_bohr"):
        if old_col in columns:
            conn.execute(text(f"ALTER TABLE model_versions DROP COLUMN {old_col}"))
            migrated = True
    if migrated:
        from loguru import logger
        logger.info("Migrated model_versions: dropped bohr_max/tau_bohr, added paco2_0/k_co2")
