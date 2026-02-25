"""Replace linear Bohr with saturating exponential

Replaces old model_versions columns (o2_start, vo2, scale_param, p50, n,
r_decay, tau_decay) with the new exponential washout + saturating Bohr
parameters (pao2_0, pvo2, tau_washout, bohr_max, tau_bohr). Existing
model versions are incompatible and are dropped.

Also clears stale fit_bounds rows so they get re-seeded with new param names.

Revision ID: 1762157ef84d
Revises: be0bbd4da6d5
Create Date: 2026-02-25 10:05:02.773091

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "1762157ef84d"
down_revision: str | Sequence[str] | None = "be0bbd4da6d5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Rebuild model_versions with new column schema."""
    # SQLite has limited ALTER TABLE — rebuild the table
    op.rename_table("model_versions", "model_versions_old")

    op.create_table(
        "model_versions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("hold_type", sa.String(length=20), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        # New model parameters
        sa.Column("pao2_0", sa.Float(), nullable=False),
        sa.Column("pvo2", sa.Float(), nullable=False),
        sa.Column("tau_washout", sa.Float(), nullable=False),
        sa.Column("n", sa.Float(), nullable=False),
        sa.Column("bohr_max", sa.Float(), nullable=False),
        sa.Column("tau_bohr", sa.Float(), nullable=False),
        sa.Column("lag", sa.Float(), nullable=False),
        sa.Column("r_offset", sa.Float(), nullable=False),
        # Fit diagnostics
        sa.Column("r_squared", sa.Float(), nullable=False),
        sa.Column("objective_val", sa.Float(), nullable=False),
        sa.Column("converged", sa.Boolean(), nullable=False),
        sa.Column("n_holds_used", sa.Integer(), nullable=False),
        sa.Column("hold_ids_json", sa.Text(), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("hold_type", "version"),
    )

    # Old model versions are incompatible — don't migrate data
    op.drop_table("model_versions_old")

    # Clear stale fit_bounds so they re-seed with new param names on startup
    op.execute("DELETE FROM fit_bounds")


def downgrade() -> None:
    """Rebuild model_versions with old column schema."""
    op.rename_table("model_versions", "model_versions_new")

    op.create_table(
        "model_versions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("hold_type", sa.String(length=20), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("o2_start", sa.Float(), nullable=False),
        sa.Column("vo2", sa.Float(), nullable=False),
        sa.Column("scale_param", sa.Float(), nullable=False),
        sa.Column("p50", sa.Float(), nullable=False),
        sa.Column("n", sa.Float(), nullable=False),
        sa.Column("r_offset", sa.Float(), nullable=False),
        sa.Column("r_decay", sa.Float(), nullable=False),
        sa.Column("tau_decay", sa.Float(), nullable=False),
        sa.Column("lag", sa.Float(), nullable=False),
        sa.Column("r_squared", sa.Float(), nullable=False),
        sa.Column("objective_val", sa.Float(), nullable=False),
        sa.Column("converged", sa.Boolean(), nullable=False),
        sa.Column("n_holds_used", sa.Integer(), nullable=False),
        sa.Column("hold_ids_json", sa.Text(), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("hold_type", "version"),
    )

    op.drop_table("model_versions_new")
    op.execute("DELETE FROM fit_bounds")
