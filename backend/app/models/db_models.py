"""SQLAlchemy ORM models for the SpO2 modelling database."""

from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Session(Base):
    """An apnea training session uploaded from CSV."""

    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    session_date: Mapped[date] = mapped_column(Date, nullable=False)
    csv_filename: Mapped[str] = mapped_column(String(500), nullable=False)
    csv_blob: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    holds: Mapped[list["Hold"]] = relationship(
        back_populates="session", cascade="all, delete-orphan", order_by="Hold.hold_number"
    )


class Hold(Base):
    """A single apnea hold detected within a session."""

    __tablename__ = "holds"
    __table_args__ = (UniqueConstraint("session_id", "hold_number"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )
    hold_number: Mapped[int] = mapped_column(Integer, nullable=False)
    hold_type: Mapped[str] = mapped_column(
        String(20), nullable=False, default="untagged"
    )  # FRC, RV, FL, untagged
    start_time_s: Mapped[int] = mapped_column(Integer, nullable=False)
    end_time_s: Mapped[int] = mapped_column(Integer, nullable=False)
    duration_s: Mapped[int] = mapped_column(Integer, nullable=False)
    min_spo2: Mapped[float | None] = mapped_column(Float, nullable=True)
    min_hr: Mapped[float | None] = mapped_column(Float, nullable=True)
    include_in_fit: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    session: Mapped["Session"] = relationship(back_populates="holds")
    data_points: Mapped[list["HoldData"]] = relationship(
        back_populates="hold", cascade="all, delete-orphan", order_by="HoldData.elapsed_s"
    )


class HoldData(Base):
    """Per-second biometric data for a hold."""

    __tablename__ = "hold_data"
    __table_args__ = (UniqueConstraint("hold_id", "elapsed_s"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    hold_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("holds.id", ondelete="CASCADE"), nullable=False
    )
    elapsed_s: Mapped[int] = mapped_column(Integer, nullable=False)
    spo2: Mapped[float] = mapped_column(Float, nullable=False)
    hr: Mapped[float] = mapped_column(Float, nullable=False)

    hold: Mapped["Hold"] = relationship(back_populates="data_points")


class ModelVersion(Base):
    """A fitted model version for a specific hold type."""

    __tablename__ = "model_versions"
    __table_args__ = (UniqueConstraint("hold_type", "version"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    hold_type: Mapped[str] = mapped_column(String(20), nullable=False)  # FRC, RV, FL
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Apnea model parameters (exponential washout + saturating Bohr effect)
    pao2_0: Mapped[float] = mapped_column(Float, nullable=False)
    pvo2: Mapped[float] = mapped_column(Float, nullable=False)
    tau_washout: Mapped[float] = mapped_column(Float, nullable=False)
    bohr_max: Mapped[float] = mapped_column(Float, nullable=False)
    tau_bohr: Mapped[float] = mapped_column(Float, nullable=False)
    lag: Mapped[float] = mapped_column(Float, nullable=False)
    r_offset: Mapped[float] = mapped_column(Float, nullable=False)

    # Fit diagnostics
    r_squared: Mapped[float] = mapped_column(Float, nullable=False)
    objective_val: Mapped[float] = mapped_column(Float, nullable=False)
    converged: Mapped[bool] = mapped_column(Boolean, nullable=False)
    n_holds_used: Mapped[int] = mapped_column(Integer, nullable=False)
    hold_ids_json: Mapped[str] = mapped_column(Text, nullable=False)  # JSON array

    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    def to_model_params(self):
        """Convert to ApneaModelParams dataclass."""
        from app.services.hill_model import ApneaModelParams

        return ApneaModelParams(
            pao2_0=self.pao2_0,
            pvo2=self.pvo2,
            tau_washout=self.tau_washout,
            bohr_max=self.bohr_max,
            tau_bohr=self.tau_bohr,
            lag=self.lag,
            r_offset=self.r_offset,
        )


class FitBounds(Base):
    """Parameter bounds for model fitting, per hold type."""

    __tablename__ = "fit_bounds"
    __table_args__ = (UniqueConstraint("hold_type", "param_name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    hold_type: Mapped[str] = mapped_column(String(20), nullable=False)
    param_name: Mapped[str] = mapped_column(String(50), nullable=False)
    lower: Mapped[float] = mapped_column(Float, nullable=False)
    upper: Mapped[float] = mapped_column(Float, nullable=False)
