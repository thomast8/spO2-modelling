"""Pydantic schemas for API request/response models."""

from datetime import date, datetime

from pydantic import BaseModel, Field


# ── Session ──────────────────────────────────────────────────────────────────


class HoldSummary(BaseModel):
    """Hold summary returned within session responses."""

    id: int
    hold_number: int
    hold_type: str
    duration_s: int
    min_spo2: float | None
    min_hr: float | None
    include_in_fit: bool
    start_time_s: int
    end_time_s: int


class SessionResponse(BaseModel):
    """Response for a session with its holds."""

    id: int
    name: str
    session_date: date
    csv_filename: str
    notes: str | None
    created_at: datetime
    holds: list[HoldSummary]

    model_config = {"from_attributes": True}


class SessionListItem(BaseModel):
    """Summary item for session list."""

    id: int
    name: str
    session_date: date
    csv_filename: str
    n_holds: int
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Hold ─────────────────────────────────────────────────────────────────────


class HoldDataPoint(BaseModel):
    """A single data point within a hold."""

    elapsed_s: int
    spo2: float
    hr: float


class HoldDetailResponse(BaseModel):
    """Detailed hold response including all data points."""

    id: int
    hold_number: int
    hold_type: str
    duration_s: int
    min_spo2: float | None
    min_hr: float | None
    include_in_fit: bool
    start_time_s: int
    end_time_s: int
    session_id: int
    data_points: list[HoldDataPoint]

    model_config = {"from_attributes": True}


class HoldUpdateRequest(BaseModel):
    """Request to update a hold's metadata."""

    hold_type: str | None = None
    include_in_fit: bool | None = None


# ── Fit ──────────────────────────────────────────────────────────────────────


class BoundsOverride(BaseModel):
    """Override for a single parameter bound."""

    lower: float
    upper: float


class FitPreviewRequest(BaseModel):
    """Request to preview a model fit."""

    hold_type: str = Field(..., pattern="^(FRC|RV|FL)$")
    hold_ids: list[int]
    bounds_override: dict[str, BoundsOverride] | None = None
    seed: int = 42


class FitPrediction(BaseModel):
    """Prediction for a single hold within a fit."""

    hold_id: int
    elapsed_s: list[float]
    observed: list[float]
    predicted: list[float]
    r_squared: float


class ApneaModelParamsResponse(BaseModel):
    """Apnea model parameters (exponential washout + saturating Bohr effect).

    p50_base and n are fixed haemoglobin constants, included for display only.
    """

    pao2_0: float
    pvo2: float
    tau_washout: float
    bohr_max: float
    tau_bohr: float
    lag: float
    r_offset: float
    # Fixed constants (not fitted), included for API transparency
    p50_base: float = 26.6
    n: float = 2.7


class FitPreviewResponse(BaseModel):
    """Response from a fit preview (not yet saved)."""

    params: ApneaModelParamsResponse
    r_squared: float
    r_squared_per_hold: list[float]
    objective_val: float
    converged: bool
    n_holds: int
    n_data_points: int
    predictions: list[FitPrediction]


class FitSaveRequest(BaseModel):
    """Request to save a fit as a new model version."""

    hold_type: str = Field(..., pattern="^(FRC|RV|FL)$")
    params: ApneaModelParamsResponse
    hold_ids: list[int]
    r_squared: float
    objective_val: float
    converged: bool
    notes: str | None = None
    set_active: bool = True


# ── Model Versions ───────────────────────────────────────────────────────────


class ModelVersionResponse(BaseModel):
    """Response for a model version."""

    id: int
    hold_type: str
    version: int
    is_active: bool
    params: ApneaModelParamsResponse
    r_squared: float
    objective_val: float
    converged: bool
    n_holds_used: int
    hold_ids: list[int]
    notes: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class ModelVersionListResponse(BaseModel):
    """List of model versions for a hold type."""

    hold_type: str
    versions: list[ModelVersionResponse]
    active_version: int | None


class AllModelsResponse(BaseModel):
    """All model versions grouped by hold type."""

    FRC: ModelVersionListResponse
    RV: ModelVersionListResponse
    FL: ModelVersionListResponse


# ── Analysis ─────────────────────────────────────────────────────────────────


class ThresholdResponse(BaseModel):
    """Response from threshold crossing analysis."""

    threshold: float
    crossing_time_s: float | None
    crossing_time_fmt: str | None
    spo2_at_end: float | None


class SensitivityPointResponse(BaseModel):
    """One point in parameter sensitivity analysis."""

    param_value: float
    pct_change: float
    crossing_time_s: float | None
    margin_s: float | None
    spo2_at_ref: float


class DesatRatePointResponse(BaseModel):
    """Desaturation rate at a time point."""

    time_s: float
    rate_per_min: float
    spo2: float


class PredictionCurveResponse(BaseModel):
    """Full prediction curve for visualization."""

    t: list[float]
    spo2: list[float]
    spo2_base: list[float]
    pao2: list[float]
    p50_eff: list[float]


# ── Bounds ───────────────────────────────────────────────────────────────────


class BoundsResponse(BaseModel):
    """Parameter bounds for a hold type."""

    hold_type: str
    bounds: dict[str, BoundsOverride]


class BoundsUpdateRequest(BaseModel):
    """Request to update parameter bounds."""

    bounds: dict[str, BoundsOverride]
