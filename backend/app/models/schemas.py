"""Pydantic schemas for API request/response models."""

from datetime import date, datetime

from pydantic import BaseModel, Field


# -- Session ------------------------------------------------------------------


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


# -- Hold ---------------------------------------------------------------------


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


# -- v7 Fit -------------------------------------------------------------------


class V7FitPreviewRequest(BaseModel):
    """Request to start a v7 two-stage model fit."""

    hold_ids: list[int]
    excluded_ids: list[int] = Field(default_factory=list)
    seed: int = 42


class V7StageAPerHoldResponse(BaseModel):
    """Stage A per-hold parameters."""

    hold_id: int
    hold_type: str
    delta: float
    s_min: float
    v_up: float
    eff_lag: float
    r_squared: float


class V7StageAParamsResponse(BaseModel):
    """Stage A global + per-hold parameters."""

    tau_0: float
    p: float
    b_s: float = 1.0
    cv: float = 0.15
    per_hold: list[V7StageAPerHoldResponse]
    converged: bool
    objective_val: float


class V7StageBPerHoldResponse(BaseModel):
    """Stage B per-hold parameters."""

    hold_id: int
    hold_type: str
    tau_washout: float
    paco2_0: float
    r_squared: float


class V7StageBParamsResponse(BaseModel):
    """Stage B global + per-hold parameters."""

    k_co2: float
    gamma: float
    pvo2_fixed: float = 25.0
    per_hold: list[V7StageBPerHoldResponse]
    converged: bool
    objective_val: float


class V7FitPrediction(BaseModel):
    """Prediction for a single hold within a v7 fit."""

    hold_id: int
    elapsed_s: list[float]
    observed: list[float]
    predicted: list[float]
    r_squared: float


class V7FitPreviewResponse(BaseModel):
    """Response from a v7 fit preview (not yet saved)."""

    stage_a: V7StageAParamsResponse
    stage_b: V7StageBParamsResponse
    predictions: list[V7FitPrediction]
    overall_r2: float
    hold_ids: list[int]


class V7FitSaveRequest(BaseModel):
    """Request to save a completed v7 fit."""

    stage_a: V7StageAParamsResponse
    stage_b: V7StageBParamsResponse
    predictions: list[V7FitPrediction]
    overall_r2: float
    hold_ids: list[int]
    s_base_values: dict[str, float]  # JSON keys are strings
    frozen_sensor: dict  # FrozenSensor as dict
    notes: str | None = None
    set_active: bool = True


# -- v7 Model Versions -------------------------------------------------------


class V7ModelVersionResponse(BaseModel):
    """Response for a v7 model version."""

    id: int
    version: int
    is_active: bool
    stage_a: V7StageAParamsResponse
    stage_b: V7StageBParamsResponse
    r_squared: float
    objective_val: float
    converged: bool
    hold_ids: list[int]
    notes: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class V7ModelVersionListResponse(BaseModel):
    """List of v7 model versions."""

    versions: list[V7ModelVersionResponse]
    active_version: int | None


# -- v7 Prediction Curve -----------------------------------------------------


class V7PredictionCurveResponse(BaseModel):
    """Full prediction curve for a hold from a v7 model."""

    hold_id: int
    t: list[float]
    spo2: list[float]
    pao2: list[float]
    p50_eff: list[float]
    sao2_raw: list[float]


# -- Background Task ----------------------------------------------------------


class TaskStatusResponse(BaseModel):
    """Background task status for polling."""

    id: str
    status: str
    result: V7FitPreviewResponse | None = None
    error: str | None = None


# -- Analysis -----------------------------------------------------------------


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


# -- Bounds -------------------------------------------------------------------


class BoundsOverride(BaseModel):
    """Override for a single parameter bound."""

    lower: float
    upper: float


class BoundsResponse(BaseModel):
    """Parameter bounds for a hold type."""

    hold_type: str
    bounds: dict[str, BoundsOverride]


class BoundsUpdateRequest(BaseModel):
    """Request to update parameter bounds."""

    bounds: dict[str, BoundsOverride]
