// -- Session ------------------------------------------------------------------

export interface HoldSummary {
  id: number;
  hold_number: number;
  hold_type: string;
  duration_s: number;
  min_spo2: number | null;
  min_hr: number | null;
  include_in_fit: boolean;
  start_time_s: number;
  end_time_s: number;
}

export interface SessionResponse {
  id: number;
  name: string;
  session_date: string;
  csv_filename: string;
  notes: string | null;
  created_at: string;
  holds: HoldSummary[];
}

export interface SessionListItem {
  id: number;
  name: string;
  session_date: string;
  csv_filename: string;
  n_holds: number;
  created_at: string;
}

// -- Hold ---------------------------------------------------------------------

export interface HoldDataPoint {
  elapsed_s: number;
  spo2: number;
  hr: number;
}

export interface HoldDetailResponse {
  id: number;
  hold_number: number;
  hold_type: string;
  duration_s: number;
  min_spo2: number | null;
  min_hr: number | null;
  include_in_fit: boolean;
  start_time_s: number;
  end_time_s: number;
  session_id: number;
  data_points: HoldDataPoint[];
}

// -- v7 Fit -------------------------------------------------------------------

export interface V7FitPreviewRequest {
  hold_ids: number[];
  excluded_ids?: number[];
  seed?: number;
}

export interface V7StageAPerHold {
  hold_id: number;
  hold_type: string;
  delta: number;
  s_min: number;
  v_up: number;
  eff_lag: number;
  r_squared: number;
}

export interface V7StageAParams {
  tau_0: number;
  p: number;
  b_s: number;
  cv: number;
  per_hold: V7StageAPerHold[];
  converged: boolean;
  objective_val: number;
}

export interface V7StageBPerHold {
  hold_id: number;
  hold_type: string;
  tau_washout: number;
  paco2_0: number;
  r_squared: number;
}

export interface V7StageBParams {
  k_co2: number;
  gamma: number;
  pvo2_fixed: number;
  per_hold: V7StageBPerHold[];
  converged: boolean;
  objective_val: number;
}

export interface V7FitPrediction {
  hold_id: number;
  elapsed_s: number[];
  observed: number[];
  predicted: number[];
  r_squared: number;
}

export interface V7FitPreviewResponse {
  stage_a: V7StageAParams;
  stage_b: V7StageBParams;
  predictions: V7FitPrediction[];
  overall_r2: number;
  hold_ids: number[];
}

export interface V7FitSaveRequest {
  stage_a: V7StageAParams;
  stage_b: V7StageBParams;
  predictions: V7FitPrediction[];
  overall_r2: number;
  hold_ids: number[];
  s_base_values: Record<string, number>;
  frozen_sensor: Record<string, unknown>;
  notes?: string;
  set_active?: boolean;
}

// -- v7 Model Versions --------------------------------------------------------

export interface V7ModelVersionResponse {
  id: number;
  version: number;
  is_active: boolean;
  stage_a: V7StageAParams;
  stage_b: V7StageBParams;
  r_squared: number;
  objective_val: number;
  converged: boolean;
  hold_ids: number[];
  notes: string | null;
  created_at: string;
}

export interface V7ModelVersionListResponse {
  versions: V7ModelVersionResponse[];
  active_version: number | null;
}

// -- v7 Prediction Curve ------------------------------------------------------

export interface V7PredictionCurve {
  hold_id: number;
  t: number[];
  spo2: number[];
  pao2: number[];
  p50_eff: number[];
  sao2_raw: number[];
}

// -- Background Task ----------------------------------------------------------

export interface TaskStatusResponse {
  id: string;
  status: "pending" | "running" | "completed" | "failed";
  result: V7FitPreviewResponse | null;
  error: string | null;
}

// -- Analysis -----------------------------------------------------------------

export interface ThresholdResponse {
  threshold: number;
  crossing_time_s: number | null;
  crossing_time_fmt: string | null;
  spo2_at_end: number | null;
}

export interface SensitivityPoint {
  param_value: number;
  pct_change: number;
  crossing_time_s: number | null;
  margin_s: number | null;
  spo2_at_ref: number;
}

export interface DesatRatePoint {
  time_s: number;
  rate_per_min: number;
  spo2: number;
}

// -- Bounds -------------------------------------------------------------------

export interface BoundsOverride {
  lower: number;
  upper: number;
}

export interface BoundsResponse {
  hold_type: string;
  bounds: Record<string, BoundsOverride>;
}

export type HoldType = "FRC" | "RV" | "FL";
