// ── Session ──────────────────────────────────────────────────

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

// ── Hold ─────────────────────────────────────────────────────

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

// ── Fit ──────────────────────────────────────────────────────

export interface BoundsOverride {
  lower: number;
  upper: number;
}

export interface FitPreviewRequest {
  hold_type: string;
  hold_ids: number[];
  bounds_override?: Record<string, BoundsOverride>;
  seed?: number;
}

export interface FitPrediction {
  hold_id: number;
  elapsed_s: number[];
  observed: number[];
  predicted: number[];
  r_squared: number;
}

export interface ApneaModelParams {
  pao2_0: number;
  pvo2: number;
  tau_washout: number;
  gamma: number;
  bohr_max: number;
  tau_bohr: number;
  lag: number;
  r_offset: number;
  // Fixed constant (not fitted), included for display
  p50_base: number;
}

export interface FitPreviewResponse {
  params: ApneaModelParams;
  r_squared: number;
  r_squared_per_hold: number[];
  objective_val: number;
  converged: boolean;
  n_holds: number;
  n_data_points: number;
  predictions: FitPrediction[];
}

export interface FitSaveRequest {
  hold_type: string;
  params: ApneaModelParams;
  hold_ids: number[];
  r_squared: number;
  objective_val: number;
  converged: boolean;
  notes?: string;
  set_active?: boolean;
}

// ── Model Versions ───────────────────────────────────────────

export interface ModelVersionResponse {
  id: number;
  hold_type: string;
  version: number;
  is_active: boolean;
  params: ApneaModelParams;
  r_squared: number;
  objective_val: number;
  converged: boolean;
  n_holds_used: number;
  hold_ids: number[];
  notes: string | null;
  created_at: string;
}

export interface ModelVersionListResponse {
  hold_type: string;
  versions: ModelVersionResponse[];
  active_version: number | null;
}

export interface AllModelsResponse {
  FRC: ModelVersionListResponse;
  RV: ModelVersionListResponse;
  FL: ModelVersionListResponse;
}

// ── Analysis ─────────────────────────────────────────────────

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

export interface PredictionCurve {
  t: number[];
  spo2: number[];
  spo2_base: number[];
  pao2: number[];
  p50_eff: number[];
}

// ── Bounds ───────────────────────────────────────────────────

export interface BoundsResponse {
  hold_type: string;
  bounds: Record<string, BoundsOverride>;
}

export type HoldType = "FRC" | "RV" | "FL";
