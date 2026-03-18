"""
v7 Two-Stage SpO2 Desaturation Model
=====================================

Two-stage architecture for modelling SpO2 desaturation during breath-hold apnea:

Stage A - Sensor Calibration:
    Power-law latent descent + gamma kernel convolution models the pulse oximeter's
    delayed, smoothed response. Global params: tau_0 (lag), p (descent exponent).
    Per-hold params: delta (lag offset), S_min (nadir), v_up (recovery slope).

Stage B - Apnea Physiology (frozen sensor):
    Exponential PaO2 washout + linear CO2 accumulation + Severinghaus ODC.
    Global params: k_co2 (CO2 rate), gamma (ODC steepness).
    Per-hold params: tau_washout, paco2_0.

Source: exp_v7_08 shipping configuration.
"""

from dataclasses import asdict, dataclass, field

import numpy as np
from loguru import logger
from scipy.special import gammainc


# -- Physical / physiological constants --------------------------------------

P50_BASE = 26.6           # Baseline P50 of the ODC (mmHg)
FIO2_PB_PH2O = 149.2     # FiO2 * (PB - PH2O) = 0.2093 * (760 - 47)
RQ = 0.8                 # Respiratory quotient
PVO2_FIXED = 25.0        # Mixed venous PO2 (mmHg), fixed
PACO2_NORMAL = 40.0      # Normal arterial PCO2 (mmHg)

# -- Config B sensor constants -----------------------------------------------

CV_FIXED = 0.15          # Gamma kernel CV (fixed)
BS_FIXED = 1.0           # Sensor scaling factor (fixed at 1.0)
EFF_LAG_MIN = 5.0        # Minimum effective lag floor (seconds)

# -- Stage A regularization lambdas ------------------------------------------

LAMBDA_TAU0 = 3.125      # LogNormal(log 18, 0.4): 1/(2*0.4^2)
LAMBDA_DELTA = 5.0       # StudentT-like shrinkage, sigma ~3s
LAMBDA_ZEROSUM = 500.0   # Zero-sum on deltas
LAMBDA_P = 4.08          # LogNormal(log 3.5, 0.35): 1/(2*0.35^2)
LAMBDA_NADIR = 500.0     # Huber timing penalty (delta=8s)
P_PRIOR_CENTER = 3.5     # Prior center for p
TAU0_PRIOR_CENTER = 18.0 # Prior center for tau_0

# -- Stage B regularization lambdas ------------------------------------------

LAMBDA_K_CO2 = 1250.0    # N(0.06, 0.02): 1/(2*0.02^2)
K_CO2_PRIOR_CENTER = 0.06
LAMBDA_PACO2 = 0.056     # N(40, 3): 1/(2*3^2)
LAMBDA_GAMMA = 2.0       # N(2.0, 0.5): 1/(2*0.5^2)
GAMMA_PRIOR_CENTER = 2.0
LAMBDA_REG = 10.0        # Per-hold IC -> type-mean

# -- Student-t NLL parameters ------------------------------------------------

NU_STUDENT = 4.0
SIGMA_STUDENT = 1.0

# -- Nadir window -------------------------------------------------------------

NADIR_WINDOW_AFTER = 45  # seconds after t_end to include in nadir window

# -- Bounds -------------------------------------------------------------------

# Stage A
TAU0_BOUNDS = (5, 45)
P_BOUNDS = (1.0, 5.0)
DELTA_BOUNDS = (-20, 20)
S_MIN_BOUNDS = (30, 100)
V_UP_BOUNDS = (0.0, 3.0)

# Stage B per-hold bounds: {hold_type: [(tau_washout_lo, tau_washout_hi), (paco2_0_lo, paco2_0_hi)]}
PERHOLD_BOUNDS: dict[str, list[tuple[float, float]]] = {
    "FL": [(50, 500), (20, 50)],
    "FRC": [(20, 100), (25, 50)],
    "RV": [(10, 80), (30, 55)],
}
PERHOLD_NAMES = ["tau_washout", "paco2_0"]
N_PH = len(PERHOLD_NAMES)

# Stage B global bounds
GAMMA_BOUNDS = (0.8, 4.0)
K_CO2_BOUNDS = (0.02, 0.25)


# -- Dataclasses --------------------------------------------------------------


@dataclass
class StageAGlobalParams:
    """Stage A global (shared across holds) parameters."""
    tau_0: float    # Base sensor lag (seconds)
    p: float        # Power-law descent exponent


@dataclass
class StageAPerHold:
    """Stage A per-hold parameters."""
    hold_id: int
    hold_type: str
    delta: float       # Lag offset for this hold
    s_min: float       # Nadir SpO2 in latent template
    v_up: float        # Recovery slope after nadir
    eff_lag: float     # Effective lag = max(tau_0 + delta, EFF_LAG_MIN)


@dataclass
class StageBGlobalParams:
    """Stage B global (shared across holds) parameters."""
    k_co2: float    # CO2 accumulation rate (mmHg/s)
    gamma: float    # Severinghaus ODC steepness exponent


@dataclass
class StageBPerHold:
    """Stage B per-hold parameters."""
    hold_id: int
    hold_type: str
    tau_washout: float  # O2 washout time constant (seconds)
    paco2_0: float      # Initial alveolar PCO2 (mmHg)


@dataclass
class FrozenSensor:
    """Frozen sensor parameters from Stage A, used in Stage B."""
    tau_0: float
    b_s: float
    p: float
    deltas: list[float]
    cv: float


@dataclass
class V7ModelParams:
    """Complete v7 model parameters (both stages)."""
    stage_a_global: StageAGlobalParams
    stage_a_per_hold: list[StageAPerHold]
    stage_b_global: StageBGlobalParams
    stage_b_per_hold: list[StageBPerHold]
    s_base_values: dict[int, float]  # hold_id -> baseline SpO2

    def to_dict(self) -> dict:
        return {
            "stage_a_global": asdict(self.stage_a_global),
            "stage_a_per_hold": [asdict(ph) for ph in self.stage_a_per_hold],
            "stage_b_global": asdict(self.stage_b_global),
            "stage_b_per_hold": [asdict(ph) for ph in self.stage_b_per_hold],
            "s_base_values": self.s_base_values,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "V7ModelParams":
        return cls(
            stage_a_global=StageAGlobalParams(**d["stage_a_global"]),
            stage_a_per_hold=[StageAPerHold(**ph) for ph in d["stage_a_per_hold"]],
            stage_b_global=StageBGlobalParams(**d["stage_b_global"]),
            stage_b_per_hold=[StageBPerHold(**ph) for ph in d["stage_b_per_hold"]],
            s_base_values={int(k): v for k, v in d["s_base_values"].items()},
        )


@dataclass
class StageResult:
    """Result from one stage of fitting."""
    converged: bool
    objective_val: float
    per_hold_r2: dict[int, float]          # hold_id -> R2
    per_hold_predictions: dict[int, dict]  # hold_id -> {elapsed_s, observed, predicted}


@dataclass
class V7FitResult:
    """Complete result from the two-stage v7 fit."""
    params: V7ModelParams
    stage_a_result: StageResult
    stage_b_result: StageResult
    overall_r2: float
    hold_ids: list[int]
    frozen_sensor: FrozenSensor
    # Per-hold flat arrays for latent curves (Stage A)
    stage_a_latent: dict[int, dict] = field(default_factory=dict)


# -- Gamma kernel convolution -------------------------------------------------


def gamma_kernel_smooth(mean_lag: float, cv: float, max_support: int = 120) -> np.ndarray:
    """Build a discrete gamma kernel for sensor lag convolution."""
    k = 1.0 / (cv * cv)
    theta = mean_lag * cv * cv
    support = min(int(mean_lag + 5.0 * mean_lag * cv), max_support)
    support = max(support, 2)
    edges = np.arange(support + 1, dtype=float)
    cdf_vals = gammainc(k, edges / theta)
    h = np.diff(cdf_vals)
    total = h.sum()
    if total > 0:
        h /= total
    return h


def apply_gamma_kernel(signal: np.ndarray, mean_lag: float, cv: float) -> np.ndarray:
    """Apply gamma kernel convolution to a signal (models sensor delay)."""
    h = gamma_kernel_smooth(mean_lag, cv)
    pad_len = len(h)
    padded = np.concatenate([np.full(pad_len, signal[0]), signal])
    convolved = np.convolve(padded, h, mode="full")[:len(padded)]
    return convolved[pad_len:]


# -- Power-law latent template ------------------------------------------------


def build_powerlaw_latent(
    t_1hz: np.ndarray,
    t_end: float,
    s_start: float,
    s_min: float,
    v_up: float,
    p: float,
) -> np.ndarray:
    """Build the power-law latent SpO2 template.

    Descends from s_start to s_min during apnea (power-law), then
    recovers linearly at rate v_up after t_end.
    """
    t_turn = max(t_end, 1.0)
    latent = np.empty_like(t_1hz)
    for i, t in enumerate(t_1hz):
        if t <= t_turn:
            frac = t / t_turn
            latent[i] = s_min + (s_start - s_min) * (1.0 - frac ** p)
        else:
            latent[i] = s_min + v_up * (t - t_turn)
    return np.clip(latent, 0.0, 100.0)


# -- Loss functions -----------------------------------------------------------


def student_t_nll(
    residuals: np.ndarray,
    nu: float = NU_STUDENT,
    sigma: float = SIGMA_STUDENT,
) -> float:
    """Robust Student-t negative log-likelihood loss."""
    return float(
        (nu + 1.0) / 2.0 * np.sum(np.log1p(residuals**2 / (nu * sigma**2)))
    )


def huber_loss(a: float | np.ndarray, delta: float = 8.0) -> float | np.ndarray:
    """Huber loss: quadratic for small errors, linear for large."""
    abs_a = np.abs(a)
    return np.where(abs_a <= delta, 0.5 * a**2, delta * (abs_a - 0.5 * delta))


def nadir_timing_penalty_huber(
    t: np.ndarray,
    pred: np.ndarray,
    t_nadir_obs: float,
    lam: float = LAMBDA_NADIR,
    huber_delta: float = 8.0,
) -> float:
    """Penalize mismatch between predicted and observed nadir timing."""
    t_nadir_pred = t[np.argmin(pred)]
    err = t_nadir_pred - t_nadir_obs
    return lam * float(huber_loss(err, delta=huber_delta))


# -- Nadir detection ----------------------------------------------------------


def compute_nadir_info(hold: dict) -> dict:
    """Detect nadir timing and SpO2 from observed data."""
    t, spo2, t_end = hold["t"], hold["spo2"], hold["t_end"]
    local_mask = (t >= t_end - 30) & (t <= t_end + 60)
    if local_mask.sum() >= 5:
        local_spo2 = spo2[local_mask]
        smoothed = np.array([
            np.median(local_spo2[max(0, i - 2):i + 3])
            for i in range(len(local_spo2))
        ])
        idx = np.where(local_mask)[0][np.argmin(smoothed)]
    else:
        idx = np.argmin(spo2)

    apnea_window_mask = t <= t_end + 5
    if apnea_window_mask.sum() > 0:
        apnea_idx = np.where(apnea_window_mask)[0][np.argmin(spo2[apnea_window_mask])]
        t_nadir_apnea = t[apnea_idx]
        spo2_nadir_apnea = spo2[apnea_idx]
    else:
        t_nadir_apnea = t[idx]
        spo2_nadir_apnea = spo2[idx]

    return {
        "t_nadir": t[idx],
        "spo2_nadir": spo2[idx],
        "in_recovery": t[idx] > t_end,
        "delay_from_end": t[idx] - t_end,
        "t_nadir_apnea": t_nadir_apnea,
        "spo2_nadir_apnea": spo2_nadir_apnea,
    }


def nadir_window_mask(
    t: np.ndarray, t_end: float, window_after: float = NADIR_WINDOW_AFTER
) -> np.ndarray:
    """Boolean mask for the nadir window (apnea + window_after seconds)."""
    return t <= t_end + window_after


# -- Physiology functions (Stage B) -------------------------------------------


def corrected_pao2_0(paco2_0: float, aa: float = 0.0) -> float:
    """Alveolar gas equation: initial PAO2 from PACO2."""
    return max(FIO2_PB_PH2O - paco2_0 / RQ - aa, 1.0)


def pao2_apnea_only(
    t: np.ndarray,
    pao2_0: float,
    pvo2: float,
    tau_washout: float,
    t_end: float,  # noqa: ARG001 - kept for API compatibility
) -> np.ndarray:
    """Exponential PAO2 decline during apnea."""
    return pvo2 + (pao2_0 - pvo2) * np.exp(-t / max(tau_washout, 0.01))


def p50_apnea_only(
    t: np.ndarray,
    paco2_0: float,
    k_co2: float,
    t_end: float,  # noqa: ARG001
) -> np.ndarray:
    """Effective P50 with linear CO2 accumulation (Bohr effect)."""
    paco2 = paco2_0 + k_co2 * t
    return P50_BASE + 0.48 * (paco2 - PACO2_NORMAL)


def odc_severinghaus(
    pao2: np.ndarray, p50_eff: np.ndarray, gamma: float
) -> np.ndarray:
    """Severinghaus ODC with variable P50 and gamma steepness."""
    pao2_virtual = pao2 * (P50_BASE / np.maximum(p50_eff, 0.01))
    pao2_adj = P50_BASE * (np.maximum(pao2_virtual, 0.01) / P50_BASE) ** gamma
    x = np.maximum(pao2_adj, 0.01)
    return 100.0 / (1.0 + 23400.0 / (x**3 + 150.0 * x))


# -- Forward model (Stage B prediction) --------------------------------------


def predict_stage_b_hold(
    t: np.ndarray,
    tau_washout: float,
    gamma: float,
    paco2_0: float,
    k_co2: float,
    b_s: float,
    mean_lag: float,
    cv: float,
    t_end: float,
    s_base: float,
    shift: float = 0.0,
) -> np.ndarray:
    """Predict SpO2 for a single hold using the v7 Stage B model.

    This is the full forward model: physiology -> ODC -> sensor convolution.
    """
    pao2_0 = corrected_pao2_0(paco2_0)
    pao2 = pao2_apnea_only(t, pao2_0, PVO2_FIXED, tau_washout, t_end)
    p50 = p50_apnea_only(t, paco2_0, k_co2, t_end)
    sa = odc_severinghaus(pao2, p50, gamma)
    eff_mean_lag = max(mean_lag + shift, EFF_LAG_MIN)
    filtered = apply_gamma_kernel(sa, eff_mean_lag, cv)
    return np.clip(s_base + b_s * (filtered - s_base), 0.0, 100.0)


def predict_spo2_v7(
    t: np.ndarray,
    hold_id: int,
    params: V7ModelParams,
) -> np.ndarray:
    """Public prediction API: predict SpO2 curve for a hold using saved v7 params.

    Args:
        t: Time array (seconds from hold start)
        hold_id: The hold to predict for
        params: Complete v7 model parameters

    Returns:
        Predicted SpO2 array
    """
    # Find per-hold params
    stage_b_ph = next(
        (ph for ph in params.stage_b_per_hold if ph.hold_id == hold_id), None
    )
    stage_a_ph = next(
        (ph for ph in params.stage_a_per_hold if ph.hold_id == hold_id), None
    )

    if stage_b_ph is None or stage_a_ph is None:
        raise ValueError(f"No parameters found for hold_id={hold_id}")

    s_base = params.s_base_values.get(hold_id, 98.0)
    t_end = t[-1] if len(t) > 0 else 0.0

    return predict_stage_b_hold(
        t=t,
        tau_washout=stage_b_ph.tau_washout,
        gamma=params.stage_b_global.gamma,
        paco2_0=stage_b_ph.paco2_0,
        k_co2=params.stage_b_global.k_co2,
        b_s=BS_FIXED,
        mean_lag=params.stage_a_global.tau_0,
        cv=CV_FIXED,
        t_end=t_end,
        s_base=s_base,
        shift=stage_a_ph.delta,
    )


def predict_spo2_v7_components(
    t: np.ndarray,
    hold_id: int,
    params: V7ModelParams,
) -> dict[str, np.ndarray]:
    """Predict SpO2 with decomposed components for visualization.

    Returns dict with keys: spo2, pao2, p50_eff, sao2_raw (pre-sensor)
    """
    stage_b_ph = next(
        (ph for ph in params.stage_b_per_hold if ph.hold_id == hold_id), None
    )
    stage_a_ph = next(
        (ph for ph in params.stage_a_per_hold if ph.hold_id == hold_id), None
    )

    if stage_b_ph is None or stage_a_ph is None:
        raise ValueError(f"No parameters found for hold_id={hold_id}")

    s_base = params.s_base_values.get(hold_id, 98.0)
    t_end = t[-1] if len(t) > 0 else 0.0

    pao2_0 = corrected_pao2_0(stage_b_ph.paco2_0)
    pao2 = pao2_apnea_only(
        t, pao2_0, PVO2_FIXED, stage_b_ph.tau_washout, t_end
    )
    p50 = p50_apnea_only(t, stage_b_ph.paco2_0, params.stage_b_global.k_co2, t_end)
    sao2_raw = odc_severinghaus(pao2, p50, params.stage_b_global.gamma)

    eff_mean_lag = max(
        params.stage_a_global.tau_0 + stage_a_ph.delta, EFF_LAG_MIN
    )
    filtered = apply_gamma_kernel(sao2_raw, eff_mean_lag, CV_FIXED)
    spo2 = np.clip(s_base + BS_FIXED * (filtered - s_base), 0.0, 100.0)

    return {
        "spo2": spo2,
        "pao2": pao2,
        "p50_eff": p50,
        "sao2_raw": sao2_raw,
    }


# -- Metrics ------------------------------------------------------------------


def compute_r2(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Compute R-squared goodness of fit."""
    ss_res = np.sum((observed - predicted) ** 2)
    ss_tot = np.sum((observed - np.mean(observed)) ** 2)
    if ss_tot == 0:
        logger.warning("R2 undefined: observed data is constant")
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def compute_rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Compute root-mean-square error."""
    return float(np.sqrt(np.mean((observed - predicted) ** 2)))
