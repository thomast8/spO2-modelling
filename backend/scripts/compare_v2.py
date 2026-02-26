"""
Model v2 Comparison: Ablation study for SpO2 desaturation model improvements.

Tests each proposed improvement independently against the current v1.2 baseline,
then combines the winners. Also includes a descriptive sigmoid as a sanity check.

Improvements tested:
  A) Weighted loss (desaturation-region upweighting)
  B) Logit-slope kappa (replaces gamma PO2 power transform)
  C) CO2-driven Bohr (replaces time-based saturating Bohr)
  W) Weibull/stretched exponential PAO2 (replaces gamma)
  WG) Weibull + gamma (tests if both are needed)

Usage:
    cd backend && uv run python scripts/compare_v2.py
    cd backend && uv run python scripts/compare_v2.py --holds 6
    cd backend && uv run python scripts/compare_v2.py --holds 6 --trim 270
"""

import sqlite3
import sys
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution
from scipy.special import expit, logit  # sigmoid and logit functions

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "spo2.db"

P50_BASE = 26.6  # Fixed haemoglobin constant (mmHg)
BOHR_BETA = 0.48  # Bohr coefficient: mmHg P50 shift per mmHg CO2 (from literature)


# ── Data loading ─────────────────────────────────────────────────────────────


def load_hold_data(hold_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load (elapsed_s, spo2, hr) arrays for a hold from the DB."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT elapsed_s, spo2, hr FROM hold_data WHERE hold_id = ? ORDER BY elapsed_s",
        (hold_id,),
    ).fetchall()
    conn.close()
    if not rows:
        print(f"No data for hold_id={hold_id}")
        sys.exit(1)
    t = np.array([r[0] for r in rows], dtype=float)
    spo2 = np.array([r[1] for r in rows], dtype=float)
    hr = np.array([r[2] for r in rows], dtype=float)
    return t, spo2, hr


# ── Shared ODC ───────────────────────────────────────────────────────────────


def severinghaus_spo2(pao2: np.ndarray) -> np.ndarray:
    """Severinghaus (1979) ODC: PO2 -> SpO2 (%)."""
    x = np.maximum(pao2, 0.01)
    return np.clip(100.0 / (1.0 + 23400.0 / (x**3 + 150.0 * x)), 0.0, 100.0)


def apply_logit_slope(s_base: np.ndarray, kappa: float) -> np.ndarray:
    """Apply logit-slope steepness in saturation space.

    kappa > 1 steepens, kappa < 1 flattens. Preserves 50% crossing.
    S_true = 100 * sigmoid(kappa * logit(S_base / 100))
    """
    # Clamp to avoid logit(0) or logit(1) = +-inf
    s_frac = np.clip(s_base / 100.0, 1e-6, 1.0 - 1e-6)
    return 100.0 * expit(kappa * logit(s_frac))


def apply_sensor_lpf(
    spo2_true: np.ndarray, t: np.ndarray, tau_sensor: float
) -> np.ndarray:
    """Apply first-order IIR low-pass filter with fixed time constant."""
    out = np.empty_like(spo2_true)
    out[0] = spo2_true[0]
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        alpha = dt / (tau_sensor + dt)
        out[i] = out[i - 1] + alpha * (spo2_true[i] - out[i - 1])
    return out


def apply_beat_sensor(
    spo2_true: np.ndarray,
    t: np.ndarray,
    hr: np.ndarray,
    b_delay: float,
    b_avg: float,
) -> np.ndarray:
    """Apply beat-based delay + beat-based IIR low-pass filter.

    b_delay: transport delay in heartbeats
    b_avg: averaging window in heartbeats
    """
    hr_hz = np.maximum(hr / 60.0, 0.5)  # beats/sec, floor at 0.5 Hz

    # Beat-based delay: shift signal backward in time
    delay_s = b_delay / hr_hz
    spo2_delayed = np.interp(t - delay_s, t, spo2_true, left=spo2_true[0])

    # Beat-based IIR low-pass
    out = np.empty_like(spo2_delayed)
    out[0] = spo2_delayed[0]
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        tau_sensor = b_avg / hr_hz[i]
        alpha = dt / (tau_sensor + dt)
        out[i] = out[i - 1] + alpha * (spo2_delayed[i] - out[i - 1])
    return out


# ── Metrics ──────────────────────────────────────────────────────────────────


def compute_r_squared(observed, predicted):
    ss_res = np.sum((observed - predicted) ** 2)
    ss_tot = np.sum((observed - np.mean(observed)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def compute_drop_r_squared(observed, predicted, threshold=95.0):
    """R-squared for the desaturation region only (SpO2 < threshold)."""
    mask = observed < threshold
    if mask.sum() < 3:
        return float("nan")
    return compute_r_squared(observed[mask], predicted[mask])


def compute_rmse(observed, predicted):
    return float(np.sqrt(np.mean((observed - predicted) ** 2)))


def compute_drop_rmse(observed, predicted, threshold=95.0):
    mask = observed < threshold
    if mask.sum() < 3:
        return float("nan")
    return float(np.sqrt(np.mean((observed[mask] - predicted[mask]) ** 2)))


def check_bounds(param_values, bounds, param_names, tol=1e-3):
    at_bounds = []
    for val, (lo, hi), name in zip(param_values, bounds, param_names, strict=True):
        if abs(val - lo) < tol or abs(val - hi) < tol:
            at_bounds.append(f"{name}={'lo' if abs(val - lo) < tol else 'hi'}")
    return at_bounds


def weighted_sse(observed, predicted, weight_alpha=0.2, threshold=95.0):
    """SSE with smooth desaturation upweighting.

    weight = 1 + alpha * max(0, threshold - SpO2_observed)
    At SpO2=95: weight=1. At SpO2=70: weight=1+0.2*25=6.
    """
    weights = 1.0 + weight_alpha * np.maximum(threshold - observed, 0.0)
    return float(np.sum(weights * (observed - predicted) ** 2))


# ── Model 0: Descriptive Sigmoid (benchmark) ────────────────────────────────


@dataclass
class SigmoidParams:
    s_0: float  # plateau SpO2
    s_floor: float  # nadir SpO2
    t50: float  # inflection time (s)
    k: float  # transition width (s)


SIGMOID_BOUNDS = [
    (95, 100),  # s_0
    (40, 85),  # s_floor
    (100, 350),  # t50
    (10, 80),  # k
]
SIGMOID_PARAM_NAMES = [f.name for f in fields(SigmoidParams)]


def predict_sigmoid(t, params, **_kw):
    """Descriptive sigmoid: S(t) = s_floor + (s_0 - s_floor) / (1 + exp((t - t50) / k))."""
    return params.s_floor + (params.s_0 - params.s_floor) / (
        1.0 + np.exp((t - params.t50) / params.k)
    )


# ── Model 1: Current v1.2 baseline ──────────────────────────────────────────


@dataclass
class V12Params:
    pao2_0: float
    pvo2: float
    tau_washout: float
    gamma: float
    bohr_max: float
    tau_bohr: float
    r_offset: float


V12_BOUNDS = [
    (100, 250),  # pao2_0
    (20, 50),  # pvo2
    (50, 250),  # tau_washout
    (0.8, 2.0),  # gamma
    (2.0, 15.0),  # bohr_max
    (40, 250),  # tau_bohr
    (-3.0, 3.0),  # r_offset
]
V12_PARAM_NAMES = [f.name for f in fields(V12Params)]


def predict_v12(t, params, **_kw):
    """Current v1.2 model: exponential washout + saturating Bohr + gamma + Severinghaus."""
    pao2 = params.pvo2 + (params.pao2_0 - params.pvo2) * np.exp(
        -t / max(params.tau_washout, 0.01)
    )
    p50_eff = P50_BASE + params.bohr_max * (
        1.0 - np.exp(-t / max(params.tau_bohr, 0.01))
    )
    pao2_virtual = pao2 * (P50_BASE / p50_eff)
    pao2_adj = P50_BASE * (
        (np.maximum(pao2_virtual, 0.01) / P50_BASE) ** params.gamma
    )
    spo2 = params.r_offset + severinghaus_spo2(pao2_adj)
    return np.clip(spo2, 0.0, 100.0)


# ── Model A: v1.2 + weighted loss ───────────────────────────────────────────
# Same model as v1.2; only the objective function changes. Handled in fit_model().


# ── Model B: Logit-slope kappa (replaces gamma) ─────────────────────────────


@dataclass
class KappaParams:
    pao2_0: float
    pvo2: float
    tau_washout: float
    kappa: float  # logit-slope steepness (replaces gamma)
    bohr_max: float
    tau_bohr: float
    r_offset: float


KAPPA_BOUNDS = [
    (100, 250),  # pao2_0
    (20, 50),  # pvo2
    (50, 250),  # tau_washout
    (0.5, 3.5),  # kappa (1.0 = standard, >1 steeper; widened upper)
    (2.0, 15.0),  # bohr_max
    (40, 250),  # tau_bohr
    (-3.0, 3.0),  # r_offset
]
KAPPA_PARAM_NAMES = [f.name for f in fields(KappaParams)]


def predict_kappa(t, params, **_kw):
    """Severinghaus with logit-slope steepness (decoupled from Bohr)."""
    pao2 = params.pvo2 + (params.pao2_0 - params.pvo2) * np.exp(
        -t / max(params.tau_washout, 0.01)
    )
    p50_eff = P50_BASE + params.bohr_max * (
        1.0 - np.exp(-t / max(params.tau_bohr, 0.01))
    )
    # Bohr shift via virtual PO2 (no gamma entanglement)
    pao2_virtual = pao2 * (P50_BASE / p50_eff)
    s_base = severinghaus_spo2(pao2_virtual)
    # Steepness in saturation space
    s_true = apply_logit_slope(s_base, params.kappa)
    spo2 = params.r_offset + s_true
    return np.clip(spo2, 0.0, 100.0)


# ── Model C: CO2-driven Bohr (replaces time-based) ──────────────────────────


@dataclass
class CO2BohrParams:
    pao2_0: float
    pvo2: float
    tau_washout: float
    gamma: float
    paco2_0: float  # initial PaCO2 (mmHg; low after hyperventilation)
    k_co2: float  # CO2 rise rate (mmHg/s)
    r_offset: float


CO2_BOHR_BOUNDS_BY_TYPE = {
    "FL": [
        (100, 250),    # pao2_0: full lungs, hyperventilated
        (20, 55),      # pvo2
        (20, 250),     # tau_washout: slowest, largest O2 reserve
        (0.8, 3.0),   # gamma
        (15, 45),      # paco2_0: often hyperventilated
        (0.02, 0.25),  # k_co2
        (-5.0, 5.0),   # r_offset
    ],
    "FRC": [
        (80, 120),     # pao2_0: normal lung volume
        (20, 50),      # pvo2
        (20, 100),     # tau_washout: faster than FL
        (0.8, 3.0),   # gamma
        (20, 45),      # paco2_0
        (0.02, 0.25),  # k_co2
        (-3.0, 3.0),   # r_offset
    ],
    "RV": [
        (70, 110),     # pao2_0: minimal lung volume
        (20, 50),      # pvo2
        (10, 80),      # tau_washout: fastest
        (0.8, 3.0),   # gamma
        (25, 45),      # paco2_0: less hyperventilation
        (0.02, 0.25),  # k_co2
        (-3.0, 3.0),   # r_offset
    ],
}
CO2_BOHR_PARAM_NAMES = [f.name for f in fields(CO2BohrParams)]


def predict_co2_bohr(t, params, **_kw):
    """Exponential washout + CO2-driven Bohr + gamma + Severinghaus."""
    pao2 = params.pvo2 + (params.pao2_0 - params.pvo2) * np.exp(
        -t / max(params.tau_washout, 0.01)
    )
    # Linear CO2 rise -> Bohr shift
    paco2 = params.paco2_0 + params.k_co2 * t
    p50_eff = P50_BASE + BOHR_BETA * (paco2 - 40.0)
    # Virtual PO2 + gamma
    pao2_virtual = pao2 * (P50_BASE / np.maximum(p50_eff, 1.0))
    pao2_adj = P50_BASE * (
        (np.maximum(pao2_virtual, 0.01) / P50_BASE) ** params.gamma
    )
    spo2 = params.r_offset + severinghaus_spo2(pao2_adj)
    return np.clip(spo2, 0.0, 100.0)


# ── Model D: v1.2 + fixed sensor LPF ────────────────────────────────────────
# Same params as v1.2; sensor filter applied post-hoc with fixed tau_sensor.

FIXED_TAU_SENSOR = 8.0  # seconds (typical pulse ox averaging window)


def predict_v12_sensor_fixed(t, params, **_kw):
    """v1.2 + fixed first-order low-pass filter (tau=8s)."""
    spo2_true = predict_v12(t, params)
    return apply_sensor_lpf(spo2_true, t, FIXED_TAU_SENSOR)


# ── Model E: v1.2 + beat-based sensor ───────────────────────────────────────


@dataclass
class V12BeatParams:
    pao2_0: float
    pvo2: float
    tau_washout: float
    gamma: float
    bohr_max: float
    tau_bohr: float
    r_offset: float
    b_delay: float  # transport delay (beats)
    b_avg: float  # averaging window (beats)


V12_BEAT_BOUNDS = [
    (100, 250),  # pao2_0
    (20, 50),  # pvo2
    (50, 250),  # tau_washout
    (0.8, 2.0),  # gamma
    (2.0, 15.0),  # bohr_max
    (40, 250),  # tau_bohr
    (-3.0, 3.0),  # r_offset
    (3, 25),  # b_delay (beats; ~5-20s at 60bpm)
    (2, 20),  # b_avg (beats; ~3-15s at 60bpm)
]
V12_BEAT_PARAM_NAMES = [f.name for f in fields(V12BeatParams)]


def predict_v12_beat(t, params, hr=None, **_kw):
    """v1.2 physiology + beat-based delay and averaging."""
    pao2 = params.pvo2 + (params.pao2_0 - params.pvo2) * np.exp(
        -t / max(params.tau_washout, 0.01)
    )
    p50_eff = P50_BASE + params.bohr_max * (
        1.0 - np.exp(-t / max(params.tau_bohr, 0.01))
    )
    pao2_virtual = pao2 * (P50_BASE / p50_eff)
    pao2_adj = P50_BASE * (
        (np.maximum(pao2_virtual, 0.01) / P50_BASE) ** params.gamma
    )
    spo2_true = np.clip(params.r_offset + severinghaus_spo2(pao2_adj), 0.0, 100.0)

    if hr is None:
        return spo2_true
    return apply_beat_sensor(spo2_true, t, hr, params.b_delay, params.b_avg)


# ── Model F: Linear PAO2 + gamma + time-Bohr ────────────────────────────────


@dataclass
class LinearPAO2Params:
    pao2_0: float
    k_o2: float  # depletion rate (mmHg/s)
    gamma: float
    bohr_max: float
    tau_bohr: float
    r_offset: float


LINEAR_PAO2_BOUNDS = [
    (100, 250),  # pao2_0
    (0.05, 1.0),  # k_o2 (mmHg/s; ~3-60 mmHg/min)
    (0.8, 2.0),  # gamma
    (2.0, 15.0),  # bohr_max
    (40, 250),  # tau_bohr
    (-3.0, 3.0),  # r_offset
]
LINEAR_PAO2_PARAM_NAMES = [f.name for f in fields(LinearPAO2Params)]


def predict_linear_pao2(t, params, **_kw):
    """Linear PAO2 decline + saturating Bohr + gamma + Severinghaus."""
    pao2 = np.maximum(params.pao2_0 - params.k_o2 * t, 0.01)
    p50_eff = P50_BASE + params.bohr_max * (
        1.0 - np.exp(-t / max(params.tau_bohr, 0.01))
    )
    pao2_virtual = pao2 * (P50_BASE / p50_eff)
    pao2_adj = P50_BASE * (
        (np.maximum(pao2_virtual, 0.01) / P50_BASE) ** params.gamma
    )
    spo2 = params.r_offset + severinghaus_spo2(pao2_adj)
    return np.clip(spo2, 0.0, 100.0)


# ── Model BC: Logit-slope + CO2-Bohr combined ───────────────────────────────


@dataclass
class KappaCO2Params:
    pao2_0: float
    pvo2: float
    tau_washout: float
    kappa: float
    paco2_0: float
    k_co2: float
    r_offset: float


KAPPA_CO2_BOUNDS = [
    (80, 200),  # pao2_0 (physiological FL: room air ~100-150, some pre-ox ~200)
    (20, 55),  # pvo2 (physiological range)
    (20, 300),  # tau_washout
    (0.5, 3.0),  # kappa (capped: prevent step-function overfitting)
    (15, 50),  # paco2_0 (post-hypervent ~20, normal ~40, mild hypercapnia ~50)
    (0.02, 0.30),  # k_co2
    (-5.0, 5.0),  # r_offset
]
KAPPA_CO2_PARAM_NAMES = [f.name for f in fields(KappaCO2Params)]


def predict_kappa_co2(t, params, **_kw):
    """Logit-slope steepness + CO2-driven Bohr (both improvements combined)."""
    pao2 = params.pvo2 + (params.pao2_0 - params.pvo2) * np.exp(
        -t / max(params.tau_washout, 0.01)
    )
    paco2 = params.paco2_0 + params.k_co2 * t
    p50_eff = P50_BASE + BOHR_BETA * (paco2 - 40.0)
    pao2_virtual = pao2 * (P50_BASE / np.maximum(p50_eff, 1.0))
    s_base = severinghaus_spo2(pao2_virtual)
    s_true = apply_logit_slope(s_base, params.kappa)
    spo2 = params.r_offset + s_true
    return np.clip(spo2, 0.0, 100.0)


# ── Model W: Weibull PAO2 (replaces gamma with eta) ─────────────────────────


@dataclass
class WeibullParams:
    pao2_0: float
    pvo2: float
    tau_washout: float
    eta: float  # Weibull shape: 1.0=exponential, >1=slow-then-fast
    paco2_0: float
    k_co2: float
    r_offset: float


WEIBULL_BOUNDS_BY_TYPE = {
    "FL": [
        (100, 250),    # pao2_0
        (20, 55),      # pvo2
        (20, 250),     # tau_washout
        (0.5, 5.0),   # eta
        (15, 45),      # paco2_0
        (0.02, 0.25),  # k_co2
        (-5.0, 5.0),   # r_offset
    ],
    "FRC": [
        (80, 120),     # pao2_0
        (20, 50),      # pvo2
        (20, 100),     # tau_washout
        (0.5, 5.0),   # eta
        (20, 45),      # paco2_0
        (0.02, 0.25),  # k_co2
        (-3.0, 3.0),   # r_offset
    ],
    "RV": [
        (70, 110),     # pao2_0
        (20, 50),      # pvo2
        (10, 80),      # tau_washout
        (0.5, 5.0),   # eta
        (25, 45),      # paco2_0
        (0.02, 0.25),  # k_co2
        (-3.0, 3.0),   # r_offset
    ],
}
WEIBULL_PARAM_NAMES = [f.name for f in fields(WeibullParams)]


def predict_weibull(t, params, **_kw):
    """Weibull PAO2 washout + CO2-driven Bohr + Severinghaus (no gamma)."""
    pao2 = params.pvo2 + (params.pao2_0 - params.pvo2) * np.exp(
        -(t / max(params.tau_washout, 0.01)) ** params.eta
    )
    paco2 = params.paco2_0 + params.k_co2 * t
    p50_eff = P50_BASE + BOHR_BETA * (paco2 - 40.0)
    pao2_virtual = pao2 * (P50_BASE / np.maximum(p50_eff, 1.0))
    # No gamma — standard Severinghaus on virtual PO2
    spo2 = params.r_offset + severinghaus_spo2(pao2_virtual)
    return np.clip(spo2, 0.0, 100.0)


# ── Model WG: Weibull + gamma (tests if both are needed) ───────────────────


@dataclass
class WeibullGammaParams:
    pao2_0: float
    pvo2: float
    tau_washout: float
    eta: float
    gamma: float
    paco2_0: float
    k_co2: float
    r_offset: float


WEIBULL_GAMMA_BOUNDS = [
    (80, 200),     # pao2_0
    (20, 55),      # pvo2
    (20, 300),     # tau_washout
    (0.5, 5.0),   # eta
    (0.8, 3.0),   # gamma
    (15, 50),      # paco2_0
    (0.02, 0.30),  # k_co2
    (-5.0, 5.0),   # r_offset
]
WEIBULL_GAMMA_PARAM_NAMES = [f.name for f in fields(WeibullGammaParams)]


def predict_weibull_gamma(t, params, **_kw):
    """Weibull PAO2 washout + CO2-driven Bohr + gamma + Severinghaus."""
    pao2 = params.pvo2 + (params.pao2_0 - params.pvo2) * np.exp(
        -(t / max(params.tau_washout, 0.01)) ** params.eta
    )
    paco2 = params.paco2_0 + params.k_co2 * t
    p50_eff = P50_BASE + BOHR_BETA * (paco2 - 40.0)
    pao2_virtual = pao2 * (P50_BASE / np.maximum(p50_eff, 1.0))
    pao2_adj = P50_BASE * (
        (np.maximum(pao2_virtual, 0.01) / P50_BASE) ** params.gamma
    )
    spo2 = params.r_offset + severinghaus_spo2(pao2_adj)
    return np.clip(spo2, 0.0, 100.0)


# ── Model D: CO2-O2 gas displacement coupling ──────────────────────────────


@dataclass
class CO2DisplaceParams:
    pao2_0: float
    pvo2: float
    tau_washout: float
    gamma: float
    paco2_0: float
    k_co2: float
    lambda_rq: float  # CO2-O2 displacement factor (~1/RQ, ~1.25)
    r_offset: float


CO2_DISPLACE_BOUNDS = [
    (80, 200),     # pao2_0
    (20, 55),      # pvo2
    (20, 300),     # tau_washout
    (0.8, 3.0),   # gamma
    (15, 50),      # paco2_0
    (0.02, 0.30),  # k_co2
    (0.5, 3.0),   # lambda_rq (physiological ~1/RQ = 1.0-1.25)
    (-5.0, 5.0),   # r_offset
]
CO2_DISPLACE_PARAM_NAMES = [f.name for f in fields(CO2DisplaceParams)]


def predict_co2_displace(t, params, **_kw):
    """Exponential washout + CO2-driven Bohr + CO2-O2 displacement + gamma."""
    pao2 = params.pvo2 + (params.pao2_0 - params.pvo2) * np.exp(
        -t / max(params.tau_washout, 0.01)
    )
    paco2 = params.paco2_0 + params.k_co2 * t
    # CO2 displaces O2 in alveolar gas (partial pressure bookkeeping)
    pao2_eff = pao2 - params.lambda_rq * (paco2 - params.paco2_0)
    pao2_eff = np.maximum(pao2_eff, 0.01)
    # Bohr shift on the effective PAO2
    p50_eff = P50_BASE + BOHR_BETA * (paco2 - 40.0)
    pao2_virtual = pao2_eff * (P50_BASE / np.maximum(p50_eff, 1.0))
    pao2_adj = P50_BASE * (
        (np.maximum(pao2_virtual, 0.01) / P50_BASE) ** params.gamma
    )
    spo2 = params.r_offset + severinghaus_spo2(pao2_adj)
    return np.clip(spo2, 0.0, 100.0)


# ── Model WD: Weibull + CO2-O2 displacement (no gamma) ─────────────────────


@dataclass
class WeibullDisplaceParams:
    pao2_0: float
    pvo2: float
    tau_washout: float
    eta: float
    paco2_0: float
    k_co2: float
    lambda_rq: float
    r_offset: float


WEIBULL_DISPLACE_BOUNDS = [
    (80, 200),     # pao2_0
    (20, 55),      # pvo2
    (20, 300),     # tau_washout
    (0.5, 5.0),   # eta
    (15, 50),      # paco2_0
    (0.02, 0.30),  # k_co2
    (0.5, 3.0),   # lambda_rq
    (-5.0, 5.0),   # r_offset
]
WEIBULL_DISPLACE_PARAM_NAMES = [f.name for f in fields(WeibullDisplaceParams)]


def predict_weibull_displace(t, params, **_kw):
    """Weibull washout + CO2 displacement + Bohr + Severinghaus (no gamma)."""
    pao2 = params.pvo2 + (params.pao2_0 - params.pvo2) * np.exp(
        -(t / max(params.tau_washout, 0.01)) ** params.eta
    )
    paco2 = params.paco2_0 + params.k_co2 * t
    pao2_eff = np.maximum(pao2 - params.lambda_rq * (paco2 - params.paco2_0), 0.01)
    p50_eff = P50_BASE + BOHR_BETA * (paco2 - 40.0)
    pao2_virtual = pao2_eff * (P50_BASE / np.maximum(p50_eff, 1.0))
    spo2 = params.r_offset + severinghaus_spo2(pao2_virtual)
    return np.clip(spo2, 0.0, 100.0)


# ── Model S: Shunt mixing (replaces gamma with s_max + tau_shunt) ───────────
#
# Physiology: increasing shunt fraction (absorption atelectasis, VQ mismatch)
# mixes venous blood into the arterial stream. This naturally creates a
# late "cliff" without distorting the ODC.
#
# S_arterial(t) = (1 - s(t)) * S_endcap(t) + s(t) * S_venous
# s(t) = s_max * (1 - exp(-t / tau_shunt))  [saturating shunt increase]
# S_venous = Severinghaus(pvo2)  [derived, not fitted separately]


@dataclass
class ShuntParams:
    pao2_0: float
    pvo2: float
    tau_washout: float
    paco2_0: float
    k_co2: float
    s_max: float      # maximum shunt fraction (0-1)
    tau_shunt: float   # shunt development time constant (s)
    r_offset: float


SHUNT_BOUNDS = [
    (80, 200),     # pao2_0
    (20, 55),      # pvo2 (also determines S_venous via Severinghaus)
    (20, 300),     # tau_washout
    (15, 50),      # paco2_0
    (0.02, 0.30),  # k_co2
    (0.01, 0.60),  # s_max (1-60% shunt; severe atelectasis can reach 30-50%)
    (20, 400),     # tau_shunt (shunt develops over minutes)
    (-5.0, 5.0),   # r_offset
]
SHUNT_PARAM_NAMES = [f.name for f in fields(ShuntParams)]


def predict_shunt(t, params, **_kw):
    """Exponential washout + CO2 Bohr + shunt mixing (no gamma)."""
    pao2 = params.pvo2 + (params.pao2_0 - params.pvo2) * np.exp(
        -t / max(params.tau_washout, 0.01)
    )
    paco2 = params.paco2_0 + params.k_co2 * t
    p50_eff = P50_BASE + BOHR_BETA * (paco2 - 40.0)
    pao2_virtual = pao2 * (P50_BASE / np.maximum(p50_eff, 1.0))
    # End-capillary saturation (no gamma — standard Severinghaus)
    s_cap = severinghaus_spo2(pao2_virtual)
    # Venous saturation (constant, derived from pvo2)
    s_ven = severinghaus_spo2(params.pvo2)
    # Time-varying shunt fraction
    shunt = params.s_max * (1.0 - np.exp(-t / max(params.tau_shunt, 0.01)))
    # Arterial = weighted mix
    spo2 = params.r_offset + (1.0 - shunt) * s_cap + shunt * s_ven
    return np.clip(spo2, 0.0, 100.0)


# ── Model SG: Shunt + gamma (tests if both needed) ─────────────────────────


@dataclass
class ShuntGammaParams:
    pao2_0: float
    pvo2: float
    tau_washout: float
    gamma: float
    paco2_0: float
    k_co2: float
    s_max: float
    tau_shunt: float
    r_offset: float


SHUNT_GAMMA_BOUNDS = [
    (80, 200),     # pao2_0
    (20, 55),      # pvo2
    (20, 300),     # tau_washout
    (0.8, 3.0),   # gamma
    (15, 50),      # paco2_0
    (0.02, 0.30),  # k_co2
    (0.01, 0.60),  # s_max
    (20, 400),     # tau_shunt
    (-5.0, 5.0),   # r_offset
]
SHUNT_GAMMA_PARAM_NAMES = [f.name for f in fields(ShuntGammaParams)]


def predict_shunt_gamma(t, params, **_kw):
    """Exponential washout + CO2 Bohr + gamma + shunt mixing."""
    pao2 = params.pvo2 + (params.pao2_0 - params.pvo2) * np.exp(
        -t / max(params.tau_washout, 0.01)
    )
    paco2 = params.paco2_0 + params.k_co2 * t
    p50_eff = P50_BASE + BOHR_BETA * (paco2 - 40.0)
    pao2_virtual = pao2 * (P50_BASE / np.maximum(p50_eff, 1.0))
    pao2_adj = P50_BASE * (
        (np.maximum(pao2_virtual, 0.01) / P50_BASE) ** params.gamma
    )
    s_cap = severinghaus_spo2(pao2_adj)
    s_ven = severinghaus_spo2(params.pvo2)
    shunt = params.s_max * (1.0 - np.exp(-t / max(params.tau_shunt, 0.01)))
    spo2 = params.r_offset + (1.0 - shunt) * s_cap + shunt * s_ven
    return np.clip(spo2, 0.0, 100.0)


# ── Model V2: Full combination ──────────────────────────────────────────────


@dataclass
class V2Params:
    pao2_0: float
    pvo2: float
    tau_washout: float
    kappa: float
    paco2_0: float
    k_co2: float
    r_offset: float
    b_delay: float
    b_avg: float


V2_BOUNDS = [
    (100, 250),  # pao2_0
    (20, 65),  # pvo2
    (30, 250),  # tau_washout
    (0.5, 3.5),  # kappa
    (15, 45),  # paco2_0
    (0.02, 0.25),  # k_co2
    (-3.0, 3.0),  # r_offset
    (3, 25),  # b_delay
    (2, 20),  # b_avg
]
V2_PARAM_NAMES = [f.name for f in fields(V2Params)]


def predict_v2(t, params, hr=None, **_kw):
    """Full v2: kappa + CO2 Bohr + beat-based sensor."""
    pao2 = params.pvo2 + (params.pao2_0 - params.pvo2) * np.exp(
        -t / max(params.tau_washout, 0.01)
    )
    paco2 = params.paco2_0 + params.k_co2 * t
    p50_eff = P50_BASE + BOHR_BETA * (paco2 - 40.0)
    pao2_virtual = pao2 * (P50_BASE / np.maximum(p50_eff, 1.0))
    s_base = severinghaus_spo2(pao2_virtual)
    s_true = apply_logit_slope(s_base, params.kappa)
    spo2_true = np.clip(params.r_offset + s_true, 0.0, 100.0)

    if hr is None:
        return spo2_true
    return apply_beat_sensor(spo2_true, t, hr, params.b_delay, params.b_avg)


# ── Fitting engine ───────────────────────────────────────────────────────────


def fit_model(
    t,
    spo2,
    hr,
    model_name,
    predict_fn,
    param_cls,
    bounds,
    param_names,
    use_weighted_loss=False,
):
    """Fit a model using differential evolution. Returns result dict."""
    print(f"\n{'='*60}")
    print(f"Fitting: {model_name} ({len(bounds)} params)")
    if use_weighted_loss:
        print("  (using weighted loss: desaturation upweighted)")
    print(f"{'='*60}")

    def objective(arr):
        params = param_cls(*arr)
        pred = predict_fn(t, params, hr=hr)
        if use_weighted_loss:
            return weighted_sse(spo2, pred)
        return float(np.sum((spo2 - pred) ** 2))

    result = differential_evolution(
        objective,
        bounds,
        maxiter=5000,
        seed=42,
        tol=1e-12,
        polish=True,
        popsize=60,
        mutation=(0.5, 1.5),
        recombination=0.9,
    )

    params = param_cls(*result.x)
    pred = predict_fn(t, params, hr=hr)
    r2 = compute_r_squared(spo2, pred)
    r2_drop = compute_drop_r_squared(spo2, pred)
    rmse = compute_rmse(spo2, pred)
    rmse_drop = compute_drop_rmse(spo2, pred)
    at_bounds = check_bounds(result.x, bounds, param_names)

    print(f"  Converged:    {result.success}")
    print(f"  R² (overall): {r2:.6f}")
    print(f"  R² (drop):    {r2_drop:.6f}")
    print(f"  RMSE:         {rmse:.4f}")
    print(f"  RMSE (drop):  {rmse_drop:.4f}")
    print(f"  At bounds ({len(at_bounds)}/{len(bounds)}): {at_bounds or 'none'}")
    print(f"  Parameters:")
    for name, val, (lo, hi) in zip(param_names, result.x, bounds, strict=True):
        marker = " <<<" if any(name in ab for ab in at_bounds) else ""
        print(f"    {name:>14s} = {val:10.4f}  [{lo:>8.2f}, {hi:>8.2f}]{marker}")

    return {
        "name": model_name,
        "pred": pred,
        "r2": r2,
        "r2_drop": r2_drop,
        "rmse": rmse,
        "rmse_drop": rmse_drop,
        "at_bounds": at_bounds,
        "n_params": len(bounds),
        "params": params,
    }


# ── Comparison output ────────────────────────────────────────────────────────


def print_comparison(t, spo2, results):
    """Print side-by-side comparison table."""
    print(f"\n{'='*100}")
    print("COMPARISON SUMMARY")
    print(f"{'='*100}")

    # Shorten names for table
    names = [r["name"] for r in results]
    short = [n[:16] for n in names]

    header = f"  {'Metric':<20s}" + "".join(f" {s:>16s}" for s in short)
    print(header)
    print(f"  {'-'*20}" + " ".join(f"{'-'*16}" for _ in short))

    for metric, key, fmt in [
        ("R² (overall)", "r2", ".6f"),
        ("R² (drop <95%)", "r2_drop", ".6f"),
        ("RMSE (overall)", "rmse", ".4f"),
        ("RMSE (drop <95%)", "rmse_drop", ".4f"),
        ("Num params", "n_params", "d"),
        ("At bounds", None, "d"),
    ]:
        row = f"  {metric:<20s}"
        for r in results:
            if key is None:  # at_bounds special case
                val = len(r["at_bounds"])
            else:
                val = r[key]
            row += f" {val:>16{fmt}}"
        print(row)

    # Residuals at key time points
    key_times = [100, 200, 250, 300, 350]
    print(f"\n  Residuals (obs - pred) at key times:")
    header = f"  {'t(s)':<6s} {'Obs':>5s}"
    for s in short:
        header += f" {s:>16s}"
    print(header)
    for tk in key_times:
        idx = np.argmin(np.abs(t - tk))
        obs = spo2[idx]
        row = f"  {t[idx]:<6.0f} {obs:>5.0f}"
        for r in results:
            res = obs - r["pred"][idx]
            row += f" {res:>+16.2f}"
        print(row)


# ── Main ─────────────────────────────────────────────────────────────────────


def run_comparison(
    hold_id: int,
    label: str = "",
    trim_t: float | None = None,
    hold_type: str = "FL",
):
    """Run comparison for a single hold."""
    print(f"\n{'#'*80}")
    print(f"  Hold {hold_id} ({label}) — type: {hold_type}")
    print(f"{'#'*80}")
    t, spo2, hr = load_hold_data(hold_id)

    if trim_t is not None:
        mask = t <= trim_t
        n_before = len(t)
        t, spo2, hr = t[mask], spo2[mask], hr[mask]
        print(f"  TRIMMED at {trim_t}s: {n_before} -> {len(t)} data points")

    # Warn about pulse ox floor (device floors at 40%)
    if spo2.min() <= 42:
        print(f"  *** WARNING: min SpO2={spo2.min():.0f}% — at/near pulse ox floor (40%) ***")
        print(f"  *** Data below ~42% is censored and unreliable ***")

    print(f"  {len(t)} data points, t=[{t[0]:.0f}, {t[-1]:.0f}]s")
    print(f"  SpO2=[{spo2.min():.0f}, {spo2.max():.0f}]%")
    print(f"  HR=[{hr.min():.0f}, {hr.max():.0f}] bpm")

    drop_count = np.sum(spo2 < 95)
    print(f"  Points in drop region (<95%): {drop_count}/{len(spo2)}")

    # Select per-hold-type bounds
    co2_bohr_bounds = CO2_BOHR_BOUNDS_BY_TYPE[hold_type]
    weibull_bounds = WEIBULL_BOUNDS_BY_TYPE[hold_type]

    all_results = []

    # 0. Descriptive sigmoid (benchmark floor)
    all_results.append(fit_model(
        t, spo2, hr, "Sigmoid (4p)", predict_sigmoid,
        SigmoidParams, SIGMOID_BOUNDS, SIGMOID_PARAM_NAMES,
    ))

    # C. CO2-driven Bohr (current production model)
    all_results.append(fit_model(
        t, spo2, hr, "CO2 Bohr", predict_co2_bohr,
        CO2BohrParams, co2_bohr_bounds, CO2_BOHR_PARAM_NAMES,
    ))

    # C+A. CO2-driven Bohr + weighted loss
    all_results.append(fit_model(
        t, spo2, hr, "CO2 Bohr wtd", predict_co2_bohr,
        CO2BohrParams, co2_bohr_bounds, CO2_BOHR_PARAM_NAMES,
        use_weighted_loss=True,
    ))

    # W. Weibull PAO2 (replaces gamma with eta)
    all_results.append(fit_model(
        t, spo2, hr, "Weibull (eta)", predict_weibull,
        WeibullParams, weibull_bounds, WEIBULL_PARAM_NAMES,
    ))

    # W+A. Weibull + weighted loss
    all_results.append(fit_model(
        t, spo2, hr, "Weibull wtd", predict_weibull,
        WeibullParams, weibull_bounds, WEIBULL_PARAM_NAMES,
        use_weighted_loss=True,
    ))

    print_comparison(t, spo2, all_results)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="SpO2 model ablation study")
    parser.add_argument(
        "--holds", type=int, nargs="+", default=[6],
        help="Hold IDs to fit (default: 6)",
    )
    parser.add_argument(
        "--trim", type=float, default=None,
        help="Trim data at this elapsed time (seconds). For hand-lift diagnostic: --trim 270",
    )
    args = parser.parse_args()

    labels = {1: "FL-nowarmup", 2: "FRC", 3: "RV", 4: "RV-floor", 5: "FRC", 6: "FL"}
    hold_types = {1: "FL", 2: "FRC", 3: "RV", 4: "RV", 5: "FRC", 6: "FL"}
    for hid in args.holds:
        run_comparison(
            hid, labels.get(hid, ""),
            trim_t=args.trim,
            hold_type=hold_types.get(hid, "FL"),
        )


if __name__ == "__main__":
    main()
