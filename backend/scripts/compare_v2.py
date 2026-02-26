"""
Model v2 Comparison: Ablation study for SpO2 desaturation model improvements.

Tests each proposed improvement independently against the current v1.2 baseline,
then combines the winners. Also includes a descriptive sigmoid as a sanity check.

Improvements tested:
  A) Weighted loss (desaturation-region upweighting)
  B) Logit-slope kappa (replaces gamma PO2 power transform)
  C) CO2-driven Bohr (replaces time-based saturating Bohr)
  D) Fixed sensor low-pass filter (tau_sensor=8s)
  E) Beat-based sensor model (using HR data)
  F) Linear PAO2 trajectory (vs exponential)

Usage:
    cd backend && uv run python scripts/compare_v2.py
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


CO2_BOHR_BOUNDS = [
    (80, 200),  # pao2_0
    (20, 55),  # pvo2
    (20, 300),  # tau_washout
    (0.8, 3.0),  # gamma (widened: let steepness find its level)
    (15, 50),  # paco2_0
    (0.02, 0.30),  # k_co2
    (-5.0, 5.0),  # r_offset
]
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


def run_comparison(hold_id: int, label: str = ""):
    """Run comparison for a single hold."""
    print(f"\nLoading hold {hold_id} ({label}) data...")
    t, spo2, hr = load_hold_data(hold_id)
    print(f"  {len(t)} data points, t=[{t[0]:.0f}, {t[-1]:.0f}]s")
    print(f"  SpO2=[{spo2.min():.0f}, {spo2.max():.0f}]%")
    print(f"  HR=[{hr.min():.0f}, {hr.max():.0f}] bpm")

    drop_count = np.sum(spo2 < 95)
    print(f"  Points in drop region (<95%): {drop_count}/{len(spo2)}")

    all_results = []

    # 0. Descriptive sigmoid (benchmark floor)
    all_results.append(fit_model(
        t, spo2, hr, "Sigmoid (4p)", predict_sigmoid,
        SigmoidParams, SIGMOID_BOUNDS, SIGMOID_PARAM_NAMES,
    ))

    # 1. Current v1.2 baseline
    all_results.append(fit_model(
        t, spo2, hr, "v1.2 baseline", predict_v12,
        V12Params, V12_BOUNDS, V12_PARAM_NAMES,
    ))

    # A. v1.2 + weighted loss
    all_results.append(fit_model(
        t, spo2, hr, "v1.2 weighted", predict_v12,
        V12Params, V12_BOUNDS, V12_PARAM_NAMES,
        use_weighted_loss=True,
    ))

    # B. Logit-slope kappa
    all_results.append(fit_model(
        t, spo2, hr, "Kappa (logit)", predict_kappa,
        KappaParams, KAPPA_BOUNDS, KAPPA_PARAM_NAMES,
    ))

    # C. CO2-driven Bohr
    all_results.append(fit_model(
        t, spo2, hr, "CO2 Bohr", predict_co2_bohr,
        CO2BohrParams, CO2_BOHR_BOUNDS, CO2_BOHR_PARAM_NAMES,
    ))

    # BC. Kappa + CO2-Bohr combined
    all_results.append(fit_model(
        t, spo2, hr, "Kappa+CO2", predict_kappa_co2,
        KappaCO2Params, KAPPA_CO2_BOUNDS, KAPPA_CO2_PARAM_NAMES,
    ))

    # BC+A. Kappa + CO2-Bohr + weighted loss
    all_results.append(fit_model(
        t, spo2, hr, "Kappa+CO2 wtd", predict_kappa_co2,
        KappaCO2Params, KAPPA_CO2_BOUNDS, KAPPA_CO2_PARAM_NAMES,
        use_weighted_loss=True,
    ))

    print_comparison(t, spo2, all_results)


def main():
    import sys as _sys

    hold_ids = [int(x) for x in _sys.argv[1:]] if len(_sys.argv) > 1 else [6]
    labels = {4: "deep", 5: "deep", 6: "FL"}
    for hid in hold_ids:
        run_comparison(hid, labels.get(hid, ""))


if __name__ == "__main__":
    main()
