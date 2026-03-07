"""
Compare Hill vs Kelman ODC for FL hold 6 data.

Standalone script — loads data directly from SQLite, fits both models,
prints side-by-side comparison.

Usage:
    cd backend && uv run python scripts/compare_odc.py
"""

import sqlite3
import sys
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

DB_PATH = Path(__file__).resolve().parents[3] / "data" / "spo2.db"

P50_BASE = 26.6  # Baseline P50 (mmHg), fixed haemoglobin constant


# ── Data loading ─────────────────────────────────────────────────────────────


def load_hold_data(hold_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Load (elapsed_s, spo2) arrays for a hold from the DB."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT elapsed_s, spo2 FROM hold_data WHERE hold_id = ? ORDER BY elapsed_s",
        (hold_id,),
    ).fetchall()
    conn.close()
    if not rows:
        print(f"No data for hold_id={hold_id}")
        sys.exit(1)
    t = np.array([r[0] for r in rows], dtype=float)
    spo2 = np.array([r[1] for r in rows], dtype=float)
    return t, spo2


# ── Hill model (current) ────────────────────────────────────────────────────


@dataclass
class HillParams:
    pao2_0: float
    pvo2: float
    tau_washout: float
    n: float
    bohr_max: float
    tau_bohr: float
    lag: float
    r_offset: float


HILL_BOUNDS_CURRENT = [
    (100, 200),    # pao2_0
    (25, 50),      # pvo2
    (50, 250),     # tau_washout
    (2.6, 3.2),    # n
    (2.0, 10.0),   # bohr_max
    (60, 180),     # tau_bohr
    (5, 45),       # lag
    (-3.0, 3.0),   # r_offset
]

HILL_BOUNDS_WIDE = [
    (80, 400),     # pao2_0 — hyperventilated FL can be very high
    (10, 50),      # pvo2 — prolonged apnea can drop low
    (30, 400),     # tau_washout
    (2.0, 5.0),    # n — wider Hill coefficient range
    (1.0, 30.0),   # bohr_max
    (30, 400),     # tau_bohr
    (0, 80),       # lag
    (-5.0, 5.0),   # r_offset
]

HILL_PARAM_NAMES = [f.name for f in fields(HillParams)]


def hill_spo2(pao2, p50, n):
    pao2 = np.maximum(pao2, 0.01)
    p50 = np.maximum(p50, 0.01)
    return 100.0 * (pao2**n) / (pao2**n + p50**n)


def predict_hill(t, params):
    t_eff = np.maximum(t - params.lag, 0.0)
    pao2 = params.pvo2 + (params.pao2_0 - params.pvo2) * np.exp(
        -t_eff / max(params.tau_washout, 0.01)
    )
    p50_eff = P50_BASE + params.bohr_max * (
        1.0 - np.exp(-t_eff / max(params.tau_bohr, 0.01))
    )
    spo2 = params.r_offset + hill_spo2(pao2, p50_eff, params.n)
    return np.clip(spo2, 0.0, 100.0)


# ── Kelman model ─────────────────────────────────────────────────────────────

# Standard Kelman (1966) rational polynomial coefficients
# SpO2 = 100 * (a1*x + a2*x² + a3*x³ + x⁴) / (a4 + a5*x + a6*x² + a7*x³ + x⁴)
# where x = PO2 (mmHg)
KELMAN_A1 = -8532.2289
KELMAN_A2 = 2121.4010
KELMAN_A3 = -67.073989
KELMAN_A4 = 935960.87
KELMAN_A5 = -31346.258
KELMAN_A6 = 2396.1674
KELMAN_A7 = -67.104406


@dataclass
class KelmanParams:
    pao2_0: float
    pvo2: float
    tau_washout: float
    # No n — the Kelman polynomial has its own shape
    bohr_max: float
    tau_bohr: float
    lag: float
    r_offset: float


KELMAN_BOUNDS_CURRENT = [
    (100, 200),    # pao2_0
    (25, 50),      # pvo2
    (50, 250),     # tau_washout
    (2.0, 10.0),   # bohr_max
    (60, 180),     # tau_bohr
    (5, 45),       # lag
    (-3.0, 3.0),   # r_offset
]

KELMAN_BOUNDS_WIDE = [
    (80, 400),     # pao2_0
    (10, 50),      # pvo2
    (30, 400),     # tau_washout
    (1.0, 30.0),   # bohr_max
    (30, 400),     # tau_bohr
    (0, 80),       # lag
    (-5.0, 5.0),   # r_offset
]

KELMAN_PARAM_NAMES = [f.name for f in fields(KelmanParams)]


def kelman_spo2(pao2):
    """Kelman (1966) ODC: PO2 → SpO2 (%).

    Valid for PO2 > ~1 mmHg. Returns 0 for very low PO2.
    """
    x = np.maximum(pao2, 0.01)
    x2 = x * x
    x3 = x2 * x
    x4 = x3 * x
    numer = KELMAN_A1 * x + KELMAN_A2 * x2 + KELMAN_A3 * x3 + x4
    denom = KELMAN_A4 + KELMAN_A5 * x + KELMAN_A6 * x2 + KELMAN_A7 * x3 + x4
    sat = 100.0 * numer / denom
    return np.clip(sat, 0.0, 100.0)


def predict_kelman(t, params):
    """Predict SpO2 using Kelman ODC with virtual PO2 Bohr scaling."""
    t_eff = np.maximum(t - params.lag, 0.0)
    pao2 = params.pvo2 + (params.pao2_0 - params.pvo2) * np.exp(
        -t_eff / max(params.tau_washout, 0.01)
    )
    # Saturating Bohr effect → effective P50
    p50_eff = P50_BASE + params.bohr_max * (
        1.0 - np.exp(-t_eff / max(params.tau_bohr, 0.01))
    )
    # Virtual PO2 scaling: shift the curve rightward by scaling PO2
    # If P50 increases, the same PO2 produces lower saturation
    pao2_virtual = pao2 * (P50_BASE / p50_eff)
    spo2 = params.r_offset + kelman_spo2(pao2_virtual)
    return np.clip(spo2, 0.0, 100.0)


# ── Severinghaus model ───────────────────────────────────────────────────────

# Severinghaus (1979): SO2 = 100 / (1 + 23400/(PO2^3 + 150*PO2))
# Simple, widely used. Has built-in asymmetry via the PO2^3 + 150*PO2 terms.
# Effective Hill coefficient ~2.7 at steep part, lower at extremes.


@dataclass
class SeveringhausParams:
    pao2_0: float
    pvo2: float
    tau_washout: float
    bohr_max: float
    tau_bohr: float
    lag: float
    r_offset: float


SEVER_BOUNDS_CURRENT = [
    (100, 200),    # pao2_0
    (25, 50),      # pvo2
    (50, 250),     # tau_washout
    (2.0, 10.0),   # bohr_max
    (60, 180),     # tau_bohr
    (5, 45),       # lag
    (-3.0, 3.0),   # r_offset
]

SEVER_BOUNDS_WIDE = [
    (80, 400),     # pao2_0
    (10, 50),      # pvo2
    (30, 400),     # tau_washout
    (1.0, 30.0),   # bohr_max
    (30, 400),     # tau_bohr
    (0, 80),       # lag
    (-5.0, 5.0),   # r_offset
]

SEVER_PARAM_NAMES = [f.name for f in fields(SeveringhausParams)]


def severinghaus_spo2(pao2):
    """Severinghaus (1979) ODC: PO2 → SpO2 (%).

    SO2 = 100 / (1 + 23400/(PO2^3 + 150*PO2))
    """
    x = np.maximum(pao2, 0.01)
    return 100.0 / (1.0 + 23400.0 / (x**3 + 150.0 * x))


def predict_severinghaus(t, params):
    """Predict SpO2 using Severinghaus ODC with virtual PO2 Bohr scaling."""
    t_eff = np.maximum(t - params.lag, 0.0)
    pao2 = params.pvo2 + (params.pao2_0 - params.pvo2) * np.exp(
        -t_eff / max(params.tau_washout, 0.01)
    )
    p50_eff = P50_BASE + params.bohr_max * (
        1.0 - np.exp(-t_eff / max(params.tau_bohr, 0.01))
    )
    pao2_virtual = pao2 * (P50_BASE / p50_eff)
    spo2 = params.r_offset + severinghaus_spo2(pao2_virtual)
    return np.clip(spo2, 0.0, 100.0)


# ── Severinghaus + gamma ─────────────────────────────────────────────────────


@dataclass
class SeveringhausGammaParams:
    pao2_0: float
    pvo2: float
    tau_washout: float
    gamma: float
    bohr_max: float
    tau_bohr: float
    lag: float
    r_offset: float


SEVER_GAMMA_BOUNDS_CURRENT = [
    (100, 200),    # pao2_0
    (25, 50),      # pvo2
    (50, 250),     # tau_washout
    (0.8, 1.5),    # gamma
    (2.0, 10.0),   # bohr_max
    (60, 180),     # tau_bohr
    (5, 45),       # lag
    (-3.0, 3.0),   # r_offset
]

SEVER_GAMMA_BOUNDS_WIDE = [
    (80, 400),     # pao2_0
    (10, 50),      # pvo2
    (30, 400),     # tau_washout
    (0.5, 2.5),    # gamma
    (1.0, 30.0),   # bohr_max
    (30, 400),     # tau_bohr
    (0, 80),       # lag
    (-5.0, 5.0),   # r_offset
]

# Targeted relaxation: widen only the bounds that were hit in "current"
SEVER_GAMMA_BOUNDS_TUNED = [
    (100, 250),    # pao2_0 — FL hyperventilation can reach ~200
    (20, 50),      # pvo2 — slightly wider, but keep physiological
    (50, 250),     # tau_washout — keep current
    (0.8, 2.0),    # gamma — allow more steepness (1.0 = standard, 2.0 generous)
    (2.0, 15.0),   # bohr_max — respiratory acidosis can cause >10 mmHg shift
    (40, 250),     # tau_bohr — slightly wider
    (5, 60),       # lag — allow up to 60s for cold extremities
    (-3.0, 3.0),   # r_offset — keep current
]

SEVER_GAMMA_PARAM_NAMES = [f.name for f in fields(SeveringhausGammaParams)]


def predict_severinghaus_gamma(t, params):
    """Predict SpO2 using Severinghaus + steepness exponent."""
    t_eff = np.maximum(t - params.lag, 0.0)
    pao2 = params.pvo2 + (params.pao2_0 - params.pvo2) * np.exp(
        -t_eff / max(params.tau_washout, 0.01)
    )
    p50_eff = P50_BASE + params.bohr_max * (
        1.0 - np.exp(-t_eff / max(params.tau_bohr, 0.01))
    )
    pao2_virtual = pao2 * (P50_BASE / p50_eff)
    pao2_adj = P50_BASE * (np.maximum(pao2_virtual, 0.01) / P50_BASE) ** params.gamma
    spo2 = params.r_offset + severinghaus_spo2(pao2_adj)
    return np.clip(spo2, 0.0, 100.0)


# ── Kelman + gamma (steepness) model ─────────────────────────────────────────

# Adds a steepness exponent `gamma` to Kelman: 8 params (same count as Hill).
# gamma > 1 → steeper transition; gamma = 1 → standard Kelman.
# Power transform: pao2_adj = P50_BASE * (pao2_virtual / P50_BASE) ^ gamma
# This preserves Kelman's asymmetry while allowing individual cooperativity.


@dataclass
class KelmanGammaParams:
    pao2_0: float
    pvo2: float
    tau_washout: float
    gamma: float  # steepness exponent (analogous to Hill n)
    bohr_max: float
    tau_bohr: float
    lag: float
    r_offset: float


KELMAN_GAMMA_BOUNDS_CURRENT = [
    (100, 200),    # pao2_0
    (25, 50),      # pvo2
    (50, 250),     # tau_washout
    (0.8, 1.5),    # gamma — steepness (1.0 = standard Kelman)
    (2.0, 10.0),   # bohr_max
    (60, 180),     # tau_bohr
    (5, 45),       # lag
    (-3.0, 3.0),   # r_offset
]

KELMAN_GAMMA_BOUNDS_WIDE = [
    (80, 400),     # pao2_0
    (10, 50),      # pvo2
    (30, 400),     # tau_washout
    (0.5, 2.5),    # gamma
    (1.0, 30.0),   # bohr_max
    (30, 400),     # tau_bohr
    (0, 80),       # lag
    (-5.0, 5.0),   # r_offset
]

KELMAN_GAMMA_PARAM_NAMES = [f.name for f in fields(KelmanGammaParams)]


def predict_kelman_gamma(t, params):
    """Predict SpO2 using Kelman ODC with steepness exponent and Bohr effect."""
    t_eff = np.maximum(t - params.lag, 0.0)
    pao2 = params.pvo2 + (params.pao2_0 - params.pvo2) * np.exp(
        -t_eff / max(params.tau_washout, 0.01)
    )
    p50_eff = P50_BASE + params.bohr_max * (
        1.0 - np.exp(-t_eff / max(params.tau_bohr, 0.01))
    )
    # Virtual PO2 for Bohr effect
    pao2_virtual = pao2 * (P50_BASE / p50_eff)
    # Power transform for steepness: preserves P50 crossing, adjusts slope
    pao2_adj = P50_BASE * (np.maximum(pao2_virtual, 0.01) / P50_BASE) ** params.gamma
    spo2 = params.r_offset + kelman_spo2(pao2_adj)
    return np.clip(spo2, 0.0, 100.0)


# ── Fitting ──────────────────────────────────────────────────────────────────


def compute_r_squared(observed, predicted):
    ss_res = np.sum((observed - predicted) ** 2)
    ss_tot = np.sum((observed - np.mean(observed)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def check_bounds(param_values, bounds, param_names, tol=1e-3):
    """Return list of param names that are at their bounds."""
    at_bounds = []
    for val, (lo, hi), name in zip(param_values, bounds, param_names, strict=True):
        if abs(val - lo) < tol or abs(val - hi) < tol:
            at_bounds.append(f"{name}={'lo' if abs(val - lo) < tol else 'hi'}")
    return at_bounds


def fit_model(t, spo2, model_name, predict_fn, param_cls, bounds, param_names):
    """Fit a model using differential evolution."""
    print(f"\n{'='*60}")
    print(f"Fitting {model_name} ({len(bounds)} params)")
    print(f"{'='*60}")

    def objective(arr):
        params = param_cls(*arr)
        pred = predict_fn(t, params)
        return np.sum((spo2 - pred) ** 2)

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
    pred = predict_fn(t, params)
    r2 = compute_r_squared(spo2, pred)
    at_bounds = check_bounds(result.x, bounds, param_names)

    print(f"  Converged: {result.success}")
    print(f"  R²:        {r2:.6f}")
    print(f"  SSE:       {result.fun:.2f}")
    print(f"  Params at bounds ({len(at_bounds)}/{len(bounds)}): {at_bounds or 'none'}")
    print(f"  Parameters:")
    for name, val, (lo, hi) in zip(param_names, result.x, bounds, strict=True):
        marker = " <<<" if any(name in ab for ab in at_bounds) else ""
        print(f"    {name:>14s} = {val:10.4f}  [{lo:>8.1f}, {hi:>8.1f}]{marker}")

    return params, pred, r2, at_bounds


# ── Main ─────────────────────────────────────────────────────────────────────


def compare_results(t, spo2, results):
    """Print side-by-side comparison of fit results."""
    names = list(results.keys())
    print(f"\n{'='*70}")
    print("COMPARISON")
    print(f"{'='*70}")

    header = f"  {'Metric':<25s}" + "".join(f" {n:>14s}" for n in names)
    print(header)
    print(f"  {'-'*25}" + " ".join(f"{'-'*14}" for _ in names))

    # R²
    row = f"  {'R²':<25s}"
    for n in names:
        row += f" {results[n]['r2']:>14.6f}"
    print(row)

    # Num params
    row = f"  {'Num params':<25s}"
    for n in names:
        row += f" {results[n]['n_params']:>14d}"
    print(row)

    # Params at bounds
    row = f"  {'Params at bounds':<25s}"
    for n in names:
        row += f" {len(results[n]['at_bounds']):>14d}"
    print(row)

    # At-bounds detail
    for n in names:
        ab = results[n]["at_bounds"]
        if ab:
            print(f"    {n}: {ab}")

    # Residuals at key time points
    key_times = [200, 247, 300, 372]
    print(f"\n  Residuals at key times:")
    header = f"  {'t (s)':<8s} {'Obs':>6s}"
    for n in names:
        header += f" {n[:12]:>14s}"
    print(header)
    print(f"  {'-'*8} {'-'*6}" + " ".join(f"{'-'*14}" for _ in names))
    for tk in key_times:
        idx = np.argmin(np.abs(t - tk))
        obs = spo2[idx]
        row = f"  {t[idx]:<8.0f} {obs:>6.0f}"
        for n in names:
            res = obs - results[n]["pred"][idx]
            row += f" {res:>+14.2f}"
        print(row)

    # RMSE
    row = f"\n  {'RMSE':<25s}"
    for n in names:
        rmse = np.sqrt(np.mean((spo2 - results[n]["pred"]) ** 2))
        row += f" {rmse:>14.4f}"
    print(row)


def kelman_spo2_gamma(pao2, gamma):
    """Kelman with steepness adjustment for ODC comparison."""
    pao2_adj = P50_BASE * (np.maximum(pao2, 0.01) / P50_BASE) ** gamma
    return kelman_spo2(pao2_adj)


def print_odc_comparison():
    """Print ODC curves for all models at key PO2 values."""
    print(f"\n{'='*90}")
    print("ODC CURVE COMPARISON (no Bohr shift, P50=26.6)")
    print(f"{'='*90}")
    po2_values = [20, 26.6, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150]
    print(f"  {'PO2':>6s}  {'Hill2.7':>7s} {'Hill3.2':>7s} {'Hill4.0':>7s}"
          f"  {'Kelman':>7s} {'KelG1.2':>7s} {'KelG1.5':>7s}"
          f"  {'Sever':>7s} {'SevG1.2':>7s} {'SevG1.5':>7s}")
    print(f"  {'-'*6}  " + "  ".join([f"{'-'*7}"] * 3) + "  "
          + "  ".join([f"{'-'*7}"] * 3) + "  " + "  ".join([f"{'-'*7}"] * 3))
    for po2 in po2_values:
        a = np.array([po2])
        h27 = hill_spo2(a, 26.6, 2.7)[0]
        h32 = hill_spo2(a, 26.6, 3.2)[0]
        h40 = hill_spo2(a, 26.6, 4.0)[0]
        k = kelman_spo2(a)[0]
        kg12 = kelman_spo2_gamma(a, 1.2)[0]
        kg15 = kelman_spo2_gamma(a, 1.5)[0]
        s = severinghaus_spo2(a)[0]
        sg12 = severinghaus_spo2(P50_BASE * (a / P50_BASE) ** 1.2)[0]
        sg15 = severinghaus_spo2(P50_BASE * (a / P50_BASE) ** 1.5)[0]
        print(f"  {po2:>6.1f}  {h27:>7.2f} {h32:>7.2f} {h40:>7.2f}"
              f"  {k:>7.2f} {kg12:>7.2f} {kg15:>7.2f}"
              f"  {s:>7.2f} {sg12:>7.2f} {sg15:>7.2f}")


def main():
    print("Loading hold 6 (FL) data...")
    t, spo2 = load_hold_data(6)
    print(f"  {len(t)} data points, t=[{t[0]:.0f}, {t[-1]:.0f}]s, "
          f"SpO2=[{spo2.min():.0f}, {spo2.max():.0f}]%")

    print_odc_comparison()

    results = {}

    # Hill — current bounds
    p, pred, r2, ab = fit_model(
        t, spo2, "Hill (current)", predict_hill, HillParams,
        HILL_BOUNDS_CURRENT, HILL_PARAM_NAMES,
    )
    results["Hill-cur"] = {"pred": pred, "r2": r2, "at_bounds": ab, "n_params": 8}

    # Hill — wide bounds
    p, pred, r2, ab = fit_model(
        t, spo2, "Hill (wide)", predict_hill, HillParams,
        HILL_BOUNDS_WIDE, HILL_PARAM_NAMES,
    )
    results["Hill-wide"] = {"pred": pred, "r2": r2, "at_bounds": ab, "n_params": 8}

    # Kelman — current bounds
    p, pred, r2, ab = fit_model(
        t, spo2, "Kelman (current)", predict_kelman, KelmanParams,
        KELMAN_BOUNDS_CURRENT, KELMAN_PARAM_NAMES,
    )
    results["Kel-cur"] = {"pred": pred, "r2": r2, "at_bounds": ab, "n_params": 7}

    # Kelman — wide bounds
    p, pred, r2, ab = fit_model(
        t, spo2, "Kelman (wide)", predict_kelman, KelmanParams,
        KELMAN_BOUNDS_WIDE, KELMAN_PARAM_NAMES,
    )
    results["Kel-wide"] = {"pred": pred, "r2": r2, "at_bounds": ab, "n_params": 7}

    # Kelman+gamma — current bounds
    p, pred, r2, ab = fit_model(
        t, spo2, "Kelman+gamma (current)", predict_kelman_gamma, KelmanGammaParams,
        KELMAN_GAMMA_BOUNDS_CURRENT, KELMAN_GAMMA_PARAM_NAMES,
    )
    results["KelG-cur"] = {"pred": pred, "r2": r2, "at_bounds": ab, "n_params": 8}

    # Kelman+gamma — wide bounds
    p, pred, r2, ab = fit_model(
        t, spo2, "Kelman+gamma (wide)", predict_kelman_gamma, KelmanGammaParams,
        KELMAN_GAMMA_BOUNDS_WIDE, KELMAN_GAMMA_PARAM_NAMES,
    )
    results["KelG-wide"] = {"pred": pred, "r2": r2, "at_bounds": ab, "n_params": 8}

    # Severinghaus — current bounds (7 params)
    p, pred, r2, ab = fit_model(
        t, spo2, "Severinghaus (current)", predict_severinghaus, SeveringhausParams,
        SEVER_BOUNDS_CURRENT, SEVER_PARAM_NAMES,
    )
    results["Sev-cur"] = {"pred": pred, "r2": r2, "at_bounds": ab, "n_params": 7}

    # Severinghaus — wide bounds (7 params)
    p, pred, r2, ab = fit_model(
        t, spo2, "Severinghaus (wide)", predict_severinghaus, SeveringhausParams,
        SEVER_BOUNDS_WIDE, SEVER_PARAM_NAMES,
    )
    results["Sev-wide"] = {"pred": pred, "r2": r2, "at_bounds": ab, "n_params": 7}

    # Severinghaus+gamma — current bounds (8 params)
    p, pred, r2, ab = fit_model(
        t, spo2, "Severinghaus+gamma (current)", predict_severinghaus_gamma,
        SeveringhausGammaParams, SEVER_GAMMA_BOUNDS_CURRENT, SEVER_GAMMA_PARAM_NAMES,
    )
    results["SevG-cur"] = {"pred": pred, "r2": r2, "at_bounds": ab, "n_params": 8}

    # Severinghaus+gamma — wide bounds (8 params)
    p, pred, r2, ab = fit_model(
        t, spo2, "Severinghaus+gamma (wide)", predict_severinghaus_gamma,
        SeveringhausGammaParams, SEVER_GAMMA_BOUNDS_WIDE, SEVER_GAMMA_PARAM_NAMES,
    )
    results["SevG-wide"] = {"pred": pred, "r2": r2, "at_bounds": ab, "n_params": 8}

    # Severinghaus+gamma — tuned bounds (8 params)
    p, pred, r2, ab = fit_model(
        t, spo2, "Severinghaus+gamma (tuned)", predict_severinghaus_gamma,
        SeveringhausGammaParams, SEVER_GAMMA_BOUNDS_TUNED, SEVER_GAMMA_PARAM_NAMES,
    )
    results["SevG-tune"] = {"pred": pred, "r2": r2, "at_bounds": ab, "n_params": 8}

    compare_results(t, spo2, results)


if __name__ == "__main__":
    main()
