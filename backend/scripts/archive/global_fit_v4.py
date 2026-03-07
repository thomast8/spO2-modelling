"""
v4 global CO2-Bohr+Delay fit: shared physiology + per-type initial conditions.

Adds sensor delay to the global fit from v3. The delay should allow physiology
params (especially gamma, k_co2) to settle at physiologically meaningful values
instead of compensating for missing circulation lag.

Parameter split (CO2-Bohr+Delay):
  Shared (5): pvo2, gamma, k_co2, r_offset, d
  Type-specific (3 per type): pao2_0, tau_washout, paco2_0

For 3 types: 5 + 3x3 = 14 params total.

Usage:
    cd backend && uv run python -u scripts/global_fit_v4.py
"""

import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution

DB_PATH = Path(__file__).resolve().parents[3] / "data" / "spo2.db"
P50_BASE = 26.6

# ── Data loading ────────────────────────────────────────────────────────────


def load_all_holds() -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    holds = conn.execute(
        "SELECT id, hold_type FROM holds WHERE hold_type != 'untagged' ORDER BY id"
    ).fetchall()
    result = []
    for hold_id, hold_type in holds:
        rows = conn.execute(
            "SELECT elapsed_s, spo2, hr FROM hold_data WHERE hold_id = ? ORDER BY elapsed_s",
            (hold_id,),
        ).fetchall()
        if not rows:
            continue
        result.append({
            "id": hold_id,
            "type": hold_type,
            "t": np.array([r[0] for r in rows], dtype=float),
            "spo2": np.array([r[1] for r in rows], dtype=float),
        })
    conn.close()
    return result


# ── Model functions ─────────────────────────────────────────────────────────


def predict_co2bohr(t, params):
    """CO2-Bohr: 7 params [pao2_0, pvo2, tau_washout, gamma, paco2_0, k_co2, r_offset]."""
    pao2_0, pvo2, tau_washout, gamma, paco2_0, k_co2, r_offset = params
    pao2 = pvo2 + (pao2_0 - pvo2) * np.exp(-t / max(tau_washout, 0.01))
    paco2 = paco2_0 + k_co2 * t
    p50 = P50_BASE + 0.48 * (paco2 - 40.0)
    pao2_v = pao2 * (P50_BASE / np.maximum(p50, 0.01))
    pao2_adj = P50_BASE * (np.maximum(pao2_v, 0.01) / P50_BASE) ** gamma
    x = np.maximum(pao2_adj, 0.01)
    sa = 100.0 / (1.0 + 23400.0 / (x**3 + 150.0 * x))
    return np.clip(sa + r_offset, 0.0, 100.0)


def predict_co2bohr_delay(t, params):
    """CO2-Bohr+Delay: 8 params, adds pure time delay."""
    pao2_0, pvo2, tau_washout, gamma, paco2_0, k_co2, r_offset, d = params
    pao2 = pvo2 + (pao2_0 - pvo2) * np.exp(-t / max(tau_washout, 0.01))
    paco2 = paco2_0 + k_co2 * t
    p50 = P50_BASE + 0.48 * (paco2 - 40.0)
    pao2_v = pao2 * (P50_BASE / np.maximum(p50, 0.01))
    pao2_adj = P50_BASE * (np.maximum(pao2_v, 0.01) / P50_BASE) ** gamma
    x = np.maximum(pao2_adj, 0.01)
    sa = 100.0 / (1.0 + 23400.0 / (x**3 + 150.0 * x))
    sa_delayed = np.interp(t - d, t, sa, left=sa[0])
    return np.clip(sa_delayed + r_offset, 0.0, 100.0)


def predict_richards(t, params):
    """Richards sigmoid: 5 params [s_max, s_min, t50, k, nu]."""
    s_max, s_min, t50, k, nu = params
    z = np.clip((t - t50) / max(k, 0.01), -500, 500)
    base = 1.0 + nu * np.exp(z)
    return np.clip(
        s_min + (s_max - s_min) / np.power(np.maximum(base, 1e-10), 1.0 / nu), 0.0, 100.0
    )


# ── Bounds ──────────────────────────────────────────────────────────────────

# CO2-Bohr (no delay): [pao2_0, pvo2, tau_washout, gamma, paco2_0, k_co2, r_offset]
CO2BOHR_BOUNDS = {
    "FL": [(100, 250), (15, 50), (50, 250), (0.8, 2.5), (20, 50), (0.02, 0.25), (-3, 3)],
    "FRC": [(80, 140), (15, 50), (20, 100), (0.8, 2.5), (25, 50), (0.02, 0.25), (-3, 3)],
    "RV": [(70, 110), (15, 50), (10, 80), (0.8, 2.5), (30, 55), (0.02, 0.25), (-3, 3)],
}
CO2BOHR_SHARED_IDX = [1, 3, 5, 6]  # pvo2, gamma, k_co2, r_offset
CO2BOHR_TYPE_SPECIFIC_IDX = [0, 2, 4]  # pao2_0, tau_washout, paco2_0
CO2BOHR_SHARED_NAMES = ["pvo2", "gamma", "k_co2", "r_offset"]
CO2BOHR_TYPE_SPECIFIC_NAMES = ["pao2_0", "tau_washout", "paco2_0"]

# CO2-Bohr+Delay: [pao2_0, pvo2, tau_washout, gamma, paco2_0, k_co2, r_offset, d]
# Gamma narrowed to 0.8-1.5; delay d in 3-30s
CO2BOHR_DELAY_BOUNDS = {
    "FL": [
        (100, 250), (15, 50), (50, 250), (0.8, 1.5), (20, 50), (0.02, 0.25), (-3, 3), (3, 30),
    ],
    "FRC": [
        (80, 140), (15, 50), (20, 100), (0.8, 1.5), (25, 50), (0.02, 0.25), (-3, 3), (3, 30),
    ],
    "RV": [
        (70, 110), (15, 50), (10, 80), (0.8, 1.5), (30, 55), (0.02, 0.25), (-3, 3), (3, 30),
    ],
}
DELAY_SHARED_IDX = [1, 3, 5, 6, 7]  # pvo2, gamma, k_co2, r_offset, d
DELAY_TYPE_SPECIFIC_IDX = [0, 2, 4]  # pao2_0, tau_washout, paco2_0
DELAY_SHARED_NAMES = ["pvo2", "gamma", "k_co2", "r_offset", "d"]
DELAY_TYPE_SPECIFIC_NAMES = ["pao2_0", "tau_washout", "paco2_0"]

# All param names for the 8-param model
ALL_DELAY_PARAM_NAMES = [
    "pao2_0", "pvo2", "tau_washout", "gamma", "paco2_0", "k_co2", "r_offset", "d",
]

RICHARDS_BOUNDS = {
    "FL": [(96, 101), (0, 96), (50, 500), (5, 80), (0.1, 10)],
    "FRC": [(96, 101), (0, 96), (20, 300), (3, 60), (0.1, 10)],
    "RV": [(96, 101), (0, 96), (10, 250), (3, 60), (0.1, 10)],
}


# ── Helpers ─────────────────────────────────────────────────────────────────


def compute_r2(obs, pred):
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def compute_rmse(obs, pred):
    return float(np.sqrt(np.mean((obs - pred) ** 2)))


def is_at_bound(val, lo, hi, tol=1e-3):
    return abs(val - lo) < tol or abs(val - hi) < tol


# ── Per-hold fitting ────────────────────────────────────────────────────────


def fit_perhold_co2bohr(hold):
    """Fit CO2-Bohr (no delay) to a single hold."""
    bounds = CO2BOHR_BOUNDS[hold["type"]]
    t, spo2 = hold["t"], hold["spo2"]

    def objective(arr):
        pred = predict_co2bohr(t, arr)
        w = np.where(spo2 < 95, 3.0, 1.0)
        return np.sum(w * (spo2 - pred) ** 2)

    result = differential_evolution(
        objective, bounds, maxiter=3000, seed=42, tol=1e-10,
        polish=True, popsize=40, mutation=(0.5, 1.5), recombination=0.9,
    )
    return result.x


def fit_perhold_co2bohr_delay(hold):
    """Fit CO2-Bohr+Delay to a single hold."""
    bounds = CO2BOHR_DELAY_BOUNDS[hold["type"]]
    t, spo2 = hold["t"], hold["spo2"]

    def objective(arr):
        pred = predict_co2bohr_delay(t, arr)
        w = np.where(spo2 < 95, 3.0, 1.0)
        return np.sum(w * (spo2 - pred) ** 2)

    result = differential_evolution(
        objective, bounds, maxiter=3000, seed=42, tol=1e-10,
        polish=True, popsize=40, mutation=(0.5, 1.5), recombination=0.9,
    )
    return result.x


def fit_perhold_richards(hold):
    """Fit Richards to a single hold."""
    bounds = RICHARDS_BOUNDS[hold["type"]]
    t, spo2 = hold["t"], hold["spo2"]

    def objective(arr):
        return np.sum((spo2 - predict_richards(t, arr)) ** 2)

    result = differential_evolution(
        objective, bounds, maxiter=3000, seed=42, tol=1e-10,
        polish=True, popsize=40, mutation=(0.5, 1.5), recombination=0.9,
    )
    return result.x


# ── Global fit (CO2-Bohr, no delay — baseline) ─────────────────────────────


def build_global_bounds_co2bohr(hold_types: list[str]) -> list[tuple]:
    """Build flat bounds: [shared..., type1_specific..., type2_specific..., ...]"""
    shared_bounds = []
    for idx in CO2BOHR_SHARED_IDX:
        lo = min(CO2BOHR_BOUNDS[ht][idx][0] for ht in hold_types)
        hi = max(CO2BOHR_BOUNDS[ht][idx][1] for ht in hold_types)
        shared_bounds.append((lo, hi))

    type_bounds = []
    for ht in hold_types:
        for idx in CO2BOHR_TYPE_SPECIFIC_IDX:
            type_bounds.append(CO2BOHR_BOUNDS[ht][idx])

    return shared_bounds + type_bounds


def unpack_global_co2bohr(flat, hold_types: list[str], hold_type: str) -> np.ndarray:
    """Unpack flat -> 7-param CO2-Bohr vector for a given type."""
    n_shared = len(CO2BOHR_SHARED_IDX)
    shared = flat[:n_shared]
    type_idx = hold_types.index(hold_type)
    offset = n_shared + type_idx * len(CO2BOHR_TYPE_SPECIFIC_IDX)
    specific = flat[offset : offset + len(CO2BOHR_TYPE_SPECIFIC_IDX)]
    return np.array([
        specific[0], shared[0], specific[1], shared[1],
        specific[2], shared[2], shared[3],
    ])


def global_fit_co2bohr(holds: list[dict]) -> tuple[np.ndarray, list[str]]:
    """Run global CO2-Bohr fit."""
    hold_types = sorted(set(h["type"] for h in holds))
    bounds = build_global_bounds_co2bohr(hold_types)
    holds_by_type = {}
    for h in holds:
        holds_by_type.setdefault(h["type"], []).append(h)

    def objective(flat):
        total_sse = 0.0
        for ht in hold_types:
            params_7 = unpack_global_co2bohr(flat, hold_types, ht)
            for h in holds_by_type[ht]:
                pred = predict_co2bohr(h["t"], params_7)
                w = np.where(h["spo2"] < 95, 3.0, 1.0)
                total_sse += np.sum(w * (h["spo2"] - pred) ** 2)
        return total_sse

    n_params = len(bounds)
    print(f"\nGlobal CO2-Bohr fit: {n_params} params "
          f"({len(CO2BOHR_SHARED_IDX)} shared + "
          f"{len(CO2BOHR_TYPE_SPECIFIC_IDX)}x{len(hold_types)} type-specific)")

    result = differential_evolution(
        objective, bounds, maxiter=5000, seed=42, tol=1e-10,
        polish=True, popsize=50, mutation=(0.5, 1.5), recombination=0.9,
    )
    print(f"Converged: {result.success}, fun={result.fun:.2f}, nfev={result.nfev}")
    return result.x, hold_types


# ── Global fit (CO2-Bohr+Delay) ────────────────────────────────────────────


def build_global_bounds_delay(hold_types: list[str]) -> list[tuple]:
    """Build flat bounds: [shared..., type1_specific..., type2_specific..., ...]"""
    shared_bounds = []
    for idx in DELAY_SHARED_IDX:
        lo = min(CO2BOHR_DELAY_BOUNDS[ht][idx][0] for ht in hold_types)
        hi = max(CO2BOHR_DELAY_BOUNDS[ht][idx][1] for ht in hold_types)
        shared_bounds.append((lo, hi))

    type_bounds = []
    for ht in hold_types:
        for idx in DELAY_TYPE_SPECIFIC_IDX:
            type_bounds.append(CO2BOHR_DELAY_BOUNDS[ht][idx])

    return shared_bounds + type_bounds


def unpack_global_delay(flat, hold_types: list[str], hold_type: str) -> np.ndarray:
    """Unpack flat -> 8-param CO2-Bohr+Delay vector for a given type."""
    n_shared = len(DELAY_SHARED_IDX)
    shared = flat[:n_shared]  # pvo2, gamma, k_co2, r_offset, d
    type_idx = hold_types.index(hold_type)
    offset = n_shared + type_idx * len(DELAY_TYPE_SPECIFIC_IDX)
    specific = flat[offset : offset + len(DELAY_TYPE_SPECIFIC_IDX)]  # pao2_0, tau_washout, paco2_0
    # Reassemble: [pao2_0, pvo2, tau_washout, gamma, paco2_0, k_co2, r_offset, d]
    return np.array([
        specific[0], shared[0], specific[1], shared[1],
        specific[2], shared[2], shared[3], shared[4],
    ])


def global_fit_delay(holds: list[dict]) -> tuple[np.ndarray, list[str]]:
    """Run global CO2-Bohr+Delay fit."""
    hold_types = sorted(set(h["type"] for h in holds))
    bounds = build_global_bounds_delay(hold_types)
    holds_by_type = {}
    for h in holds:
        holds_by_type.setdefault(h["type"], []).append(h)

    def objective(flat):
        total_sse = 0.0
        for ht in hold_types:
            params_8 = unpack_global_delay(flat, hold_types, ht)
            for h in holds_by_type[ht]:
                pred = predict_co2bohr_delay(h["t"], params_8)
                w = np.where(h["spo2"] < 95, 3.0, 1.0)
                total_sse += np.sum(w * (h["spo2"] - pred) ** 2)
        return total_sse

    n_params = len(bounds)
    print(f"\nGlobal CO2-Bohr+Delay fit: {n_params} params "
          f"({len(DELAY_SHARED_IDX)} shared + "
          f"{len(DELAY_TYPE_SPECIFIC_IDX)}x{len(hold_types)} type-specific)")

    result = differential_evolution(
        objective, bounds, maxiter=6000, seed=42, tol=1e-10,
        polish=True, popsize=60, mutation=(0.5, 1.5), recombination=0.9,
    )
    print(f"Converged: {result.success}, fun={result.fun:.2f}, nfev={result.nfev}")
    return result.x, hold_types


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    all_holds = load_all_holds()
    hold_labels = {h["id"]: f"{h['type']}#{h['id']}" for h in all_holds}

    # Exclude FL#1 — only 2% SpO2 variation, no useful signal
    EXCLUDED_IDS = {1}
    holds = [h for h in all_holds if h["id"] not in EXCLUDED_IDS]
    excluded = [h for h in all_holds if h["id"] in EXCLUDED_IDS]

    print("=" * 90)
    print("GLOBAL v4 FIT: CO2-Bohr vs CO2-Bohr+Delay")
    print("=" * 90)
    print(f"\nLoaded {len(all_holds)} holds, excluded {len(excluded)} "
          f"({', '.join(hold_labels[h['id']] for h in excluded)}):")
    for h in all_holds:
        tag = " [EXCLUDED]" if h["id"] in EXCLUDED_IDS else ""
        print(f"  {hold_labels[h['id']]}: {len(h['t'])} pts, "
              f"SpO2 {h['spo2'].min():.0f}-{h['spo2'].max():.0f}%{tag}")

    # ── Step 1: Per-hold baselines ──────────────────────────────────────────

    print(f"\n{'='*90}")
    print("STEP 1: Per-hold baseline fits")
    print(f"{'='*90}")

    perhold_co2bohr = {}
    perhold_delay = {}
    perhold_richards = {}

    for h in all_holds:
        label = hold_labels[h["id"]]

        print(f"\n  Fitting CO2-Bohr on {label}...", end="", flush=True)
        perhold_co2bohr[h["id"]] = fit_perhold_co2bohr(h)
        pred = predict_co2bohr(h["t"], perhold_co2bohr[h["id"]])
        print(f" R²={compute_r2(h['spo2'], pred):.4f}", flush=True)

        print(f"  Fitting CO2-Bohr+Delay on {label}...", end="", flush=True)
        perhold_delay[h["id"]] = fit_perhold_co2bohr_delay(h)
        pred = predict_co2bohr_delay(h["t"], perhold_delay[h["id"]])
        print(f" R²={compute_r2(h['spo2'], pred):.4f}", flush=True)

        print(f"  Fitting Richards on {label}...", end="", flush=True)
        perhold_richards[h["id"]] = fit_perhold_richards(h)
        pred = predict_richards(h["t"], perhold_richards[h["id"]])
        print(f" R²={compute_r2(h['spo2'], pred):.4f}", flush=True)

    # ── Step 2: Global fits ─────────────────────────────────────────────────

    print(f"\n{'='*90}")
    print("STEP 2: Global fits")
    print(f"{'='*90}")

    flat_co2bohr, hold_types = global_fit_co2bohr(holds)
    flat_delay, _ = global_fit_delay(holds)

    # ── Step 3: Display shared params ───────────────────────────────────────

    print(f"\n{'='*90}")
    print("STEP 3: Shared parameter comparison")
    print(f"{'='*90}")

    # CO2-Bohr global shared
    n_shared_co2 = len(CO2BOHR_SHARED_IDX)
    shared_co2 = flat_co2bohr[:n_shared_co2]
    global_bounds_co2 = build_global_bounds_co2bohr(hold_types)

    print(f"\n  CO2-Bohr global shared:")
    for i, (name, val) in enumerate(zip(CO2BOHR_SHARED_NAMES, shared_co2)):
        lo, hi = global_bounds_co2[i]
        flag = " ** AT BOUND **" if is_at_bound(val, lo, hi) else ""
        print(f"    {name:>12s} = {val:8.4f}  [{lo}, {hi}]{flag}")

    # CO2-Bohr+Delay global shared
    n_shared_delay = len(DELAY_SHARED_IDX)
    shared_delay = flat_delay[:n_shared_delay]
    global_bounds_delay = build_global_bounds_delay(hold_types)

    print(f"\n  CO2-Bohr+Delay global shared:")
    for i, (name, val) in enumerate(zip(DELAY_SHARED_NAMES, shared_delay)):
        lo, hi = global_bounds_delay[i]
        flag = " ** AT BOUND **" if is_at_bound(val, lo, hi) else ""
        print(f"    {name:>12s} = {val:8.4f}  [{lo}, {hi}]{flag}")

    # Type-specific for both
    print(f"\n  Type-specific parameters:")
    for ht in hold_types:
        print(f"\n    {ht}:")
        params_7 = unpack_global_co2bohr(flat_co2bohr, hold_types, ht)
        params_8 = unpack_global_delay(flat_delay, hold_types, ht)
        print(f"      {'Param':<12s}  {'CO2-Bohr':>10s}  {'CO2-B+Delay':>12s}")
        for idx, name in zip(CO2BOHR_TYPE_SPECIFIC_IDX, CO2BOHR_TYPE_SPECIFIC_NAMES):
            v_co2 = params_7[idx]
            v_delay = params_8[idx]
            print(f"      {name:<12s}  {v_co2:>10.2f}  {v_delay:>12.2f}")

    # ── Step 4: Per-hold R² comparison ──────────────────────────────────────

    print(f"\n{'='*90}")
    print("STEP 4: Per-hold R² comparison")
    print(f"{'='*90}")

    header = (f"  {'Hold':<14s} {'PH CO2B':>10s} {'PH CO2B+D':>10s} "
              f"{'Glob CO2B':>10s} {'Glob CO2B+D':>12s} {'Richards':>10s}")
    print(header)
    print(f"  {'-'*14} {'-'*10} {'-'*10} {'-'*10} {'-'*12} {'-'*10}")

    r2_ph_co2_list, r2_ph_delay_list = [], []
    r2_gl_co2_list, r2_gl_delay_list = [], []
    r2_richards_list = []

    for h in all_holds:
        label = hold_labels[h["id"]]
        tag = " (excl)" if h["id"] in EXCLUDED_IDS else ""

        r2_ph_co2 = compute_r2(h["spo2"], predict_co2bohr(h["t"], perhold_co2bohr[h["id"]]))
        r2_ph_delay = compute_r2(h["spo2"], predict_co2bohr_delay(h["t"], perhold_delay[h["id"]]))

        params_7 = unpack_global_co2bohr(flat_co2bohr, hold_types, h["type"])
        r2_gl_co2 = compute_r2(h["spo2"], predict_co2bohr(h["t"], params_7))

        params_8 = unpack_global_delay(flat_delay, hold_types, h["type"])
        r2_gl_delay = compute_r2(h["spo2"], predict_co2bohr_delay(h["t"], params_8))

        r2_ri = compute_r2(h["spo2"], predict_richards(h["t"], perhold_richards[h["id"]]))

        print(f"  {label + tag:<14s} {r2_ph_co2:>10.4f} {r2_ph_delay:>10.4f} "
              f"{r2_gl_co2:>10.4f} {r2_gl_delay:>12.4f} {r2_ri:>10.4f}")

        r2_ph_co2_list.append(r2_ph_co2)
        r2_ph_delay_list.append(r2_ph_delay)
        r2_gl_co2_list.append(r2_gl_co2)
        r2_gl_delay_list.append(r2_gl_delay)
        r2_richards_list.append(r2_ri)

    # Averages (fitted holds only)
    fitted_idx = [i for i, h in enumerate(all_holds) if h["id"] not in EXCLUDED_IDS]
    avg_ph_co2 = np.mean([r2_ph_co2_list[i] for i in fitted_idx])
    avg_ph_delay = np.mean([r2_ph_delay_list[i] for i in fitted_idx])
    avg_gl_co2 = np.mean([r2_gl_co2_list[i] for i in fitted_idx])
    avg_gl_delay = np.mean([r2_gl_delay_list[i] for i in fitted_idx])
    avg_ri = np.mean([r2_richards_list[i] for i in fitted_idx])
    print(f"\n  {'Avg (fitted)':<14s} {avg_ph_co2:>10.4f} {avg_ph_delay:>10.4f} "
          f"{avg_gl_co2:>10.4f} {avg_gl_delay:>12.4f} {avg_ri:>10.4f}")

    # ── Step 5: RMSE comparison ─────────────────────────────────────────────

    print(f"\n{'='*90}")
    print("STEP 5: Per-hold RMSE comparison")
    print(f"{'='*90}")

    header = (f"  {'Hold':<14s} {'PH CO2B':>10s} {'PH CO2B+D':>10s} "
              f"{'Glob CO2B':>10s} {'Glob CO2B+D':>12s} {'Richards':>10s}")
    print(header)
    print(f"  {'-'*14} {'-'*10} {'-'*10} {'-'*10} {'-'*12} {'-'*10}")

    for h in all_holds:
        label = hold_labels[h["id"]]
        tag = " (excl)" if h["id"] in EXCLUDED_IDS else ""
        rmse_ph_co2 = compute_rmse(h["spo2"], predict_co2bohr(h["t"], perhold_co2bohr[h["id"]]))
        rmse_ph_d = compute_rmse(h["spo2"], predict_co2bohr_delay(h["t"], perhold_delay[h["id"]]))
        params_7 = unpack_global_co2bohr(flat_co2bohr, hold_types, h["type"])
        rmse_gl_co2 = compute_rmse(h["spo2"], predict_co2bohr(h["t"], params_7))
        params_8 = unpack_global_delay(flat_delay, hold_types, h["type"])
        rmse_gl_d = compute_rmse(h["spo2"], predict_co2bohr_delay(h["t"], params_8))
        rmse_ri = compute_rmse(h["spo2"], predict_richards(h["t"], perhold_richards[h["id"]]))
        print(f"  {label + tag:<14s} {rmse_ph_co2:>10.2f} {rmse_ph_d:>10.2f} "
              f"{rmse_gl_co2:>10.2f} {rmse_gl_d:>12.2f} {rmse_ri:>10.2f}")

    # ── Step 6: Identifiability ─────────────────────────────────────────────

    print(f"\n{'='*90}")
    print("STEP 6: Parameter identifiability")
    print(f"{'='*90}")

    for fit_name, flat, shared_names, shared_idx, ts_names, ts_idx, bounds_dict, build_fn in [
        ("CO2-Bohr", flat_co2bohr, CO2BOHR_SHARED_NAMES, CO2BOHR_SHARED_IDX,
         CO2BOHR_TYPE_SPECIFIC_NAMES, CO2BOHR_TYPE_SPECIFIC_IDX, CO2BOHR_BOUNDS,
         build_global_bounds_co2bohr),
        ("CO2-Bohr+Delay", flat_delay, DELAY_SHARED_NAMES, DELAY_SHARED_IDX,
         DELAY_TYPE_SPECIFIC_NAMES, DELAY_TYPE_SPECIFIC_IDX, CO2BOHR_DELAY_BOUNDS,
         build_global_bounds_delay),
    ]:
        global_bounds = build_fn(hold_types)
        n_shared = len(shared_idx)
        shared_params = flat[:n_shared]
        total_params = len(flat)
        total_at_bound = 0

        print(f"\n  --- {fit_name} ({total_params} params) ---")
        print(f"  Shared params at bounds:")
        for i, (name, val) in enumerate(zip(shared_names, shared_params)):
            lo, hi = global_bounds[i]
            if is_at_bound(val, lo, hi):
                total_at_bound += 1
                print(f"    {name}: {val:.4f} (bound: [{lo}, {hi}])")
        if total_at_bound == 0:
            print(f"    None — all interior")

        print(f"  Type-specific params at bounds:")
        for ht in hold_types:
            if fit_name == "CO2-Bohr":
                params_full = unpack_global_co2bohr(flat, hold_types, ht)
            else:
                params_full = unpack_global_delay(flat, hold_types, ht)
            for idx, name in zip(ts_idx, ts_names):
                val = params_full[idx]
                lo, hi = bounds_dict[ht][idx]
                if is_at_bound(val, lo, hi):
                    total_at_bound += 1
                    print(f"    {ht}/{name}: {val:.4f} (bound: [{lo}, {hi}])")
        print(f"  Total: {total_at_bound}/{total_params} at bounds")

    # ── Step 7: Per-hold delay values ───────────────────────────────────────

    print(f"\n{'='*90}")
    print("STEP 7: Per-hold delay values (d)")
    print(f"{'='*90}")

    print(f"\n  {'Hold':<14s} {'Per-hold d':>10s}")
    print(f"  {'-'*14} {'-'*10}")
    d_values = []
    for h in all_holds:
        label = hold_labels[h["id"]]
        d_val = perhold_delay[h["id"]][7]  # d is index 7
        d_values.append(d_val)
        print(f"  {label:<14s} {d_val:>10.2f}")
    print(f"\n  Range: {min(d_values):.1f} - {max(d_values):.1f}s, "
          f"Median: {np.median(d_values):.1f}s, Global: {shared_delay[4]:.1f}s")

    # ── Step 8: Shared-param consistency ────────────────────────────────────

    print(f"\n{'='*90}")
    print("STEP 8: Shared-param consistency (per-hold vs global)")
    print(f"{'='*90}")

    print(f"\n  Per-hold CO2-Bohr+Delay shared params:")
    header = f"  {'Hold':<14s}" + "".join(f" {n:>10s}" for n in DELAY_SHARED_NAMES)
    print(header)
    print(f"  {'-'*14}" + "".join(f" {'-'*10}" for _ in DELAY_SHARED_NAMES))
    for h in all_holds:
        label = hold_labels[h["id"]]
        tag = " (excl)" if h["id"] in EXCLUDED_IDS else ""
        p = perhold_delay[h["id"]]
        row = f"  {label + tag:<14s}"
        for idx in DELAY_SHARED_IDX:
            row += f" {p[idx]:>10.4f}"
        print(row)
    row = f"  {'GLOBAL':<14s}"
    for val in shared_delay:
        row += f" {val:>10.4f}"
    print(row)

    print(f"\n  Range across per-hold fits (fitted holds only):")
    for i, (name, idx) in enumerate(zip(DELAY_SHARED_NAMES, DELAY_SHARED_IDX)):
        vals = [perhold_delay[h["id"]][idx] for h in holds]
        print(f"    {name}: {min(vals):.4f} - {max(vals):.4f}  "
              f"(range={max(vals)-min(vals):.4f}, global={shared_delay[i]:.4f})")

    # ── Step 9: Summary ────────────────────────────────────────────────────

    print(f"\n{'='*90}")
    print("SUMMARY")
    print(f"{'='*90}")

    n_fitted = len(holds)
    n_types = len(hold_types)

    # Count at-bounds for summary
    at_bound_co2 = 0
    gb_co2 = build_global_bounds_co2bohr(hold_types)
    for i in range(len(CO2BOHR_SHARED_IDX)):
        if is_at_bound(flat_co2bohr[i], gb_co2[i][0], gb_co2[i][1]):
            at_bound_co2 += 1
    for ht in hold_types:
        p = unpack_global_co2bohr(flat_co2bohr, hold_types, ht)
        for idx in CO2BOHR_TYPE_SPECIFIC_IDX:
            if is_at_bound(p[idx], CO2BOHR_BOUNDS[ht][idx][0], CO2BOHR_BOUNDS[ht][idx][1]):
                at_bound_co2 += 1
    total_co2 = len(flat_co2bohr)

    at_bound_delay = 0
    gb_delay = build_global_bounds_delay(hold_types)
    for i in range(len(DELAY_SHARED_IDX)):
        if is_at_bound(flat_delay[i], gb_delay[i][0], gb_delay[i][1]):
            at_bound_delay += 1
    for ht in hold_types:
        p = unpack_global_delay(flat_delay, hold_types, ht)
        for idx in DELAY_TYPE_SPECIFIC_IDX:
            if is_at_bound(p[idx], CO2BOHR_DELAY_BOUNDS[ht][idx][0],
                           CO2BOHR_DELAY_BOUNDS[ht][idx][1]):
                at_bound_delay += 1
    total_delay = len(flat_delay)

    print(f"""
  | Metric                    | Global CO2-Bohr | Global CO2-B+Delay | Richards (PH) |
  |---------------------------|:---------------:|:------------------:|:-------------:|
  | Total free params         | {total_co2:<15d} | {total_delay:<18d} | {5*n_fitted:<13d} |
  | Shared params             | {len(CO2BOHR_SHARED_IDX):<15d} | {len(DELAY_SHARED_IDX):<18d} | {'N/A':<13s} |
  | Avg R² (fitted, global)   | {avg_gl_co2:<15.4f} | {avg_gl_delay:<18.4f} | {avg_ri:<13.4f} |
  | Avg R² (fitted, per-hold) | {avg_ph_co2:<15.4f} | {avg_ph_delay:<18.4f} | {avg_ri:<13.4f} |
  | Params at bounds          | {f"{at_bound_co2}/{total_co2}":<15s} | {f"{at_bound_delay}/{total_delay}":<18s} | {'see PH':<13s} |
  | Sensor delay (d)          | {'N/A':<15s} | {f"{shared_delay[4]:.1f}s":<18s} | {'N/A':<13s} |
  | Gamma                     | {shared_co2[1]:<15.4f} | {shared_delay[1]:<18.4f} | {'N/A':<13s} |
  | k_co2                     | {shared_co2[2]:<15.4f} | {shared_delay[2]:<18.4f} | {'N/A':<13s} |
""")

    # ── Step 10: Plot ──────────────────────────────────────────────────────

    print("Generating plot...", flush=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    sorted_holds = sorted(
        all_holds, key=lambda h: ({"FL": 0, "FRC": 1, "RV": 2}[h["type"]], h["id"])
    )

    for ax_idx, h in enumerate(sorted_holds):
        row, col = divmod(ax_idx, 3)
        ax = axes[row, col]
        label = hold_labels[h["id"]]
        is_excluded = h["id"] in EXCLUDED_IDS

        ax.plot(h["t"], h["spo2"], "k.", ms=2, alpha=0.4, label="Observed", zorder=1)

        # Global CO2-Bohr
        params_7 = unpack_global_co2bohr(flat_co2bohr, hold_types, h["type"])
        pred_gl_co2 = predict_co2bohr(h["t"], params_7)
        r2_gl_co2 = compute_r2(h["spo2"], pred_gl_co2)
        ax.plot(h["t"], pred_gl_co2, color="#1f77b4", lw=1.5, alpha=0.6, ls="--",
                label=f"Global CO2B (R²={r2_gl_co2:.3f})", zorder=2)

        # Global CO2-Bohr+Delay
        params_8 = unpack_global_delay(flat_delay, hold_types, h["type"])
        pred_gl_delay = predict_co2bohr_delay(h["t"], params_8)
        r2_gl_delay = compute_r2(h["spo2"], pred_gl_delay)
        ax.plot(h["t"], pred_gl_delay, color="#ff7f0e", lw=2.0, alpha=0.9,
                label=f"Global CO2B+D (R²={r2_gl_delay:.3f})", zorder=3)

        # Richards benchmark
        pred_ri = predict_richards(h["t"], perhold_richards[h["id"]])
        r2_ri = compute_r2(h["spo2"], pred_ri)
        ax.plot(h["t"], pred_ri, color="#2ca02c", lw=1.5, alpha=0.7, ls="--",
                label=f"Richards PH (R²={r2_ri:.3f})", zorder=2)

        title = f"{label} [EXCLUDED]" if is_excluded else label
        ax.set_title(title, fontsize=12, fontweight="bold",
                     color="red" if is_excluded else "black")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("SpO2 (%)")
        ax.set_ylim(30, 105)
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(True, alpha=0.3)

    n_shared_co2_total = len(CO2BOHR_SHARED_IDX)
    n_shared_delay_total = len(DELAY_SHARED_IDX)
    fig.suptitle(
        f"Global Fit Comparison: CO2-Bohr ({n_shared_co2_total}+{len(CO2BOHR_TYPE_SPECIFIC_IDX)}x"
        f"{n_types}={total_co2}p) vs CO2-Bohr+Delay ({n_shared_delay_total}+"
        f"{len(DELAY_TYPE_SPECIFIC_IDX)}x{n_types}={total_delay}p)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()

    out_path = Path(__file__).resolve().parent / "global_fit_v4.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    main()
