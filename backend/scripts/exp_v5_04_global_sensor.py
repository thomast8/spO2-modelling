"""
v5 Experiment 4: Global Fit with Gamma Constraint.

Since Exp 3 showed d still pins at 30 (confounded), we take the fallback path:
fix d=0, fit tau_f as shared param, gamma in (0.9, 1.3).

Gamma sweep: free (0.8-2.0), narrow (0.9-1.3), fixed (1.0) — to quantify
how much R² is lost by constraining gamma now that the filter exists.

Parameter split:
  Shared: pvo2, gamma, k_co2, r_offset, tau_f (5 params)
  Type-specific: pao2_0, tau_washout, paco2_0 (3 per type × 3 types = 9)
  Total: 14 params

Baseline: v4 global fits (CO2-Bohr 13p: R²=0.9602, CO2-Bohr+Delay 14p: R²=0.9606).

Usage:
    cd backend && uv run python -u scripts/exp_v5_04_global_sensor.py
"""

import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution
from scipy.signal import lfilter

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "spo2.db"
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
    """CO2-Bohr: 7 params."""
    pao2_0, pvo2, tau_washout, gamma, paco2_0, k_co2, r_offset = params
    pao2 = pvo2 + (pao2_0 - pvo2) * np.exp(-t / max(tau_washout, 0.01))
    paco2 = paco2_0 + k_co2 * t
    p50 = P50_BASE + 0.48 * (paco2 - 40.0)
    pao2_v = pao2 * (P50_BASE / np.maximum(p50, 0.01))
    pao2_adj = P50_BASE * (np.maximum(pao2_v, 0.01) / P50_BASE) ** gamma
    x = np.maximum(pao2_adj, 0.01)
    sa = 100.0 / (1.0 + 23400.0 / (x**3 + 150.0 * x))
    return np.clip(sa + r_offset, 0.0, 100.0)


def predict_co2bohr_filter(t, params):
    """CO2-Bohr+Filter: 8 params, IIR filter only (no delay)."""
    pao2_0, pvo2, tau_washout, gamma, paco2_0, k_co2, r_offset, tau_f = params
    pao2 = pvo2 + (pao2_0 - pvo2) * np.exp(-t / max(tau_washout, 0.01))
    paco2 = paco2_0 + k_co2 * t
    p50 = P50_BASE + 0.48 * (paco2 - 40.0)
    pao2_v = pao2 * (P50_BASE / np.maximum(p50, 0.01))
    pao2_adj = P50_BASE * (np.maximum(pao2_v, 0.01) / P50_BASE) ** gamma
    x = np.maximum(pao2_adj, 0.01)
    sa = 100.0 / (1.0 + 23400.0 / (x**3 + 150.0 * x))
    # IIR filter
    dt = 1.0
    alpha = dt / (max(tau_f, 0.01) + dt)
    s_meas = lfilter([alpha], [1.0, -(1.0 - alpha)], sa)
    return np.clip(s_meas + r_offset, 0.0, 100.0)


def predict_co2bohr_delay(t, params):
    """CO2-Bohr+Delay: 8 params (from v4)."""
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
    s_max, s_min, t50, k, nu = params
    z = np.clip((t - t50) / max(k, 0.01), -500, 500)
    base = 1.0 + nu * np.exp(z)
    return np.clip(
        s_min + (s_max - s_min) / np.power(np.maximum(base, 1e-10), 1.0 / nu), 0.0, 100.0
    )


# ── Helpers ─────────────────────────────────────────────────────────────────


def compute_r2(obs, pred):
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def compute_rmse(obs, pred):
    return float(np.sqrt(np.mean((obs - pred) ** 2)))


def is_at_bound(val, lo, hi, tol=1e-3):
    return abs(val - lo) < tol or abs(val - hi) < tol


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

# CO2-Bohr+Filter: same + tau_f; gamma bounds adjusted per sweep
FILTER_BOUNDS_TEMPLATE = {
    "FL": [(100, 250), (15, 50), (50, 250), None, (20, 50), (0.02, 0.25), (-3, 3), (1, 20)],
    "FRC": [(80, 140), (15, 50), (20, 100), None, (25, 50), (0.02, 0.25), (-3, 3), (1, 20)],
    "RV": [(70, 110), (15, 50), (10, 80), None, (30, 55), (0.02, 0.25), (-3, 3), (1, 20)],
}
FILTER_SHARED_IDX = [1, 3, 5, 6, 7]  # pvo2, gamma, k_co2, r_offset, tau_f
FILTER_TYPE_SPECIFIC_IDX = [0, 2, 4]  # pao2_0, tau_washout, paco2_0
FILTER_SHARED_NAMES = ["pvo2", "gamma", "k_co2", "r_offset", "tau_f"]
FILTER_TYPE_SPECIFIC_NAMES = ["pao2_0", "tau_washout", "paco2_0"]

# CO2-Bohr+Delay (v4 baseline): d shared
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
DELAY_TYPE_SPECIFIC_IDX = [0, 2, 4]
DELAY_SHARED_NAMES = ["pvo2", "gamma", "k_co2", "r_offset", "d"]
DELAY_TYPE_SPECIFIC_NAMES = ["pao2_0", "tau_washout", "paco2_0"]

RICHARDS_BOUNDS = {
    "FL": [(96, 101), (0, 96), (50, 500), (5, 80), (0.1, 10)],
    "FRC": [(96, 101), (0, 96), (20, 300), (3, 60), (0.1, 10)],
    "RV": [(96, 101), (0, 96), (10, 250), (3, 60), (0.1, 10)],
}


def make_filter_bounds(gamma_bounds: tuple[float, float]) -> dict:
    """Create filter bounds with specified gamma range."""
    result = {}
    for ht, template in FILTER_BOUNDS_TEMPLATE.items():
        bounds = list(template)
        bounds[3] = gamma_bounds
        result[ht] = bounds
    return result


# ── Global fit builders ─────────────────────────────────────────────────────


def build_global_bounds(bounds_dict, shared_idx, ts_idx, hold_types):
    shared_bounds = []
    for idx in shared_idx:
        lo = min(bounds_dict[ht][idx][0] for ht in hold_types)
        hi = max(bounds_dict[ht][idx][1] for ht in hold_types)
        shared_bounds.append((lo, hi))

    type_bounds = []
    for ht in hold_types:
        for idx in ts_idx:
            type_bounds.append(bounds_dict[ht][idx])

    return shared_bounds + type_bounds


def unpack_global(flat, shared_idx, ts_idx, hold_types, hold_type, n_full_params):
    """Unpack flat global vector into full per-hold params."""
    n_shared = len(shared_idx)
    shared = flat[:n_shared]
    type_idx = hold_types.index(hold_type)
    offset = n_shared + type_idx * len(ts_idx)
    specific = flat[offset : offset + len(ts_idx)]

    # Reassemble: interleave shared and type-specific in original param order
    result = [0.0] * n_full_params
    for i, idx in enumerate(shared_idx):
        result[idx] = shared[i]
    for i, idx in enumerate(ts_idx):
        result[idx] = specific[i]
    return np.array(result)


def global_fit(
    holds, hold_types, bounds_dict, shared_idx, ts_idx, shared_names,
    predict_fn, n_full_params, label,
):
    """Run a global fit with specified parameter split."""
    bounds = build_global_bounds(bounds_dict, shared_idx, ts_idx, hold_types)
    holds_by_type = {}
    for h in holds:
        holds_by_type.setdefault(h["type"], []).append(h)

    def objective(flat):
        total_sse = 0.0
        for ht in hold_types:
            params = unpack_global(flat, shared_idx, ts_idx, hold_types, ht, n_full_params)
            for h in holds_by_type[ht]:
                pred = predict_fn(h["t"], params)
                w = np.where(h["spo2"] < 95, 3.0, 1.0)
                total_sse += np.sum(w * (h["spo2"] - pred) ** 2)
        return total_sse

    n_params = len(bounds)
    n_types = len(hold_types)
    print(f"\n  {label}: {n_params} params "
          f"({len(shared_idx)} shared + {len(ts_idx)}×{n_types} type-specific)",
          flush=True)

    result = differential_evolution(
        objective, bounds, maxiter=6000, seed=42, tol=1e-10,
        polish=True, popsize=60, mutation=(0.5, 1.5), recombination=0.9,
    )
    print(f"  Converged: {result.success}, fun={result.fun:.2f}, nfev={result.nfev}",
          flush=True)
    return result.x


# ── Per-hold fits ───────────────────────────────────────────────────────────


def fit_perhold_richards(hold):
    bounds = RICHARDS_BOUNDS[hold["type"]]
    t, spo2 = hold["t"], hold["spo2"]

    def objective(arr):
        return np.sum((spo2 - predict_richards(t, arr)) ** 2)

    result = differential_evolution(
        objective, bounds, maxiter=3000, seed=42, tol=1e-10,
        polish=True, popsize=40, mutation=(0.5, 1.5), recombination=0.9,
    )
    return result.x


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    all_holds = load_all_holds()
    hold_labels = {h["id"]: f"{h['type']}#{h['id']}" for h in all_holds}

    EXCLUDED_IDS = {1}
    holds = [h for h in all_holds if h["id"] not in EXCLUDED_IDS]
    excluded = [h for h in all_holds if h["id"] in EXCLUDED_IDS]
    hold_types = sorted(set(h["type"] for h in holds))

    print("=" * 100)
    print("v5 EXPERIMENT 4: Global Fit with Gamma Constraint + Filter")
    print("=" * 100)
    print(f"\nLoaded {len(all_holds)} holds, excluded {len(excluded)} "
          f"({', '.join(hold_labels[h['id']] for h in excluded)}):")
    for h in all_holds:
        tag = " [EXCLUDED]" if h["id"] in EXCLUDED_IDS else ""
        print(f"  {hold_labels[h['id']]}: {len(h['t'])} pts, "
              f"SpO2 {h['spo2'].min():.0f}-{h['spo2'].max():.0f}%{tag}")

    # ── Per-hold Richards ───────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print("PER-HOLD RICHARDS BASELINE")
    print(f"{'='*100}")

    perhold_richards = {}
    for h in all_holds:
        label = hold_labels[h["id"]]
        print(f"  Fitting Richards on {label}...", end="", flush=True)
        perhold_richards[h["id"]] = fit_perhold_richards(h)
        pred = predict_richards(h["t"], perhold_richards[h["id"]])
        print(f" R²={compute_r2(h['spo2'], pred):.4f}", flush=True)

    # ── Global fits ─────────────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print("GLOBAL FITS")
    print(f"{'='*100}")

    # 1. CO2-Bohr baseline (no sensor model, 13p)
    flat_co2bohr = global_fit(
        holds, hold_types, CO2BOHR_BOUNDS,
        CO2BOHR_SHARED_IDX, CO2BOHR_TYPE_SPECIFIC_IDX,
        CO2BOHR_SHARED_NAMES,
        predict_co2bohr, 7, "CO2-Bohr (13p baseline)",
    )

    # 2. CO2-Bohr+Delay (v4 baseline, 14p)
    flat_delay = global_fit(
        holds, hold_types, CO2BOHR_DELAY_BOUNDS,
        DELAY_SHARED_IDX, DELAY_TYPE_SPECIFIC_IDX,
        DELAY_SHARED_NAMES,
        predict_co2bohr_delay, 8, "CO2-Bohr+Delay (14p v4 baseline)",
    )

    # 3. CO2-Bohr+Filter with gamma sweep
    gamma_sweeps = [
        ("free", (0.8, 2.5)),
        ("narrow", (0.9, 1.3)),
        ("fixed_1.0", (0.99, 1.01)),
    ]

    flat_filter = {}
    for gamma_label, gamma_range in gamma_sweeps:
        filter_bounds = make_filter_bounds(gamma_range)
        flat_filter[gamma_label] = global_fit(
            holds, hold_types, filter_bounds,
            FILTER_SHARED_IDX, FILTER_TYPE_SPECIFIC_IDX,
            FILTER_SHARED_NAMES,
            predict_co2bohr_filter, 8,
            f"CO2-Bohr+Filter γ={gamma_label} (14p)",
        )

    # ── Results ─────────────────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print("SHARED PARAMETER COMPARISON")
    print(f"{'='*100}")

    # CO2-Bohr shared
    n_shared_co2 = len(CO2BOHR_SHARED_IDX)
    shared_co2 = flat_co2bohr[:n_shared_co2]
    gb_co2 = build_global_bounds(CO2BOHR_BOUNDS, CO2BOHR_SHARED_IDX,
                                  CO2BOHR_TYPE_SPECIFIC_IDX, hold_types)
    print(f"\n  CO2-Bohr (no sensor):")
    for i, (name, val) in enumerate(zip(CO2BOHR_SHARED_NAMES, shared_co2)):
        lo, hi = gb_co2[i]
        flag = " ** AT BOUND **" if is_at_bound(val, lo, hi) else ""
        print(f"    {name:>12s} = {val:8.4f}  [{lo}, {hi}]{flag}")

    # CO2-Bohr+Delay shared
    n_shared_delay = len(DELAY_SHARED_IDX)
    shared_delay = flat_delay[:n_shared_delay]
    gb_delay = build_global_bounds(CO2BOHR_DELAY_BOUNDS, DELAY_SHARED_IDX,
                                    DELAY_TYPE_SPECIFIC_IDX, hold_types)
    print(f"\n  CO2-Bohr+Delay (v4):")
    for i, (name, val) in enumerate(zip(DELAY_SHARED_NAMES, shared_delay)):
        lo, hi = gb_delay[i]
        flag = " ** AT BOUND **" if is_at_bound(val, lo, hi) else ""
        print(f"    {name:>12s} = {val:8.4f}  [{lo}, {hi}]{flag}")

    # Filter variants
    for gamma_label, gamma_range in gamma_sweeps:
        flat = flat_filter[gamma_label]
        n_shared = len(FILTER_SHARED_IDX)
        shared = flat[:n_shared]
        filter_bounds = make_filter_bounds(gamma_range)
        gb = build_global_bounds(filter_bounds, FILTER_SHARED_IDX,
                                  FILTER_TYPE_SPECIFIC_IDX, hold_types)
        print(f"\n  CO2-Bohr+Filter γ={gamma_label}:")
        for i, (name, val) in enumerate(zip(FILTER_SHARED_NAMES, shared)):
            lo, hi = gb[i]
            flag = " ** AT BOUND **" if is_at_bound(val, lo, hi) else ""
            print(f"    {name:>12s} = {val:8.4f}  [{lo}, {hi}]{flag}")

    # ── Per-hold R² comparison ──────────────────────────────────────────────
    print(f"\n{'='*100}")
    print("PER-HOLD R² COMPARISON")
    print(f"{'='*100}")

    col_labels = ["CO2-B(13p)", "CO2-B+D(14p)"] + \
                 [f"Filt-γ{gl}(14p)" for gl, _ in gamma_sweeps] + \
                 ["Richards(PH)"]
    header = f"  {'Hold':<14s}" + "".join(f" {l:>14s}" for l in col_labels)
    print(header)
    print(f"  {'-'*14}" + "".join(f" {'-'*14}" for _ in col_labels))

    r2_lists = {label: [] for label in col_labels}

    for h in all_holds:
        label = hold_labels[h["id"]]
        tag = " (excl)" if h["id"] in EXCLUDED_IDS else ""

        # CO2-Bohr
        params_7 = unpack_global(flat_co2bohr, CO2BOHR_SHARED_IDX,
                                  CO2BOHR_TYPE_SPECIFIC_IDX, hold_types, h["type"], 7)
        r2_co2 = compute_r2(h["spo2"], predict_co2bohr(h["t"], params_7))

        # CO2-Bohr+Delay
        params_8d = unpack_global(flat_delay, DELAY_SHARED_IDX,
                                   DELAY_TYPE_SPECIFIC_IDX, hold_types, h["type"], 8)
        r2_delay = compute_r2(h["spo2"], predict_co2bohr_delay(h["t"], params_8d))

        # Filter variants
        r2_filters = []
        for gamma_label, gamma_range in gamma_sweeps:
            flat = flat_filter[gamma_label]
            params_8f = unpack_global(flat, FILTER_SHARED_IDX,
                                       FILTER_TYPE_SPECIFIC_IDX, hold_types, h["type"], 8)
            r2_f = compute_r2(h["spo2"], predict_co2bohr_filter(h["t"], params_8f))
            r2_filters.append(r2_f)

        # Richards
        r2_ri = compute_r2(h["spo2"], predict_richards(h["t"], perhold_richards[h["id"]]))

        all_r2 = [r2_co2, r2_delay] + r2_filters + [r2_ri]
        row = f"  {label + tag:<14s}" + "".join(f" {r:>14.4f}" for r in all_r2)
        print(row)

        for cl, r2 in zip(col_labels, all_r2):
            r2_lists[cl].append(r2)

    # Averages (fitted holds only)
    fitted_idx = [i for i, h in enumerate(all_holds) if h["id"] not in EXCLUDED_IDS]
    print(f"\n  {'Avg (fitted)':<14s}", end="")
    for cl in col_labels:
        avg = np.mean([r2_lists[cl][i] for i in fitted_idx])
        print(f" {avg:>14.4f}", end="")
    print()

    # ── Identifiability ─────────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print("IDENTIFIABILITY: PARAMS AT BOUNDS")
    print(f"{'='*100}")

    for fit_name, flat, s_names, s_idx, ts_names, ts_idx, bounds_dict, n_full in [
        ("CO2-Bohr", flat_co2bohr, CO2BOHR_SHARED_NAMES, CO2BOHR_SHARED_IDX,
         CO2BOHR_TYPE_SPECIFIC_NAMES, CO2BOHR_TYPE_SPECIFIC_IDX, CO2BOHR_BOUNDS, 7),
        ("CO2-Bohr+Delay", flat_delay, DELAY_SHARED_NAMES, DELAY_SHARED_IDX,
         DELAY_TYPE_SPECIFIC_NAMES, DELAY_TYPE_SPECIFIC_IDX, CO2BOHR_DELAY_BOUNDS, 8),
    ]:
        gb = build_global_bounds(bounds_dict, s_idx, ts_idx, hold_types)
        n_shared = len(s_idx)
        total_at_bound = 0

        print(f"\n  --- {fit_name} ({len(flat)} params) ---")
        for i, (name, val) in enumerate(zip(s_names, flat[:n_shared])):
            lo, hi = gb[i]
            if is_at_bound(val, lo, hi):
                total_at_bound += 1
                print(f"    {name}: {val:.4f} (bound: [{lo}, {hi}])")
        for ht in hold_types:
            params = unpack_global(flat, s_idx, ts_idx, hold_types, ht, n_full)
            for idx, name in zip(ts_idx, ts_names):
                val = params[idx]
                lo, hi = bounds_dict[ht][idx]
                if is_at_bound(val, lo, hi):
                    total_at_bound += 1
                    print(f"    {ht}/{name}: {val:.4f} (bound: [{lo}, {hi}])")
        print(f"  Total: {total_at_bound}/{len(flat)} at bounds")

    for gamma_label, gamma_range in gamma_sweeps:
        flat = flat_filter[gamma_label]
        filter_bounds = make_filter_bounds(gamma_range)
        gb = build_global_bounds(filter_bounds, FILTER_SHARED_IDX,
                                  FILTER_TYPE_SPECIFIC_IDX, hold_types)
        n_shared = len(FILTER_SHARED_IDX)
        total_at_bound = 0

        print(f"\n  --- CO2-Bohr+Filter γ={gamma_label} ({len(flat)} params) ---")
        for i, (name, val) in enumerate(zip(FILTER_SHARED_NAMES, flat[:n_shared])):
            lo, hi = gb[i]
            if is_at_bound(val, lo, hi):
                total_at_bound += 1
                print(f"    {name}: {val:.4f} (bound: [{lo}, {hi}])")
        for ht in hold_types:
            params = unpack_global(flat, FILTER_SHARED_IDX, FILTER_TYPE_SPECIFIC_IDX,
                                    hold_types, ht, 8)
            for idx, name in zip(FILTER_TYPE_SPECIFIC_IDX, FILTER_TYPE_SPECIFIC_NAMES):
                val = params[idx]
                lo, hi = filter_bounds[ht][idx]
                if is_at_bound(val, lo, hi):
                    total_at_bound += 1
                    print(f"    {ht}/{name}: {val:.4f} (bound: [{lo}, {hi}])")
        print(f"  Total: {total_at_bound}/{len(flat)} at bounds")

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print("GAMMA SWEEP SUMMARY")
    print(f"{'='*100}")

    print(f"\n  {'Variant':<28s} {'γ':>8s} {'tau_f':>8s} {'R² avg':>10s} "
          f"{'#at_bounds':>10s} {'total_p':>8s}")
    print(f"  {'-'*28} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")

    # CO2-Bohr
    avg_co2 = np.mean([r2_lists[col_labels[0]][i] for i in fitted_idx])
    print(f"  {'CO2-Bohr (no sensor)':<28s} {shared_co2[1]:>8.4f} {'N/A':>8s} "
          f"{avg_co2:>10.4f} {'':>10s} {len(flat_co2bohr):>8d}")

    # CO2-Bohr+Delay
    avg_delay = np.mean([r2_lists[col_labels[1]][i] for i in fitted_idx])
    print(f"  {'CO2-Bohr+Delay (v4)':<28s} {shared_delay[1]:>8.4f} {'N/A':>8s} "
          f"{avg_delay:>10.4f} {'':>10s} {len(flat_delay):>8d}")

    # Filter variants
    for i, (gamma_label, gamma_range) in enumerate(gamma_sweeps):
        flat = flat_filter[gamma_label]
        shared = flat[:len(FILTER_SHARED_IDX)]
        avg_f = np.mean([r2_lists[col_labels[2 + i]][j] for j in fitted_idx])
        print(f"  {'Filter γ=' + gamma_label:<28s} {shared[1]:>8.4f} {shared[4]:>8.2f} "
              f"{avg_f:>10.4f} {'':>10s} {len(flat):>8d}")

    # ── Plot ────────────────────────────────────────────────────────────────
    print("\nGenerating plot...", flush=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    sorted_holds = sorted(
        all_holds, key=lambda h: ({"FL": 0, "FRC": 1, "RV": 2}[h["type"]], h["id"])
    )

    colors_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for ax_idx, h in enumerate(sorted_holds):
        row, col = divmod(ax_idx, 3)
        ax = axes[row, col]
        label = hold_labels[h["id"]]
        is_excluded = h["id"] in EXCLUDED_IDS

        ax.plot(h["t"], h["spo2"], "k.", ms=2, alpha=0.4, label="Observed", zorder=1)

        # CO2-Bohr
        params_7 = unpack_global(flat_co2bohr, CO2BOHR_SHARED_IDX,
                                  CO2BOHR_TYPE_SPECIFIC_IDX, hold_types, h["type"], 7)
        pred = predict_co2bohr(h["t"], params_7)
        r2 = compute_r2(h["spo2"], pred)
        ax.plot(h["t"], pred, color=colors_list[0], lw=1.5, alpha=0.5, ls="--",
                label=f"CO2B (R²={r2:.3f})", zorder=2)

        # CO2-Bohr+Delay
        params_8d = unpack_global(flat_delay, DELAY_SHARED_IDX,
                                   DELAY_TYPE_SPECIFIC_IDX, hold_types, h["type"], 8)
        pred = predict_co2bohr_delay(h["t"], params_8d)
        r2 = compute_r2(h["spo2"], pred)
        ax.plot(h["t"], pred, color=colors_list[1], lw=1.5, alpha=0.5, ls="--",
                label=f"CO2B+D (R²={r2:.3f})", zorder=2)

        # Filter variants
        for i, (gamma_label, gamma_range) in enumerate(gamma_sweeps):
            flat = flat_filter[gamma_label]
            params_8f = unpack_global(flat, FILTER_SHARED_IDX,
                                       FILTER_TYPE_SPECIFIC_IDX, hold_types, h["type"], 8)
            pred = predict_co2bohr_filter(h["t"], params_8f)
            r2 = compute_r2(h["spo2"], pred)
            style = "-" if gamma_label == "narrow" else "--"
            ax.plot(h["t"], pred, color=colors_list[2 + i], lw=2.0 if gamma_label == "narrow" else 1.5,
                    alpha=0.9 if gamma_label == "narrow" else 0.6, ls=style,
                    label=f"Filt-γ{gamma_label} (R²={r2:.3f})", zorder=3)

        title = f"{label} [EXCLUDED]" if is_excluded else label
        ax.set_title(title, fontsize=12, fontweight="bold",
                     color="red" if is_excluded else "black")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("SpO2 (%)")
        ax.set_ylim(30, 105)
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "v5 Exp 4: Global Fit — CO2-Bohr vs CO2-Bohr+Delay vs CO2-Bohr+Filter (gamma sweep)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    out_path = Path(__file__).resolve().parent / "exp_v5_04_global_sensor.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    main()
