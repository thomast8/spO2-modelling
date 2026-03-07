"""
v6 Experiment 2: Structural Fixes — Per-Hold ICs, Constrained Recovery, d Sensitivity.

v6.01 showed that exponential re-oxygenation + global sensor params does NOT break
the d<->pao2_0 confounding:
  - d=29.96 (at bound), tau_reoxy=60 (at bound)
  - gamma=2.06 (worse, not better)
  - FRC#2 vs FRC#5 forced to share ICs but have very different recovery fits

This experiment applies three structural fixes:
  1. Alveolar Gas Equation coupling (pao2_0 derived from paco2_0 + delta_Aa)
  2. Gamma regularization (soft penalty pulling gamma toward 1.0)
  3. Constrained tau_reoxy [5, 15]

Sub-experiments:
  A: Type-specific ICs + AGEq + gamma reg + constrained tau_reoxy (16 params)
  B: Per-hold ICs + L2 regularization + all structural fixes (22 params)
  C: d-sensitivity sweep fixing d at 10, 15, 20, 25 using Exp B structure

Usage:
    cd backend && uv run python -u scripts/exp_v6_02_structural.py
"""

import csv
import io
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution
from scipy.signal import lfilter, lfilter_zi

DB_PATH = Path(__file__).resolve().parents[3] / "data" / "spo2.db"

P50_BASE = 26.6
P_EQ = 100.0
PACO2_NORMAL = 40.0
TAU_CLEAR_FIXED = 30.0
FIO2_PB_PH2O = 149.2  # FiO2 * (PB - PH2O) = 0.2093 * (760 - 47)
RQ = 0.8

EXCLUDED_IDS = {1}  # FL#1 excluded (only 2% SpO2 variation)

# Regularization strengths
LAMBDA_GAMMA = 500.0  # gamma -> 1.0 penalty
LAMBDA_REG = 10.0  # per-hold IC -> type-mean penalty


# ── CSV recovery data extraction (reused from v6.01) ────────────────────────


def _parse_time_to_seconds(time_str: str) -> int:
    parts = time_str.strip().split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    raise ValueError(f"Cannot parse time: {time_str!r}")


def load_holds_with_recovery(
    recovery_max_s: int = 90,
    recovery_spo2_ceiling: int = 97,
) -> list[dict]:
    """Load holds from DB with recovery data appended from raw CSV."""
    conn = sqlite3.connect(DB_PATH)
    holds_db = conn.execute(
        "SELECT id, hold_type FROM holds WHERE hold_type != 'untagged' ORDER BY id"
    ).fetchall()

    hold_data_db = {}
    for hold_id, hold_type in holds_db:
        rows = conn.execute(
            "SELECT elapsed_s, spo2, hr FROM hold_data WHERE hold_id = ? ORDER BY elapsed_s",
            (hold_id,),
        ).fetchall()
        if not rows:
            continue
        hold_data_db[hold_id] = {
            "id": hold_id,
            "type": hold_type,
            "t": np.array([r[0] for r in rows], dtype=float),
            "spo2": np.array([r[1] for r in rows], dtype=float),
            "hr": np.array([r[2] for r in rows], dtype=float),
        }

    csv_blob = conn.execute("SELECT csv_blob FROM sessions WHERE id = 1").fetchone()[0]
    conn.close()

    csv_text = csv_blob.decode("utf-8-sig")
    if csv_text.startswith("\ufeff"):
        csv_text = csv_text[1:]
    reader = csv.reader(io.StringIO(csv_text))
    rows = list(reader)

    bio_start = None
    for i, row in enumerate(rows):
        if row and row[0].strip() == "Biometrics":
            bio_start = i + 2
            break
    if bio_start is None:
        raise ValueError("No Biometrics section in CSV")

    intervals = []
    current_type = None
    current_block = []

    for row in rows[bio_start:]:
        if not row or len(row) < 5:
            continue
        itype = row[2].strip()
        try:
            hr = int(row[3].strip())
            spo2 = int(row[4].strip())
        except (ValueError, IndexError):
            continue

        if itype != current_type:
            if current_block:
                intervals.append((current_type, current_block))
            current_block = []
            current_type = itype

        current_block.append({
            "abs_time": row[0].strip(),
            "int_time": row[1].strip(),
            "type": itype,
            "hr": hr,
            "spo2": spo2,
        })

    if current_block:
        intervals.append((current_type, current_block))

    apnea_idx = 0
    result = []

    for i, (itype, block) in enumerate(intervals):
        if itype != "Apnea":
            continue

        duration = (
            _parse_time_to_seconds(block[-1]["abs_time"])
            - _parse_time_to_seconds(block[0]["abs_time"])
        )
        if duration < 30:
            continue

        apnea_idx += 1
        hold_id = apnea_idx

        if hold_id not in hold_data_db:
            continue

        db_hold = hold_data_db[hold_id]
        t_end = float(db_hold["t"][-1])

        t_recovery = np.array([], dtype=float)
        spo2_recovery = np.array([], dtype=float)
        hr_recovery = np.array([], dtype=float)

        if i + 1 < len(intervals):
            next_type, next_block = intervals[i + 1]
            if next_type in ("Rest", "Cooldown"):
                for r in next_block:
                    rt = _parse_time_to_seconds(r["int_time"])
                    if rt > recovery_max_s:
                        break
                    if rt > 5 and r["spo2"] >= recovery_spo2_ceiling:
                        break
                    t_recovery = np.append(t_recovery, t_end + rt + 1)
                    spo2_recovery = np.append(spo2_recovery, float(r["spo2"]))
                    hr_recovery = np.append(hr_recovery, float(r["hr"]))

        t_full = np.concatenate([db_hold["t"], t_recovery])
        spo2_full = np.concatenate([db_hold["spo2"], spo2_recovery])
        hr_full = np.concatenate([db_hold["hr"], hr_recovery])

        result.append({
            "id": hold_id,
            "type": db_hold["type"],
            "t": t_full,
            "spo2": spo2_full,
            "hr": hr_full,
            "t_end": t_end,
            "t_apnea": db_hold["t"],
            "spo2_apnea": db_hold["spo2"],
            "t_recovery": t_recovery,
            "spo2_recovery": spo2_recovery,
        })

    return result


# ── Physiology functions ─────────────────────────────────────────────────────


def alveolar_gas_equation(paco2_0, delta_aa):
    """Derive PAO2 from PaCO2 via alveolar gas equation + A-a gradient correction."""
    return FIO2_PB_PH2O - paco2_0 / RQ + delta_aa


def pao2_with_exp_recovery(t, pao2_0, pvo2, tau_washout, tau_reoxy, t_end):
    """Piecewise PAO2: exponential decay during apnea, exponential rise during recovery."""
    pao2_end = pvo2 + (pao2_0 - pvo2) * np.exp(-t_end / max(tau_washout, 0.01))
    return np.where(
        t <= t_end,
        pvo2 + (pao2_0 - pvo2) * np.exp(-t / max(tau_washout, 0.01)),
        P_EQ - (P_EQ - pao2_end) * np.exp(-(t - t_end) / max(tau_reoxy, 0.01)),
    )


def p50_with_exp_recovery(t, paco2_0, k_co2, tau_clear, t_end):
    """Piecewise P50: linear CO2 rise during apnea, exponential clearance during recovery."""
    paco2_end = paco2_0 + k_co2 * t_end
    paco2 = np.where(
        t <= t_end,
        paco2_0 + k_co2 * t,
        PACO2_NORMAL
        + (paco2_end - PACO2_NORMAL) * np.exp(-(t - t_end) / max(tau_clear, 0.01)),
    )
    return P50_BASE + 0.48 * (paco2 - PACO2_NORMAL)


def odc_severinghaus(pao2, p50_eff, gamma):
    """Severinghaus ODC with Bohr shift and gamma steepness."""
    pao2_virtual = pao2 * (P50_BASE / np.maximum(p50_eff, 0.01))
    pao2_adj = P50_BASE * (np.maximum(pao2_virtual, 0.01) / P50_BASE) ** gamma
    x = np.maximum(pao2_adj, 0.01)
    return 100.0 / (1.0 + 23400.0 / (x**3 + 150.0 * x))


# ── Predict with AGEq coupling ──────────────────────────────────────────────


def predict_recovery_sensor_ageq(t, params, t_end):
    """CO2-Bohr+Recovery+Sensor with Alveolar Gas Equation coupling.

    Params: [delta_Aa, pvo2, tau_washout, gamma, paco2_0, k_co2, r_offset, tau_reoxy, d, tau_f]
    PAO2_0 is derived from paco2_0 + delta_Aa via alveolar gas equation.
    """
    delta_aa, pvo2, tau_washout, gamma, paco2_0, k_co2, r_offset, tau_reoxy, d, tau_f = params
    pao2_0 = alveolar_gas_equation(paco2_0, delta_aa)
    pao2_0 = max(pao2_0, 1.0)  # safety clamp

    pao2 = pao2_with_exp_recovery(t, pao2_0, pvo2, tau_washout, tau_reoxy, t_end)
    p50 = p50_with_exp_recovery(t, paco2_0, k_co2, TAU_CLEAR_FIXED, t_end)
    sa = odc_severinghaus(pao2, p50, gamma)

    # Delay
    sa_delayed = np.interp(t - d, t, sa, left=sa[0])

    # IIR filter with preconditioned initial state
    dt = 1.0
    alpha = dt / (max(tau_f, 0.01) + dt)
    b_coeff = [alpha]
    a_coeff = [1.0, -(1.0 - alpha)]
    zi = lfilter_zi(b_coeff, a_coeff) * sa_delayed[0]
    s_meas, _ = lfilter(b_coeff, a_coeff, sa_delayed, zi=zi)

    return np.clip(s_meas + r_offset, 0.0, 100.0)


# ── Metrics ──────────────────────────────────────────────────────────────────


def compute_r2(obs, pred):
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def compute_rmse(obs, pred):
    return float(np.sqrt(np.mean((obs - pred) ** 2)))


def is_at_bound(val, lo, hi, tol=1e-3):
    return abs(val - lo) < tol or abs(val - hi) < tol


# ── Bounds ───────────────────────────────────────────────────────────────────

# Type-specific ICs (AGEq version): delta_Aa, tau_washout, paco2_0
TYPE_SPECIFIC_BOUNDS = {
    "FL": [(-15, 15), (50, 250), (20, 50)],
    "FRC": [(-15, 15), (20, 100), (25, 50)],
    "RV": [(-15, 15), (10, 80), (30, 55)],
}
TYPE_SPECIFIC_NAMES = ["delta_Aa", "tau_washout", "paco2_0"]

# Shared params: pvo2, gamma, k_co2, r_offset, tau_reoxy, d, tau_f
SHARED_BOUNDS = [(15, 50), (0.8, 2.5), (0.02, 0.25), (-3, 3), (5, 15), (1, 30), (1, 30)]
SHARED_NAMES = ["pvo2", "gamma", "k_co2", "r_offset", "tau_reoxy", "d", "tau_f"]

N_SHARED = len(SHARED_BOUNDS)
N_TS = len(TYPE_SPECIFIC_NAMES)


# ── Exp A: Type-specific ICs (same as v6.01 C but with structural fixes) ────


def build_type_bounds(shared_bounds, hold_types):
    """Build flat bounds: shared + type-specific for each type."""
    bounds = list(shared_bounds)
    for ht in hold_types:
        bounds.extend(TYPE_SPECIFIC_BOUNDS[ht])
    return bounds


def unpack_type_params(flat, hold_types, hold_type):
    """Unpack flat vector into (shared, type_specific) for a given hold type."""
    shared = flat[:N_SHARED]
    type_idx = hold_types.index(hold_type)
    offset = N_SHARED + type_idx * N_TS
    specific = flat[offset : offset + N_TS]
    return shared, specific


def assemble_ageq_params(shared, specific):
    """Assemble full 10-param vector for predict_recovery_sensor_ageq.

    shared: [pvo2, gamma, k_co2, r_offset, tau_reoxy, d, tau_f]
    specific: [delta_Aa, tau_washout, paco2_0]
    -> [delta_Aa, pvo2, tau_washout, gamma, paco2_0, k_co2, r_offset, tau_reoxy, d, tau_f]
    """
    pvo2, gamma, k_co2, r_offset, tau_reoxy, d, tau_f = shared
    delta_aa, tau_washout, paco2_0 = specific
    return np.array([
        delta_aa, pvo2, tau_washout, gamma, paco2_0, k_co2, r_offset, tau_reoxy, d, tau_f,
    ])


def run_exp_a(holds_by_type, hold_types):
    """Exp A: Type-specific ICs + AGEq + gamma reg + constrained tau_reoxy."""
    bounds = build_type_bounds(SHARED_BOUNDS, hold_types)
    n_types = len(hold_types)
    n_total = len(bounds)

    def objective(flat):
        total_sse = 0.0
        gamma = flat[1]  # gamma is 2nd shared param
        for ht in hold_types:
            shared, specific = unpack_type_params(flat, hold_types, ht)
            params = assemble_ageq_params(shared, specific)
            for h in holds_by_type[ht]:
                pred = predict_recovery_sensor_ageq(h["t"], params, h["t_end"])
                w = np.where(h["spo2"] < 95, 3.0, 1.0)
                total_sse += np.sum(w * (h["spo2"] - pred) ** 2)
        # Gamma regularization
        total_sse += LAMBDA_GAMMA * (gamma - 1.0) ** 2
        return total_sse

    print(
        f"\n  Exp A: {n_total} params "
        f"({N_SHARED} shared + {N_TS}x{n_types} type-specific)",
        flush=True,
    )

    result = differential_evolution(
        objective, bounds, maxiter=6000, seed=42, tol=1e-10,
        polish=True, popsize=60, mutation=(0.5, 1.5), recombination=0.9,
    )
    print(f"  Converged: {result.success}, fun={result.fun:.2f}, nfev={result.nfev}", flush=True)
    return result.x, result.success


# ── Exp B: Per-hold ICs with L2 regularization ──────────────────────────────


def build_perhold_bounds(shared_bounds, holds):
    """Build flat bounds: shared + per-hold ICs."""
    bounds = list(shared_bounds)
    for h in holds:
        bounds.extend(TYPE_SPECIFIC_BOUNDS[h["type"]])
    return bounds


def unpack_perhold_params(flat, hold_idx):
    """Unpack flat vector into (shared, hold_specific) for hold at index hold_idx."""
    shared = flat[:N_SHARED]
    offset = N_SHARED + hold_idx * N_TS
    specific = flat[offset : offset + N_TS]
    return shared, specific


def run_exp_b(holds_by_type, hold_types, fit_holds):
    """Exp B: Per-hold ICs + L2 regularization + AGEq + gamma reg."""
    bounds = build_perhold_bounds(SHARED_BOUNDS, fit_holds)
    n_holds = len(fit_holds)
    n_total = len(bounds)

    # Build index mapping: hold -> index in fit_holds
    hold_id_to_idx = {h["id"]: i for i, h in enumerate(fit_holds)}

    # Build type groups for regularization
    type_groups = {}
    for i, h in enumerate(fit_holds):
        type_groups.setdefault(h["type"], []).append(i)

    def objective(flat):
        total_sse = 0.0
        gamma = flat[1]

        for i, h in enumerate(fit_holds):
            shared, specific = unpack_perhold_params(flat, i)
            params = assemble_ageq_params(shared, specific)
            pred = predict_recovery_sensor_ageq(h["t"], params, h["t_end"])
            w = np.where(h["spo2"] < 95, 3.0, 1.0)
            total_sse += np.sum(w * (h["spo2"] - pred) ** 2)

        # Gamma regularization
        total_sse += LAMBDA_GAMMA * (gamma - 1.0) ** 2

        # Per-hold IC regularization toward type means
        penalty = 0.0
        for ht, indices in type_groups.items():
            if len(indices) < 2:
                continue
            for p_offset in range(N_TS):
                values = [flat[N_SHARED + idx * N_TS + p_offset] for idx in indices]
                mean_val = np.mean(values)
                penalty += LAMBDA_REG * sum((v - mean_val) ** 2 for v in values)
        total_sse += penalty

        return total_sse

    print(f"\n  Exp B: {n_total} params ({N_SHARED} shared + {N_TS}x{n_holds} per-hold)", flush=True)

    result = differential_evolution(
        objective, bounds, maxiter=6000, seed=42, tol=1e-10,
        polish=True, popsize=60, mutation=(0.5, 1.5), recombination=0.9,
    )
    print(f"  Converged: {result.success}, fun={result.fun:.2f}, nfev={result.nfev}", flush=True)
    return result.x, result.success


# ── Exp C: d-sensitivity sweep using Exp B structure ────────────────────────


def run_exp_c(holds_by_type, hold_types, fit_holds, d_values):
    """Exp C: Fix d at each value, re-optimize using Exp B structure."""
    n_holds = len(fit_holds)
    d_idx_in_shared = 5  # d is 6th shared param (0-indexed)

    type_groups = {}
    for i, h in enumerate(fit_holds):
        type_groups.setdefault(h["type"], []).append(i)

    results_per_d = {}

    for d_fixed in d_values:
        print(f"\n    d={d_fixed:.1f}:", flush=True)

        # Build bounds with d fixed
        fixed_shared = list(SHARED_BOUNDS)
        fixed_shared[d_idx_in_shared] = (d_fixed - 0.01, d_fixed + 0.01)
        bounds = build_perhold_bounds(fixed_shared, fit_holds)

        def objective(flat, _type_groups=type_groups):
            total_sse = 0.0
            gamma = flat[1]

            for i, h in enumerate(fit_holds):
                shared, specific = unpack_perhold_params(flat, i)
                params = assemble_ageq_params(shared, specific)
                pred = predict_recovery_sensor_ageq(h["t"], params, h["t_end"])
                w = np.where(h["spo2"] < 95, 3.0, 1.0)
                total_sse += np.sum(w * (h["spo2"] - pred) ** 2)

            total_sse += LAMBDA_GAMMA * (gamma - 1.0) ** 2

            penalty = 0.0
            for ht, indices in _type_groups.items():
                if len(indices) < 2:
                    continue
                for p_offset in range(N_TS):
                    values = [flat[N_SHARED + idx * N_TS + p_offset] for idx in indices]
                    mean_val = np.mean(values)
                    penalty += LAMBDA_REG * sum((v - mean_val) ** 2 for v in values)
            total_sse += penalty

            return total_sse

        result = differential_evolution(
            objective, bounds, maxiter=6000, seed=42, tol=1e-10,
            polish=True, popsize=60, mutation=(0.5, 1.5), recombination=0.9,
        )
        print(f"      loss={result.fun:.2f}, nfev={result.nfev}", flush=True)
        results_per_d[d_fixed] = {"flat": result.x, "loss": result.fun, "success": result.success}

    return results_per_d


# ── Evaluation ───────────────────────────────────────────────────────────────


def evaluate_type_specific(flat, all_holds, hold_types, label):
    """Evaluate Exp A (type-specific ICs) on all holds."""
    results = []
    for h in all_holds:
        if h["type"] not in hold_types:
            continue
        shared, specific = unpack_type_params(flat, hold_types, h["type"])
        params = assemble_ageq_params(shared, specific)

        pred_full = predict_recovery_sensor_ageq(h["t"], params, h["t_end"])
        r2_full = compute_r2(h["spo2"], pred_full)

        pred_apnea = predict_recovery_sensor_ageq(h["t_apnea"], params, h["t_end"])
        r2_apnea = compute_r2(h["spo2_apnea"], pred_apnea)

        r2_recovery = None
        if len(h["t_recovery"]) > 3:
            pred_rec = predict_recovery_sensor_ageq(h["t_recovery"], params, h["t_end"])
            r2_recovery = compute_r2(h["spo2_recovery"], pred_rec)

        pao2_0 = alveolar_gas_equation(specific[2], specific[0])  # paco2_0, delta_Aa

        ts_bounds = TYPE_SPECIFIC_BOUNDS[h["type"]]
        at_bounds = []
        for val, (lo, hi), name in zip(specific, ts_bounds, TYPE_SPECIFIC_NAMES, strict=True):
            if is_at_bound(val, lo, hi):
                at_bounds.append(f"{name}={'lo' if abs(val - lo) < 1e-3 else 'hi'}")

        results.append({
            "variant": label,
            "hold_id": h["id"],
            "hold_type": h["type"],
            "r2_full": r2_full,
            "r2_apnea": r2_apnea,
            "r2_recovery": r2_recovery,
            "at_bounds_ts": at_bounds,
            "params": params,
            "pred_full": pred_full,
            "shared": shared,
            "specific": specific,
            "pao2_0_derived": pao2_0,
            "is_excluded": h["id"] in EXCLUDED_IDS,
        })
    return results


def evaluate_perhold(flat, all_holds, fit_holds, label):
    """Evaluate Exp B/C (per-hold ICs) on all holds."""
    hold_id_to_idx = {h["id"]: i for i, h in enumerate(fit_holds)}
    results = []

    for h in all_holds:
        if h["id"] not in hold_id_to_idx:
            # Excluded hold — use shared params only, can't predict meaningfully
            # but include for completeness
            shared = flat[:N_SHARED]
            # Use type-mean ICs from fitted holds of same type
            type_indices = [
                i for i, fh in enumerate(fit_holds) if fh["type"] == h["type"]
            ]
            if not type_indices:
                continue
            avg_specific = np.mean(
                [flat[N_SHARED + idx * N_TS : N_SHARED + (idx + 1) * N_TS] for idx in type_indices],
                axis=0,
            )
            params = assemble_ageq_params(shared, avg_specific)
            pred_full = predict_recovery_sensor_ageq(h["t"], params, h["t_end"])
            r2_full = compute_r2(h["spo2"], pred_full)
            pred_apnea = predict_recovery_sensor_ageq(h["t_apnea"], params, h["t_end"])
            r2_apnea = compute_r2(h["spo2_apnea"], pred_apnea)
            r2_recovery = None
            if len(h["t_recovery"]) > 3:
                pred_rec = predict_recovery_sensor_ageq(h["t_recovery"], params, h["t_end"])
                r2_recovery = compute_r2(h["spo2_recovery"], pred_rec)
            pao2_0 = alveolar_gas_equation(avg_specific[2], avg_specific[0])

            results.append({
                "variant": label,
                "hold_id": h["id"],
                "hold_type": h["type"],
                "r2_full": r2_full,
                "r2_apnea": r2_apnea,
                "r2_recovery": r2_recovery,
                "at_bounds_ts": [],
                "params": params,
                "pred_full": pred_full,
                "shared": shared,
                "specific": avg_specific,
                "pao2_0_derived": pao2_0,
                "is_excluded": True,
            })
            continue

        idx = hold_id_to_idx[h["id"]]
        shared, specific = unpack_perhold_params(flat, idx)
        params = assemble_ageq_params(shared, specific)

        pred_full = predict_recovery_sensor_ageq(h["t"], params, h["t_end"])
        r2_full = compute_r2(h["spo2"], pred_full)

        pred_apnea = predict_recovery_sensor_ageq(h["t_apnea"], params, h["t_end"])
        r2_apnea = compute_r2(h["spo2_apnea"], pred_apnea)

        r2_recovery = None
        if len(h["t_recovery"]) > 3:
            pred_rec = predict_recovery_sensor_ageq(h["t_recovery"], params, h["t_end"])
            r2_recovery = compute_r2(h["spo2_recovery"], pred_rec)

        pao2_0 = alveolar_gas_equation(specific[2], specific[0])

        ts_bounds = TYPE_SPECIFIC_BOUNDS[h["type"]]
        at_bounds = []
        for val, (lo, hi), name in zip(specific, ts_bounds, TYPE_SPECIFIC_NAMES, strict=True):
            if is_at_bound(val, lo, hi):
                at_bounds.append(f"{name}={'lo' if abs(val - lo) < 1e-3 else 'hi'}")

        results.append({
            "variant": label,
            "hold_id": h["id"],
            "hold_type": h["type"],
            "r2_full": r2_full,
            "r2_apnea": r2_apnea,
            "r2_recovery": r2_recovery,
            "at_bounds_ts": at_bounds,
            "params": params,
            "pred_full": pred_full,
            "shared": shared,
            "specific": specific,
            "pao2_0_derived": pao2_0,
            "is_excluded": h["id"] in EXCLUDED_IDS,
        })
    return results


# ── Output ───────────────────────────────────────────────────────────────────


def print_comparison_table(all_eval_results, variant_names):
    """Print per-hold R2 comparison across variants."""
    print(f"\n{'='*120}")
    print("PER-HOLD R2 COMPARISON")
    print(f"{'='*120}")

    header = f"  {'Hold':<18s}"
    for vn in variant_names:
        header += f" {'R2f':>6s} {'R2a':>6s} {'R2r':>6s}"
    print(header)

    sub = f"  {'':18s}"
    for vn in variant_names:
        short = vn[:20]
        sub += f" {short:>20s}"
    print(sub)
    print(f"  {'-'*18}" + f" {'-'*20}" * len(variant_names))

    by_hold = {}
    for r in all_eval_results:
        key = (r["hold_id"], r["hold_type"])
        by_hold.setdefault(key, {})[r["variant"]] = r

    for (hid, htype), variants in sorted(by_hold.items()):
        tag = " (excl)" if hid in EXCLUDED_IDS else ""
        label = f"{htype}#{hid}{tag}"
        row = f"  {label:<18s}"
        for vn in variant_names:
            r = variants.get(vn)
            if r:
                r2f = f"{r['r2_full']:.4f}"
                r2a = f"{r['r2_apnea']:.4f}"
                r2r = f"{r['r2_recovery']:.4f}" if r["r2_recovery"] is not None else "N/A"
                row += f" {r2f:>6s} {r2a:>6s} {r2r:>6s}"
            else:
                row += f" {'N/A':>6s} {'N/A':>6s} {'N/A':>6s}"
        print(row)

    # Averages (fitted holds only)
    print(f"\n  {'Avg (fitted)':<18s}", end="")
    for vn in variant_names:
        r2fs, r2as, r2rs = [], [], []
        for (hid, _), variants in sorted(by_hold.items()):
            if hid in EXCLUDED_IDS:
                continue
            r = variants.get(vn)
            if r:
                r2fs.append(r["r2_full"])
                r2as.append(r["r2_apnea"])
                if r["r2_recovery"] is not None:
                    r2rs.append(r["r2_recovery"])
        avg_f = np.mean(r2fs) if r2fs else float("nan")
        avg_a = np.mean(r2as) if r2as else float("nan")
        avg_r = np.mean(r2rs) if r2rs else float("nan")
        print(f" {avg_f:>6.4f} {avg_a:>6.4f} {avg_r:>6.4f}", end="")
    print()


def print_shared_params(flat, shared_bounds, shared_names, label):
    """Print shared parameter values with bounds."""
    n_shared = len(shared_bounds)
    shared = flat[:n_shared]
    print(f"\n  {label}:")
    for name, val, (lo, hi) in zip(shared_names, shared, shared_bounds):
        flag = " ** AT BOUND **" if is_at_bound(val, lo, hi) else ""
        print(f"    {name:>12s} = {val:8.4f}  [{lo:>6.2f}, {hi:>6.2f}]{flag}")


def print_perhold_params(flat, fit_holds, label):
    """Print per-hold IC parameters with derived PAO2."""
    print(f"\n  {label} — Per-hold ICs:")
    for i, h in enumerate(fit_holds):
        _, specific = unpack_perhold_params(flat, i)
        delta_aa, tau_washout, paco2_0 = specific
        pao2_0 = alveolar_gas_equation(paco2_0, delta_aa)
        ts_bounds = TYPE_SPECIFIC_BOUNDS[h["type"]]

        at_bounds = []
        for val, (lo, hi), name in zip(specific, ts_bounds, TYPE_SPECIFIC_NAMES, strict=True):
            if is_at_bound(val, lo, hi):
                at_bounds.append(f"{name}={'lo' if abs(val - lo) < 1e-3 else 'hi'}")

        bound_str = f"  [{', '.join(at_bounds)}]" if at_bounds else ""
        print(
            f"    {h['type']}#{h['id']}: "
            f"delta_Aa={delta_aa:+.2f}, tau_w={tau_washout:.1f}, "
            f"paco2_0={paco2_0:.1f}, PAO2_0={pao2_0:.1f}{bound_str}"
        )


def print_type_specific_params_a(flat, hold_types, all_holds, label):
    """Print type-specific params for Exp A."""
    print(f"\n  {label} — Type-specific ICs:")
    for ht in hold_types:
        _, specific = unpack_type_params(flat, hold_types, ht)
        delta_aa, tau_washout, paco2_0 = specific
        pao2_0 = alveolar_gas_equation(paco2_0, delta_aa)
        ts_bounds = TYPE_SPECIFIC_BOUNDS[ht]
        at_bounds = []
        for val, (lo, hi), name in zip(specific, ts_bounds, TYPE_SPECIFIC_NAMES, strict=True):
            if is_at_bound(val, lo, hi):
                at_bounds.append(f"{name}={'lo' if abs(val - lo) < 1e-3 else 'hi'}")
        bound_str = f"  [{', '.join(at_bounds)}]" if at_bounds else ""
        print(
            f"    {ht}: delta_Aa={delta_aa:+.2f}, tau_w={tau_washout:.1f}, "
            f"paco2_0={paco2_0:.1f}, PAO2_0={pao2_0:.1f}{bound_str}"
        )


def plot_per_hold_detail(eval_results_by_variant, all_holds, variant_names, output_path):
    """Detailed per-hold plots with time axis and apnea end marker."""
    by_hold_results = {}
    for vn, results in eval_results_by_variant.items():
        for r in results:
            by_hold_results.setdefault(r["hold_id"], {})[vn] = r

    holds_dict = {h["id"]: h for h in all_holds}
    hold_ids = sorted(by_hold_results.keys())
    n_holds = len(hold_ids)

    fig, axes = plt.subplots(n_holds, 1, figsize=(16, 4.5 * n_holds), squeeze=False)
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

    for idx, hid in enumerate(hold_ids):
        ax = axes[idx, 0]
        h = holds_dict[hid]
        variants = by_hold_results[hid]

        ax.plot(h["t"], h["spo2"], "k.", markersize=2, alpha=0.5, label="Observed")
        ax.axvline(x=h["t_end"], color="red", linestyle="--", alpha=0.5, label="Apnea end")

        for i, vn in enumerate(variant_names):
            r = variants.get(vn)
            if not r:
                continue

            if len(r["pred_full"]) == len(h["t"]):
                t_plot = h["t"]
            elif len(r["pred_full"]) == len(h["t_apnea"]):
                t_plot = h["t_apnea"]
            else:
                t_plot = np.arange(len(r["pred_full"]))

            r2_str = f"R2={r['r2_full']:.4f}"
            if r["r2_recovery"] is not None:
                r2_str += f", rec={r['r2_recovery']:.4f}"
            label = f"{vn} ({r2_str})"
            ax.plot(
                t_plot, r["pred_full"], color=colors[i % len(colors)],
                linewidth=1.5, alpha=0.8, label=label,
            )

        tag = " [EXCLUDED]" if hid in EXCLUDED_IDS else ""
        ax.set_title(
            f"{h['type']}#{hid}{tag}", fontsize=13, fontweight="bold",
            color="red" if hid in EXCLUDED_IDS else "black",
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("SpO2 (%)")
        ax.set_ylim(30, 105)
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "v6.02: Structural Fixes — Per-Hold Fit Detail",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {output_path}")


def plot_d_sensitivity(results_per_d, d_values, output_path):
    """Plot shared params vs fixed d for Exp C."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    # Params to plot: gamma, pvo2, k_co2, tau_f, tau_reoxy, loss
    param_configs = [
        (1, "gamma", "Gamma (ODC steepness)"),
        (0, "pvo2", "PvO2 (mmHg)"),
        (2, "k_co2", "k_CO2 (mmHg/s)"),
        (6, "tau_f", "tau_f (filter, s)"),
        (4, "tau_reoxy", "tau_reoxy (s)"),
        (None, "loss", "Total Loss (weighted SSE + penalties)"),
    ]

    d_list = sorted(results_per_d.keys())

    for ax_idx, (param_idx, name, title) in enumerate(param_configs):
        ax = axes[ax_idx]
        if param_idx is not None:
            values = [results_per_d[d]["flat"][param_idx] for d in d_list]
        else:
            values = [results_per_d[d]["loss"] for d in d_list]

        ax.plot(d_list, values, "o-", color="#1f77b4", linewidth=2, markersize=6)
        ax.set_xlabel("d (fixed delay, s)", fontsize=10)
        ax.set_ylabel(name, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "v6.02 Exp C: d-Sensitivity Sweep (shared params vs fixed d)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"d-sensitivity plot saved to {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    print("=" * 120)
    print("v6.02: Structural Fixes — Per-Hold ICs, AGEq Coupling, Constrained Recovery")
    print("=" * 120)

    # ── Load data ────────────────────────────────────────────────────────────
    print("\nLoading holds with recovery data...")
    all_holds = load_holds_with_recovery()

    for h in all_holds:
        n_apnea = len(h["t_apnea"])
        n_rec = len(h["t_recovery"])
        tag = " [EXCLUDED]" if h["id"] in EXCLUDED_IDS else ""
        rec_range = ""
        if n_rec > 0:
            rec_range = (
                f", recovery SpO2 {h['spo2_recovery'].min():.0f}"
                f"-{h['spo2_recovery'][-1]:.0f}%"
            )
        print(
            f"  {h['type']}#{h['id']}{tag}: {n_apnea} apnea + {n_rec} recovery pts "
            f"(t_end={h['t_end']:.0f}s{rec_range})"
        )

    fit_holds = [h for h in all_holds if h["id"] not in EXCLUDED_IDS]
    hold_types = sorted(set(h["type"] for h in fit_holds))
    holds_by_type = {}
    for h in fit_holds:
        holds_by_type.setdefault(h["type"], []).append(h)

    print(
        f"\nFitting on {len(fit_holds)} holds (types: {hold_types}), "
        f"excluding {sum(1 for h in all_holds if h['id'] in EXCLUDED_IDS)} hold(s)"
    )
    print(f"\nStructural changes active:")
    print(f"  - Alveolar Gas Equation coupling (PAO2_0 = {FIO2_PB_PH2O} - PaCO2/{RQ} + delta_Aa)")
    print(f"  - Gamma regularization (lambda={LAMBDA_GAMMA}, pulling toward 1.0)")
    print(f"  - tau_reoxy constrained to [{SHARED_BOUNDS[4][0]}, {SHARED_BOUNDS[4][1]}]")

    # ── Exp A ────────────────────────────────────────────────────────────────
    print(f"\n{'='*120}")
    print("EXP A: Type-specific ICs + AGEq + gamma reg + constrained tau_reoxy")
    print(f"{'='*120}")

    flat_a, conv_a = run_exp_a(holds_by_type, hold_types)

    print_shared_params(flat_a, SHARED_BOUNDS, SHARED_NAMES, "Exp A")
    print_type_specific_params_a(flat_a, hold_types, all_holds, "Exp A")

    eval_a = evaluate_type_specific(flat_a, all_holds, hold_types, "A:TypeICs")

    # ── Exp B ────────────────────────────────────────────────────────────────
    print(f"\n{'='*120}")
    print("EXP B: Per-hold ICs + L2 regularization + AGEq + gamma reg")
    print(f"{'='*120}")

    flat_b, conv_b = run_exp_b(holds_by_type, hold_types, fit_holds)

    print_shared_params(flat_b, SHARED_BOUNDS, SHARED_NAMES, "Exp B")
    print_perhold_params(flat_b, fit_holds, "Exp B")

    eval_b = evaluate_perhold(flat_b, all_holds, fit_holds, "B:PerHoldICs")

    # ── Exp C ────────────────────────────────────────────────────────────────
    print(f"\n{'='*120}")
    print("EXP C: d-Sensitivity Sweep (fix d, re-optimize using Exp B structure)")
    print(f"{'='*120}")

    d_values = [10.0, 15.0, 20.0, 25.0]
    results_per_d = run_exp_c(holds_by_type, hold_types, fit_holds, d_values)

    # Evaluate each d-fixed run
    eval_c_by_d = {}
    for d_fixed, res in sorted(results_per_d.items()):
        label = f"C:d={d_fixed:.0f}"
        eval_c_by_d[d_fixed] = evaluate_perhold(res["flat"], all_holds, fit_holds, label)

    # ── Print results ────────────────────────────────────────────────────────
    variant_names_ab = ["A:TypeICs", "B:PerHoldICs"]
    all_eval_ab = eval_a + eval_b
    print_comparison_table(all_eval_ab, variant_names_ab)

    # Exp C comparison
    print(f"\n{'='*120}")
    print("EXP C: d-SENSITIVITY RESULTS")
    print(f"{'='*120}")

    print(f"\n  {'d':>5s} | {'loss':>10s} | {'gamma':>8s} | {'pvo2':>8s} | "
          f"{'k_co2':>8s} | {'tau_f':>8s} | {'tau_reoxy':>10s} | {'r_offset':>10s}")
    print(f"  {'-'*5}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}")
    for d_fixed in sorted(results_per_d.keys()):
        res = results_per_d[d_fixed]
        shared = res["flat"][:N_SHARED]
        pvo2, gamma, k_co2, r_offset, tau_reoxy, d, tau_f = shared
        print(
            f"  {d_fixed:5.1f} | {res['loss']:10.2f} | {gamma:8.4f} | {pvo2:8.2f} | "
            f"{k_co2:8.4f} | {tau_f:8.2f} | {tau_reoxy:10.2f} | {r_offset:10.4f}"
        )

    # Per-hold R2 for each d value
    print(f"\n  Per-hold R2(apnea) / R2(recovery) by d:")
    print(f"  {'Hold':<12s}", end="")
    for d_fixed in sorted(results_per_d.keys()):
        print(f" | {'d=' + str(int(d_fixed)):>12s}", end="")
    print()
    print(f"  {'-'*12}" + f"-+-{'-'*12}" * len(results_per_d))

    for h in fit_holds:
        row = f"  {h['type']}#{h['id']:<8d}"
        for d_fixed in sorted(results_per_d.keys()):
            eval_list = eval_c_by_d[d_fixed]
            r = next((r for r in eval_list if r["hold_id"] == h["id"]), None)
            if r:
                r2a_s = f"{r['r2_apnea']:.3f}"
                r2r_s = f"{r['r2_recovery']:.3f}" if r["r2_recovery"] is not None else "N/A"
                row += f" | {r2a_s:>5s}/{r2r_s:>5s}"
            else:
                row += f" |      N/A    "
        print(row)

    # ── Identifiability summary ──────────────────────────────────────────────
    print(f"\n{'='*120}")
    print("IDENTIFIABILITY SUMMARY")
    print(f"{'='*120}")

    for label, flat in [("A:TypeICs", flat_a), ("B:PerHoldICs", flat_b)]:
        shared = flat[:N_SHARED]
        n_at = 0
        at_list = []
        for name, val, (lo, hi) in zip(SHARED_NAMES, shared, SHARED_BOUNDS):
            if is_at_bound(val, lo, hi):
                n_at += 1
                at_list.append(f"{name}={'lo' if abs(val - lo) < 1e-3 else 'hi'}")
        n_total = len(flat)
        print(f"\n  {label}: {n_at}/{n_total} shared params at bounds")
        if at_list:
            for ab in at_list:
                print(f"    {ab}")

    # ── Plots ────────────────────────────────────────────────────────────────
    output_dir = Path(__file__).resolve().parent

    eval_results_by_variant = {"A:TypeICs": eval_a, "B:PerHoldICs": eval_b}
    plot_per_hold_detail(
        eval_results_by_variant, all_holds, variant_names_ab,
        output_dir / "exp_v6_02_structural.png",
    )

    plot_d_sensitivity(
        results_per_d, d_values,
        output_dir / "exp_v6_02_d_sensitivity.png",
    )

    # ── Success criteria check ───────────────────────────────────────────────
    print(f"\n{'='*120}")
    print("SUCCESS CRITERIA CHECK")
    print(f"{'='*120}")

    shared_a = flat_a[:N_SHARED]
    shared_b = flat_b[:N_SHARED]

    # 1. Exp A: d moves away from upper bound
    d_a = shared_a[5]
    d_interior_a = not is_at_bound(d_a, 1, 30)
    print(f"\n  1. [Exp A] d={d_a:.2f} interior (not at bound): {'PASS' if d_interior_a else 'FAIL'}")

    # 2. Exp B: FRC holds get different ICs
    frc_holds = [h for h in fit_holds if h["type"] == "FRC"]
    if len(frc_holds) >= 2:
        frc_ics = []
        for h in frc_holds:
            idx = [fh["id"] for fh in fit_holds].index(h["id"])
            _, specific = unpack_perhold_params(flat_b, idx)
            frc_ics.append(specific.copy())
        ic_diff = np.linalg.norm(frc_ics[0] - frc_ics[1])
        print(f"  2. [Exp B] FRC IC divergence (L2 norm): {ic_diff:.2f} "
              f"({'PASS (different ICs)' if ic_diff > 1.0 else 'FAIL (too similar)'})")

    # 3. Exp B: gamma closer to 1.0 than v6.01 (was 2.06)
    gamma_b = shared_b[1]
    gamma_improvement = abs(2.06 - 1.0) - abs(gamma_b - 1.0)
    print(f"  3. [Exp B] gamma={gamma_b:.4f} (v6.01 was 2.06, improvement={gamma_improvement:.4f}): "
          f"{'PASS' if gamma_improvement > 0 else 'FAIL'}")

    # 4. Recovery R2 > 0
    rec_r2s = [r["r2_recovery"] for r in eval_b if r["r2_recovery"] is not None and not r["is_excluded"]]
    rec_avg = np.mean(rec_r2s) if rec_r2s else float("nan")
    rec_pass = all(r > 0 for r in rec_r2s) if rec_r2s else False
    print(f"  4. [Exp B] R2(recovery) avg={rec_avg:.4f}, all>0: {'PASS' if rec_pass else 'FAIL'}")

    # 5. R2(apnea) >= 0.95
    apn_r2s = [r["r2_apnea"] for r in eval_b if not r["is_excluded"]]
    apn_avg = np.mean(apn_r2s) if apn_r2s else float("nan")
    apn_pass = apn_avg >= 0.95
    print(f"  5. [Exp B] R2(apnea) avg={apn_avg:.4f} >= 0.95: {'PASS' if apn_pass else 'FAIL'}")

    # 6. Exp C: d is nuisance if gamma/pvo2/k_co2 stable across d values
    d_list = sorted(results_per_d.keys())
    gammas_c = [results_per_d[d]["flat"][1] for d in d_list]
    pvo2s_c = [results_per_d[d]["flat"][0] for d in d_list]
    k_co2s_c = [results_per_d[d]["flat"][2] for d in d_list]
    gamma_range = max(gammas_c) - min(gammas_c)
    pvo2_range = max(pvo2s_c) - min(pvo2s_c)
    k_co2_range = max(k_co2s_c) - min(k_co2s_c)
    print(f"\n  6. [Exp C] Parameter stability across d={d_list}:")
    print(f"     gamma range: {gamma_range:.4f} ({'STABLE' if gamma_range < 0.1 else 'VARIES'})")
    print(f"     pvo2 range:  {pvo2_range:.2f} ({'STABLE' if pvo2_range < 3.0 else 'VARIES'})")
    print(f"     k_co2 range: {k_co2_range:.4f} ({'STABLE' if k_co2_range < 0.02 else 'VARIES'})")
    is_nuisance = gamma_range < 0.1 and pvo2_range < 3.0 and k_co2_range < 0.02
    print(f"     -> d is {'NUISANCE (can fix at any value)' if is_nuisance else 'NOT nuisance (params depend on d)'}")


if __name__ == "__main__":
    main()
