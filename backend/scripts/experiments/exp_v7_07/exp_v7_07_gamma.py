"""
v7 Experiment 07: Gamma Profile + FL#6 Washout Diagnostic.

v7.06 resolved the b_s problem (b_s=1.0 naturally in all configs) and established Config B
(b_s=1 fixed, CV=0.15) as the winner. Stage A is identifiable, transferable (LOHO-Inf R2n=0.96),
and structurally clean. The only remaining structural issue is gamma=2.0 at its upper bound
in Stage B (all configs).

v7.07 changes from v7.06:
  1. Use Config B only (b_s=1 fixed, CV=0.15). Stage A sensor frozen from v7.06.
  2. Gamma profile in Stage B: fix gamma on grid [0.8..4.0], refit all other Stage B params.
  3. Widen gamma bounds from [0.8, 2.0] to [0.8, 4.0] and refit freely.
  4. FL#6 washout diagnostic: Stage B with and without FL#6, and with widened tau_washout
     bounds (250 -> 500) for FL-type holds.
  5. Seed robustness check for Stage A: seeds {42, 123, 456}.
  6. Gamma prior relaxed from N(1, 0.15) [lambda=22.2] to N(1.5, 0.5) [lambda=2.0].

Carries forward from v7.06:
  Config B sensor model, power-law descent, baseline-corrected measurement,
  nadir-window fitting, Student-t NLL + Huber timing penalty, DE popsize=40/maxiter=4000.

Usage:
    cd backend && uv run python -u scripts/experiments/exp_v7_07/exp_v7_07_gamma.py
"""

import csv
import io
import os
import multiprocessing
import sqlite3
import sys
from pathlib import Path

# Use fork context to avoid spawn-based deadlocks on macOS with Python 3.13
_mp_ctx = multiprocessing.get_context("fork")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution
from scipy.special import gammainc

N_WORKERS = min(max(1, os.cpu_count() - 1), 8)


def make_de_callback(label, maxiter):
    """Create a progress callback for differential_evolution."""
    state = {"gen": 0}

    def callback(xk, convergence=0):
        state["gen"] += 1
        gen = state["gen"]
        pct = gen / maxiter * 100
        bar_len = 30
        filled = int(bar_len * gen / maxiter)
        bar = "=" * filled + "-" * (bar_len - filled)
        sys.stdout.write(f"\r  [{bar}] {pct:5.1f}% gen {gen}/{maxiter} (conv={convergence:.2e})  ")
        sys.stdout.flush()

    return callback


# -- Paths and constants -----------------------------------------------------

DB_PATH = Path(__file__).resolve().parents[4] / "data" / "spo2.db"

P50_BASE = 26.6
P_EQ = 100.0
PACO2_NORMAL = 40.0
TAU_CLEAR_FIXED = 30.0
FIO2_PB_PH2O = 149.2  # FiO2 * (PB - PH2O) = 0.2093 * (760 - 47)
RQ = 0.8

# FL#1 excluded (only 2% SpO2 variation), RV#4 excluded (nadir during apnea)
EXCLUDED_IDS = {1, 4}

# -- Config B constants (b_s=1 fixed, CV=0.15) ------------------------------

CV_FIXED = 0.15
BS_FIXED = 1.0

# -- Stage A: Sensor regularization -----------------------------------------

LAMBDA_TAU0 = 3.125        # LogNormal(log 18, 0.4): 1/(2*0.4^2)
LAMBDA_DELTA = 5.0         # StudentT-like shrinkage, sigma ~3s
LAMBDA_ZEROSUM = 500.0     # Zero-sum on deltas
LAMBDA_P = 4.08            # LogNormal(log 3.5, 0.35): 1/(2*0.35^2)
LAMBDA_NADIR = 500.0       # Huber timing penalty (delta=8s)
EFF_LAG_MIN = 5.0          # Minimum effective lag floor (seconds)
P_PRIOR_CENTER = 3.5

# -- Stage B: Physiology regularization -------------------------------------

PVO2_FIXED = 25.0
LAMBDA_K_CO2 = 1250.0      # N(0.06, 0.02): 1/(2*0.02^2)
LAMBDA_PACO2 = 0.056       # N(40, 3): 1/(2*3^2)
# v7.07: gamma prior relaxed from N(1, 0.15) to N(1.5, 0.5)
LAMBDA_GAMMA = 2.0          # N(1.5, 0.5): 1/(2*0.5^2)
GAMMA_PRIOR_CENTER = 1.5
LAMBDA_REG = 10.0          # Per-hold IC -> type-mean

# -- Shared constants -------------------------------------------------------

TAU0_PRIOR_CENTER = 18.0
NADIR_WINDOW_AFTER = 45

# -- Bounds ------------------------------------------------------------------

# Stage A
TAU0_BOUNDS = (5, 45)
P_BOUNDS = (1.0, 5.0)
DELTA_BOUNDS = (-20, 20)

# Power-law latent
S_MIN_BOUNDS = (30, 100)
V_UP_BOUNDS = (0.0, 3.0)

# Stage B - v7.07: gamma widened from [0.8, 2.0] to [0.8, 4.0]
PERHOLD_BOUNDS = {
    "FL": [(50, 250), (20, 50)],
    "FRC": [(20, 100), (25, 50)],
    "RV": [(10, 80), (30, 55)],
}
PERHOLD_BOUNDS_WIDE_WASHOUT = {
    "FL": [(50, 500), (20, 50)],
    "FRC": [(20, 100), (25, 50)],
    "RV": [(10, 80), (30, 55)],
}
PERHOLD_NAMES = ["tau_washout", "paco2_0"]
N_PH = len(PERHOLD_NAMES)
GAMMA_BOUNDS = (0.8, 4.0)

# Student-t NLL parameters
NU_STUDENT = 4.0
SIGMA_STUDENT = 1.0


# -- Data loading ------------------------------------------------------------


def _parse_time_to_seconds(time_str):
    parts = time_str.strip().split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    raise ValueError(f"Cannot parse time: {time_str!r}")


def load_holds_with_recovery(recovery_max_s=90, recovery_spo2_ceiling=97):
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


# -- Physiology functions ----------------------------------------------------


def corrected_pao2_0(paco2_0, aa):
    return max(FIO2_PB_PH2O - paco2_0 / RQ - aa, 1.0)


def pao2_apnea_only(t, pao2_0, pvo2, tau_washout, t_end):
    return pvo2 + (pao2_0 - pvo2) * np.exp(-t / max(tau_washout, 0.01))


def p50_apnea_only(t, paco2_0, k_co2, t_end):
    paco2 = paco2_0 + k_co2 * t
    return P50_BASE + 0.48 * (paco2 - PACO2_NORMAL)


def odc_severinghaus(pao2, p50_eff, gamma):
    pao2_virtual = pao2 * (P50_BASE / np.maximum(p50_eff, 0.01))
    pao2_adj = P50_BASE * (np.maximum(pao2_virtual, 0.01) / P50_BASE) ** gamma
    x = np.maximum(pao2_adj, 0.01)
    return 100.0 / (1.0 + 23400.0 / (x**3 + 150.0 * x))


# -- Smooth Discrete Gamma Kernel -------------------------------------------


def gamma_kernel_smooth(mean_lag, cv, max_support=120):
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


def apply_gamma_kernel(signal, mean_lag, cv):
    h = gamma_kernel_smooth(mean_lag, cv)
    pad_len = len(h)
    padded = np.concatenate([np.full(pad_len, signal[0]), signal])
    convolved = np.convolve(padded, h, mode="full")[:len(padded)]
    return convolved[pad_len:]


# -- Power-law latent template ----------------------------------------------


def build_powerlaw_latent(t_1hz, t_end, S_start, S_min, v_up, p):
    t_turn = max(t_end, 1.0)
    latent = np.empty_like(t_1hz)
    for i, t in enumerate(t_1hz):
        if t <= t_turn:
            frac = t / t_turn
            latent[i] = S_min + (S_start - S_min) * (1.0 - frac ** p)
        else:
            latent[i] = S_min + v_up * (t - t_turn)
    return np.clip(latent, 0.0, 100.0)


# -- Student-t NLL loss -----------------------------------------------------


def student_t_nll(residuals, nu=NU_STUDENT, sigma=SIGMA_STUDENT):
    return (nu + 1.0) / 2.0 * np.sum(np.log1p(residuals**2 / (nu * sigma**2)))


# -- Nadir + loss helpers ----------------------------------------------------


def compute_nadir_info(hold):
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


def nadir_window_mask(t, t_end, window_after=NADIR_WINDOW_AFTER):
    return t <= t_end + window_after


def huber_loss(a, delta=8.0):
    abs_a = np.abs(a)
    return np.where(abs_a <= delta, 0.5 * a**2, delta * (abs_a - 0.5 * delta))


def nadir_timing_penalty_huber(t, pred, t_nadir_obs, lam=LAMBDA_NADIR, huber_delta=8.0):
    t_nadir_pred = t[np.argmin(pred)]
    err = t_nadir_pred - t_nadir_obs
    return lam * float(huber_loss(err, delta=huber_delta))


# -- Metrics -----------------------------------------------------------------


def compute_r2(obs, pred):
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def compute_rmse(obs, pred):
    return float(np.sqrt(np.mean((obs - pred) ** 2)))


def is_at_bound(val, lo, hi, tol=1e-3):
    return abs(val - lo) < tol or abs(val - hi) < tol


# -- Stage A: Sensor-First Calibration (Config B: b_s=1 fixed) ---------------


def run_stage_a(fit_holds, nadir_info, s_base_values, seed=42):
    """Stage A: Sensor calibration with b_s=1 fixed, CV=0.15."""
    n_holds = len(fit_holds)

    # Params: tau_0, p, deltas..., (S_min, v_up) per hold
    bounds = [TAU0_BOUNDS, P_BOUNDS]
    n_global = 2

    delta_offset = n_global
    for _ in fit_holds:
        bounds.append(DELTA_BOUNDS)

    latent_offset = delta_offset + n_holds
    for _ in fit_holds:
        bounds.append(S_MIN_BOUNDS)
        bounds.append(V_UP_BOUNDS)

    print(f"\n  Stage A: {len(bounds)} params ({n_global} global + {n_holds} delta + "
          f"2x{n_holds} latent), seed={seed}")
    print(f"  b_s=1.0 (fixed), cv={CV_FIXED}")

    masks = [nadir_window_mask(h["t"], h["t_end"]) for h in fit_holds]
    nadir_ts = [nadir_info[h["id"]]["t_nadir"] for h in fit_holds]
    t_grids = [np.arange(0, h["t"][-1] + 1, 1.0) for h in fit_holds]

    def objective(flat):
        tau_0, p = flat[:n_global]
        deltas = flat[delta_offset:delta_offset + n_holds]
        total = 0.0

        for i, h in enumerate(fit_holds):
            lp_start = latent_offset + i * 2
            S_min, v_up = flat[lp_start:lp_start + 2]
            S_start = s_base_values[h["id"]]
            t_1hz = t_grids[i]
            latent = build_powerlaw_latent(t_1hz, h["t_end"], S_start, S_min, v_up, p)
            eff_lag = max(tau_0 + deltas[i], EFF_LAG_MIN)
            filtered = apply_gamma_kernel(latent, eff_lag, CV_FIXED)
            # b_s=1: pred = B_h + 1*(filtered - B_h) = filtered
            pred_at_obs = np.interp(h["t"], t_1hz, filtered)
            pred_display = np.clip(pred_at_obs, 0.0, 100.0)
            m = masks[i]
            residuals = h["spo2"][m] - pred_at_obs[m]
            total += student_t_nll(residuals)
            total += nadir_timing_penalty_huber(h["t"][m], pred_display[m], nadir_ts[i])

        # Priors
        total += LAMBDA_TAU0 * (np.log(max(tau_0, 1.0)) - np.log(TAU0_PRIOR_CENTER)) ** 2
        total += LAMBDA_DELTA * np.sum(np.log1p(deltas**2 / 9.0))
        total += LAMBDA_ZEROSUM * np.sum(deltas) ** 2
        total += LAMBDA_P * (np.log(max(p, 0.1)) - np.log(P_PRIOR_CENTER)) ** 2

        return total

    maxiter_a = 4000
    res = differential_evolution(
        objective, bounds, maxiter=maxiter_a, seed=seed, tol=1e-10,
        polish=True, popsize=40, mutation=(0.5, 1.5), recombination=0.9,
        callback=make_de_callback("Stage A", maxiter_a),
    )
    print(f"\n  Converged: {res.success}, fun={res.fun:.2f}, nfev={res.nfev}")

    # Build full vector with b_s=1.0 inserted at index 1 for compatibility
    full = np.empty(len(res.x) + 1)
    full[0] = res.x[0]       # tau_0
    full[1] = BS_FIXED        # b_s (fixed)
    full[2:] = res.x[1:]     # p, deltas, latent params

    return full, res.success, res.fun


def evaluate_stage_a(flat_a, fit_holds, nadir_info, s_base_values):
    """Evaluate Stage A results. flat_a has [tau_0, b_s, p, deltas..., latent...]."""
    n_holds = len(fit_holds)
    n_global = 3
    delta_offset = n_global
    latent_offset = delta_offset + n_holds

    tau_0, b_s, p = flat_a[:n_global]
    deltas = flat_a[delta_offset:delta_offset + n_holds]

    results = []
    latent_curves = []

    for i, h in enumerate(fit_holds):
        lp_start = latent_offset + i * 2
        S_min, v_up = flat_a[lp_start:lp_start + 2]
        S_start = s_base_values[h["id"]]

        t_1hz = np.arange(0, h["t"][-1] + 1, 1.0)
        latent = build_powerlaw_latent(t_1hz, h["t_end"], S_start, S_min, v_up, p)
        eff_lag = max(tau_0 + deltas[i], EFF_LAG_MIN)
        filtered = apply_gamma_kernel(latent, eff_lag, CV_FIXED)
        # b_s=1: pred = filtered
        pred_at_obs = np.interp(h["t"], t_1hz, filtered)
        pred_at_obs = np.clip(pred_at_obs, 0.0, 100.0)

        r2_full = compute_r2(h["spo2"], pred_at_obs)
        apnea_mask = h["t"] <= h["t_end"]
        r2_apnea = (compute_r2(h["spo2"][apnea_mask], pred_at_obs[apnea_mask])
                     if apnea_mask.sum() > 3 else None)
        mask = nadir_window_mask(h["t"], h["t_end"])
        r2_nadir = compute_r2(h["spo2"][mask], pred_at_obs[mask]) if mask.sum() > 3 else None

        t_nadir_obs = nadir_info[h["id"]]["t_nadir"]
        t_nadir_pred = h["t"][np.argmin(pred_at_obs)]
        nadir_err = t_nadir_pred - t_nadir_obs

        results.append({
            "hold_id": h["id"],
            "hold_type": h["type"],
            "r2_full": r2_full,
            "r2_apnea": r2_apnea,
            "r2_nadir": r2_nadir,
            "pred_full": pred_at_obs,
            "nadir_err": nadir_err,
            "effective_lag": eff_lag,
            "delta": deltas[i],
        })
        latent_curves.append({
            "hold_id": h["id"],
            "t_1hz": t_1hz,
            "latent": latent,
            "filtered": filtered,
            "S_start": S_start,
            "S_min": S_min,
            "v_up": v_up,
        })

    return results, latent_curves


def extract_frozen_sensor(flat_a, fit_holds):
    n_holds = len(fit_holds)
    n_global = 3
    delta_offset = n_global
    tau_0, b_s, p = flat_a[:n_global]
    deltas = flat_a[delta_offset:delta_offset + n_holds]
    return {
        "tau_0": tau_0,
        "b_s": b_s,
        "p": p,
        "deltas": deltas,
        "cv": CV_FIXED,
    }


# -- Stage B: Physiology (apnea-only, frozen sensor) ------------------------


def predict_v7(t, pvo2, tau_washout, gamma, paco2_0, k_co2, b_s,
               mean_lag, cv, t_end, s_base, shift=0.0):
    aa = 0.0
    pao2_0 = corrected_pao2_0(paco2_0, aa)
    pao2 = pao2_apnea_only(t, pao2_0, pvo2, tau_washout, t_end)
    p50 = p50_apnea_only(t, paco2_0, k_co2, t_end)
    sa = odc_severinghaus(pao2, p50, gamma)
    eff_mean_lag = max(mean_lag + shift, EFF_LAG_MIN)
    filtered = apply_gamma_kernel(sa, eff_mean_lag, cv)
    return np.clip(s_base + b_s * (filtered - s_base), 0.0, 100.0)


def run_stage_b(fit_holds, nadir_info, frozen_sensor, s_base_values,
                gamma_bounds=None, gamma_fixed=None, perhold_bounds=None,
                label="Stage B"):
    """Stage B with truly frozen sensor.

    If gamma_fixed is not None, gamma is fixed to that value.
    If gamma_bounds is provided, overrides the default GAMMA_BOUNDS.
    If perhold_bounds is provided, overrides PERHOLD_BOUNDS.
    """
    tau_0_frozen = frozen_sensor["tau_0"]
    b_s_frozen = frozen_sensor["b_s"]
    cv_frozen = frozen_sensor["cv"]
    deltas_frozen = frozen_sensor["deltas"]

    if gamma_bounds is None:
        gamma_bounds = GAMMA_BOUNDS
    if perhold_bounds is None:
        perhold_bounds = PERHOLD_BOUNDS

    n_holds = len(fit_holds)

    bounds = [(0.02, 0.25)]  # k_co2
    if gamma_fixed is not None:
        bounds.append((gamma_fixed - 0.001, gamma_fixed + 0.001))
    else:
        bounds.append(gamma_bounds)
    n_phys = 2

    ic_offset = n_phys
    for h in fit_holds:
        bounds.extend(perhold_bounds[h["type"]])

    type_groups = {}
    for i, h in enumerate(fit_holds):
        type_groups.setdefault(h["type"], []).append(i)

    apnea_window = 5
    masks = [h["t"] <= h["t_end"] + apnea_window for h in fit_holds]
    weights = [np.where(h["spo2"][m] < 95, 3.0, 1.0) for h, m in zip(fit_holds, masks)]

    gamma_str = f"gamma={'fixed=' + f'{gamma_fixed:.2f}' if gamma_fixed is not None else f'free [{gamma_bounds[0]:.1f}, {gamma_bounds[1]:.1f}]'}"
    print(f"\n  {label}: {len(bounds)} params, {gamma_str}")

    def objective(flat):
        k_co2, gamma_val = flat[:n_phys]
        total = 0.0

        for i, h in enumerate(fit_holds):
            ph_offset = ic_offset + i * N_PH
            tau_washout, paco2_0 = flat[ph_offset:ph_offset + N_PH]

            pred = predict_v7(
                h["t"], PVO2_FIXED, tau_washout, gamma_val,
                paco2_0, k_co2, b_s_frozen,
                tau_0_frozen, cv_frozen, h["t_end"],
                s_base=s_base_values[h["id"]],
                shift=deltas_frozen[i],
            )
            m = masks[i]
            total += np.sum(weights[i] * (h["spo2"][m] - pred[m]) ** 2)
            total += LAMBDA_PACO2 * (paco2_0 - 40.0) ** 2

        total += LAMBDA_K_CO2 * (k_co2 - 0.06) ** 2
        total += LAMBDA_GAMMA * (gamma_val - GAMMA_PRIOR_CENTER) ** 2

        for ht, indices in type_groups.items():
            if len(indices) < 2:
                continue
            for p_off in range(N_PH):
                values = [flat[ic_offset + idx * N_PH + p_off] for idx in indices]
                mean_val = np.mean(values)
                total += LAMBDA_REG * sum((v - mean_val) ** 2 for v in values)

        return total

    maxiter_b = 2000
    res = differential_evolution(
        objective, bounds, maxiter=maxiter_b, seed=42, tol=1e-10,
        polish=True, popsize=25, mutation=(0.5, 1.5), recombination=0.9,
        callback=make_de_callback(label, maxiter_b),
    )
    print(f"\n  Converged: {res.success}, fun={res.fun:.2f}, nfev={res.nfev}")

    k_co2, gamma_val = res.x[:n_phys]
    ics = res.x[n_phys:]
    # Insert frozen deltas for evaluate_stage_b compatibility
    full_b = np.concatenate([
        [k_co2, gamma_val],
        deltas_frozen,
        ics,
    ])
    return full_b, res.success, res.fun


def run_stage_b_weak_lag(fit_holds, nadir_info, frozen_sensor, s_base_values,
                          gamma_bounds=None, perhold_bounds=None,
                          label="Stage B (weak-lag)"):
    """Stage B with weak lag prior: delta free with N(delta_stageA, sigma=2s)."""
    tau_0_frozen = frozen_sensor["tau_0"]
    b_s_frozen = frozen_sensor["b_s"]
    cv_frozen = frozen_sensor["cv"]
    delta_a_values = frozen_sensor["deltas"]

    if gamma_bounds is None:
        gamma_bounds = GAMMA_BOUNDS
    if perhold_bounds is None:
        perhold_bounds = PERHOLD_BOUNDS

    n_holds = len(fit_holds)
    bounds = [
        (0.02, 0.25),  # k_co2
        gamma_bounds,   # gamma
    ]
    n_phys = 2

    delta_offset = n_phys
    for _ in fit_holds:
        bounds.append(DELTA_BOUNDS)

    ic_offset = delta_offset + n_holds
    for h in fit_holds:
        bounds.extend(perhold_bounds[h["type"]])

    type_groups = {}
    for i, h in enumerate(fit_holds):
        type_groups.setdefault(h["type"], []).append(i)

    apnea_window = 5
    masks = [h["t"] <= h["t_end"] + apnea_window for h in fit_holds]
    weights = [np.where(h["spo2"][m] < 95, 3.0, 1.0) for h, m in zip(fit_holds, masks)]

    lambda_wl = 1.0 / (2.0 * 2.0**2)  # sigma=2s

    print(f"\n  {label}: deltas free with N(delta_stageA, sigma=2s)")

    def objective(flat):
        k_co2, gamma_val = flat[:n_phys]
        deltas = flat[delta_offset:delta_offset + n_holds]
        total = 0.0

        for i, h in enumerate(fit_holds):
            ph_offset = ic_offset + i * N_PH
            tau_washout, paco2_0 = flat[ph_offset:ph_offset + N_PH]

            pred = predict_v7(
                h["t"], PVO2_FIXED, tau_washout, gamma_val,
                paco2_0, k_co2, b_s_frozen,
                tau_0_frozen, cv_frozen, h["t_end"],
                s_base=s_base_values[h["id"]],
                shift=deltas[i],
            )
            m = masks[i]
            total += np.sum(weights[i] * (h["spo2"][m] - pred[m]) ** 2)
            total += LAMBDA_PACO2 * (paco2_0 - 40.0) ** 2

        total += LAMBDA_K_CO2 * (k_co2 - 0.06) ** 2
        total += LAMBDA_GAMMA * (gamma_val - GAMMA_PRIOR_CENTER) ** 2

        for i in range(n_holds):
            total += lambda_wl * (deltas[i] - delta_a_values[i]) ** 2
        total += LAMBDA_ZEROSUM * np.sum(deltas) ** 2

        for ht, indices in type_groups.items():
            if len(indices) < 2:
                continue
            for p_off in range(N_PH):
                values = [flat[ic_offset + idx * N_PH + p_off] for idx in indices]
                mean_val = np.mean(values)
                total += LAMBDA_REG * sum((v - mean_val) ** 2 for v in values)

        return total

    maxiter_wl = 2000
    res = differential_evolution(
        objective, bounds, maxiter=maxiter_wl, seed=42, tol=1e-10,
        polish=True, popsize=25, mutation=(0.5, 1.5), recombination=0.9,
        callback=make_de_callback(label, maxiter_wl),
    )
    print(f"\n  Converged: {res.success}, fun={res.fun:.2f}, nfev={res.nfev}")
    return res.x, res.success


def evaluate_stage_b(flat_b, fit_holds, nadir_info,
                     frozen_sensor, s_base_values, all_holds=None):
    """Evaluate Stage B. flat_b layout: [k_co2, gamma, deltas..., ICs...]."""
    results = []
    target_holds = all_holds if all_holds is not None else fit_holds
    fit_ids = {h["id"] for h in fit_holds}
    n_holds = len(fit_holds)

    n_phys = 2
    delta_offset = n_phys
    ic_offset = delta_offset + n_holds

    k_co2, gamma_val = flat_b[:n_phys]
    deltas = flat_b[delta_offset:delta_offset + n_holds]

    tau_0 = frozen_sensor["tau_0"]
    cv = frozen_sensor["cv"]
    b_s = frozen_sensor["b_s"]

    for h in target_holds:
        if h["id"] not in fit_ids:
            type_indices = [i for i, fh in enumerate(fit_holds) if fh["type"] == h["type"]]
            if not type_indices:
                continue
            avg_ph = np.mean(
                [flat_b[ic_offset + idx * N_PH:ic_offset + (idx + 1) * N_PH]
                 for idx in type_indices],
                axis=0,
            )
            tau_washout, paco2_0 = avg_ph
            hold_idx = None
            is_excl = True
        else:
            hold_idx = next(i for i, fh in enumerate(fit_holds) if fh["id"] == h["id"])
            ph_start = ic_offset + hold_idx * N_PH
            tau_washout, paco2_0 = flat_b[ph_start:ph_start + N_PH]
            is_excl = False

        shift = 0.0
        delta_val = 0.0
        if hold_idx is not None:
            delta_val = deltas[hold_idx]
            shift = delta_val

        s_base = s_base_values[h["id"]]
        pred_full = predict_v7(
            h["t"], PVO2_FIXED, tau_washout, gamma_val,
            paco2_0, k_co2, b_s,
            tau_0, cv, h["t_end"], s_base=s_base,
            shift=shift,
        )
        pred_apnea = predict_v7(
            h["t_apnea"], PVO2_FIXED, tau_washout, gamma_val,
            paco2_0, k_co2, b_s,
            tau_0, cv, h["t_end"], s_base=s_base,
            shift=shift,
        )

        r2_full = compute_r2(h["spo2"], pred_full)
        r2_apnea = compute_r2(h["spo2_apnea"], pred_apnea)

        mask = nadir_window_mask(h["t"], h["t_end"])
        r2_nadir = compute_r2(h["spo2"][mask], pred_full[mask]) if mask.sum() > 3 else None

        r2_recovery = None
        if len(h["t_recovery"]) > 3:
            pred_rec = predict_v7(
                h["t_recovery"], PVO2_FIXED, tau_washout, gamma_val,
                paco2_0, k_co2, b_s,
                tau_0, cv, h["t_end"], s_base=s_base,
                shift=shift,
            )
            r2_recovery = compute_r2(h["spo2_recovery"], pred_rec)

        t_nadir_obs = nadir_info[h["id"]]["t_nadir_apnea"]
        nadir_spo2_obs = nadir_info[h["id"]]["spo2_nadir_apnea"]
        apnea_nadir_mask = h["t"] <= h["t_end"] + 5
        t_nadir_pred = h["t"][apnea_nadir_mask][np.argmin(pred_full[apnea_nadir_mask])]
        nadir_err = t_nadir_pred - t_nadir_obs
        nadir_spo2_pred = float(np.min(pred_full[apnea_nadir_mask]))
        nadir_spo2_err = nadir_spo2_pred - nadir_spo2_obs

        results.append({
            "hold_id": h["id"],
            "hold_type": h["type"],
            "r2_full": r2_full,
            "r2_apnea": r2_apnea,
            "r2_nadir": r2_nadir,
            "r2_recovery": r2_recovery,
            "pred_full": pred_full,
            "nadir_err": nadir_err,
            "t_nadir_pred": t_nadir_pred,
            "nadir_spo2_pred": nadir_spo2_pred,
            "nadir_spo2_err": nadir_spo2_err,
            "is_excluded": is_excl or h["id"] in EXCLUDED_IDS,
            "effective_lag": max(tau_0 + shift, EFF_LAG_MIN),
            "delta": delta_val,
            "tau_washout": tau_washout,
            "paco2_0": paco2_0,
        })
    return results


# -- Gamma profile worker (for parallel execution) --------------------------


def _gamma_profile_worker(args):
    """Worker for one gamma profile point."""
    gamma_val, fit_holds, nadir_info, frozen_sensor, s_base_values, perhold_bounds_dict = args

    tau_0_frozen = frozen_sensor["tau_0"]
    b_s_frozen = frozen_sensor["b_s"]
    cv_frozen = frozen_sensor["cv"]
    deltas_frozen = frozen_sensor["deltas"]
    n_holds = len(fit_holds)

    # k_co2 + gamma (fixed) + per-hold ICs
    bounds = [
        (0.02, 0.25),
        (gamma_val - 0.001, gamma_val + 0.001),
    ]
    n_phys = 2
    ic_offset = n_phys
    for h in fit_holds:
        bounds.extend(perhold_bounds_dict[h["type"]])

    type_groups = {}
    for i, h in enumerate(fit_holds):
        type_groups.setdefault(h["type"], []).append(i)

    apnea_window = 5
    masks = [h["t"] <= h["t_end"] + apnea_window for h in fit_holds]
    weights = [np.where(h["spo2"][m] < 95, 3.0, 1.0) for h, m in zip(fit_holds, masks)]

    def objective(flat):
        k_co2, gv = flat[:n_phys]
        total = 0.0
        for i, h in enumerate(fit_holds):
            ph_offset = ic_offset + i * N_PH
            tau_washout, paco2_0 = flat[ph_offset:ph_offset + N_PH]
            pred = predict_v7(
                h["t"], PVO2_FIXED, tau_washout, gv,
                paco2_0, k_co2, b_s_frozen,
                tau_0_frozen, cv_frozen, h["t_end"],
                s_base=s_base_values[h["id"]],
                shift=deltas_frozen[i],
            )
            m = masks[i]
            total += np.sum(weights[i] * (h["spo2"][m] - pred[m]) ** 2)
            total += LAMBDA_PACO2 * (paco2_0 - 40.0) ** 2
        total += LAMBDA_K_CO2 * (k_co2 - 0.06) ** 2
        # No gamma prior in profile (we want raw loss landscape)
        for ht, indices in type_groups.items():
            if len(indices) < 2:
                continue
            for p_off in range(N_PH):
                values = [flat[ic_offset + idx * N_PH + p_off] for idx in indices]
                mean_val = np.mean(values)
                total += LAMBDA_REG * sum((v - mean_val) ** 2 for v in values)
        return total

    res = differential_evolution(
        objective, bounds, maxiter=2000, seed=42, tol=1e-10,
        polish=True, popsize=25, mutation=(0.5, 1.5), recombination=0.9,
    )

    k_co2 = res.x[0]
    ics = res.x[n_phys:]
    # Extract per-hold ICs for reporting
    per_hold_ics = {}
    for i, h in enumerate(fit_holds):
        ph_offset = i * N_PH
        per_hold_ics[h["id"]] = {
            "tau_washout": ics[ph_offset],
            "paco2_0": ics[ph_offset + 1],
        }

    return gamma_val, {
        "loss": res.fun,
        "k_co2": k_co2,
        "per_hold_ics": per_hold_ics,
        "success": res.success,
    }


# -- Sponge diagnostics -----------------------------------------------------


def sponge_diagnostics(flat, bounds, prior_sigmas, param_names, label=""):
    print(f"\n  Sponge Diagnostics {label}:")
    print(f"  {'Param':<15s} | {'Value':>10s} | {'[Lo, Hi]':>16s} | {'AtBound':>7s} | "
          f"{'Contraction':>11s}")
    print(f"  {'-'*15}-+-{'-'*10}-+-{'-'*16}-+-{'-'*7}-+-{'-'*11}")

    at_bound_count = 0
    for i, (name, val, (lo, hi)) in enumerate(zip(param_names, flat[:len(param_names)], bounds)):
        at_b = is_at_bound(val, lo, hi)
        if at_b:
            at_bound_count += 1
        sigma = prior_sigmas.get(name, None)
        if sigma:
            prior_range = hi - lo
            posterior_width = 2 * sigma
            contraction = min(posterior_width / prior_range, 1.0) if prior_range > 0 else 1.0
            contr_str = f"{contraction:.3f}"
        else:
            contr_str = "N/A"
        print(f"  {name:<15s} | {val:10.4f} | [{lo:>6.2f}, {hi:>6.2f}] | "
              f"{'YES' if at_b else '   ':>7s} | {contr_str:>11s}")

    print(f"\n  Total at bound: {at_bound_count}/{len(param_names)}")
    return at_bound_count


# -- Plots -------------------------------------------------------------------


def plot_gamma_profile(gamma_results, output_path):
    """Gamma profile: loss and k_co2 vs gamma."""
    gamma_vals = sorted(gamma_results.keys())
    losses = [gamma_results[g]["loss"] for g in gamma_vals]
    k_co2s = [gamma_results[g]["k_co2"] for g in gamma_vals]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = "#1f77b4"
    ax1.plot(gamma_vals, losses, "o-", color=color1, linewidth=2, markersize=8, label="Loss")
    ax1.set_xlabel("gamma (fixed)", fontsize=12)
    ax1.set_ylabel("Loss (no gamma prior)", fontsize=12, color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    # Mark minimum
    min_idx = np.argmin(losses)
    min_gamma = gamma_vals[min_idx]
    min_loss = losses[min_idx]
    ax1.plot(min_gamma, min_loss, "r*", markersize=15, zorder=5,
             label=f"Min at gamma={min_gamma:.2f}")

    ax2 = ax1.twinx()
    color2 = "#ff7f0e"
    ax2.plot(gamma_vals, k_co2s, "s--", color=color2, linewidth=2, markersize=6, label="k_co2")
    ax2.set_ylabel("k_co2", fontsize=12, color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    # v7.06 gamma bound
    ax1.axvline(x=2.0, color="red", linestyle=":", alpha=0.5, label="v7.06 upper bound")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

    ax1.set_title("v7.07 Gamma Profile Likelihood (Stage B)", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nGamma profile plot saved to {output_path}")


def plot_stage_b_fits(eval_b, all_holds, nadir_info, label, output_path):
    """Per-hold Stage B fit plots."""
    holds_dict = {h["id"]: h for h in all_holds}
    fitted = [r for r in eval_b if not r["is_excluded"]]
    n_holds = len(fitted)

    fig, axes = plt.subplots(n_holds, 1, figsize=(10, 4.5 * n_holds), squeeze=False)

    for row, res in enumerate(fitted):
        ax = axes[row, 0]
        h = holds_dict[res["hold_id"]]
        ni = nadir_info[res["hold_id"]]

        ax.plot(h["t"], h["spo2"], "k.", markersize=2, alpha=0.5, label="Observed")
        ax.axvline(x=h["t_end"], color="red", linestyle="--", alpha=0.5)
        ax.plot(ni["t_nadir"], ni["spo2_nadir"], "r*", markersize=10, zorder=5)

        r2_str = f"R2a={res['r2_apnea']:.3f}" if res["r2_apnea"] is not None else ""
        ax.plot(h["t"], res["pred_full"], color="#2ca02c", linewidth=2, alpha=0.8,
                label=f"Stage B ({r2_str})")

        ax.set_ylabel(f"{h['type']}#{h['id']}\nSpO2 (%)")
        ax.set_ylim(30, 105)
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(True, alpha=0.3)
        if row == n_holds - 1:
            ax.set_xlabel("Time (s)")

    fig.suptitle(f"v7.07 {label}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Stage B fits plot saved to {output_path}")


def plot_fl6_sensitivity(results_with, results_without, output_path):
    """FL#6 sensitivity: compare gamma and k_co2."""
    labels = ["All 4 holds", "Without FL#6", "Wide washout"]
    gammas = [r["gamma"] for r in [results_with, results_without] + ([results_with] if len([results_with, results_without]) < 3 else [])]
    k_co2s = [r["k_co2"] for r in [results_with, results_without] + ([results_with] if len([results_with, results_without]) < 3 else [])]

    # Simplified: just show the comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    configs = list(results_with.keys()) if isinstance(results_with, dict) else ["All 4 holds", "Without FL#6"]
    # This will be called with specific data, handle generically
    fig.suptitle("v7.07 FL#6 Sensitivity", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"FL#6 sensitivity plot saved to {output_path}")


def plot_fl6_comparison(fl6_configs, output_path):
    """Bar chart comparing gamma and k_co2 across FL#6 sensitivity configs."""
    labels = list(fl6_configs.keys())
    gammas = [fl6_configs[l]["gamma"] for l in labels]
    k_co2s = [fl6_configs[l]["k_co2"] for l in labels]
    losses = [fl6_configs[l]["loss"] for l in labels]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Gamma
    colors_g = ["#2ca02c" if not is_at_bound(g, *GAMMA_BOUNDS) else "#d62728" for g in gammas]
    axes[0].bar(labels, gammas, color=colors_g, alpha=0.8)
    axes[0].axhline(y=GAMMA_BOUNDS[0], color="gray", linestyle="--", alpha=0.3)
    axes[0].axhline(y=GAMMA_BOUNDS[1], color="gray", linestyle="--", alpha=0.3)
    axes[0].axhline(y=2.0, color="red", linestyle=":", alpha=0.5, label="v7.06 bound")
    axes[0].set_ylabel("gamma")
    axes[0].set_title("gamma", fontweight="bold")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3, axis="y")

    # k_co2
    axes[1].bar(labels, k_co2s, color="#1f77b4", alpha=0.8)
    axes[1].set_ylabel("k_co2")
    axes[1].set_title("k_co2", fontweight="bold")
    axes[1].grid(True, alpha=0.3, axis="y")

    # Loss
    axes[2].bar(labels, losses, color="#ff7f0e", alpha=0.8)
    axes[2].set_ylabel("Loss")
    axes[2].set_title("Stage B Loss", fontweight="bold")
    axes[2].grid(True, alpha=0.3, axis="y")

    fig.suptitle("v7.07 FL#6 Washout Sensitivity", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"FL#6 comparison plot saved to {output_path}")


# -- Main --------------------------------------------------------------------


def main():
    print("=" * 120)
    print("v7.07: Gamma Profile + FL#6 Washout Diagnostic")
    print("=" * 120)

    # == Load data ===========================================================
    print("\nLoading holds with recovery data...")
    all_holds = load_holds_with_recovery()

    for h in all_holds:
        n_a = len(h["t_apnea"])
        n_r = len(h["t_recovery"])
        tag = " [EXCLUDED]" if h["id"] in EXCLUDED_IDS else ""
        rec = ""
        if n_r > 0:
            rec = f", recovery SpO2 {h['spo2_recovery'].min():.0f}-{h['spo2_recovery'][-1]:.0f}%"
        print(f"  {h['type']}#{h['id']}{tag}: {n_a} apnea + {n_r} recovery pts "
              f"(t_end={h['t_end']:.0f}s{rec})")

    fit_holds = [h for h in all_holds if h["id"] not in EXCLUDED_IDS]
    print(f"\nFitting on {len(fit_holds)} holds: "
          f"{[f'{h['type']}#{h['id']}' for h in fit_holds]}")

    # == Nadir info ==========================================================
    nadir_info = {}
    for h in all_holds:
        ni = compute_nadir_info(h)
        nadir_info[h["id"]] = ni
        tag = " [EXCLUDED]" if h["id"] in EXCLUDED_IDS else ""
        loc = "recovery" if ni["in_recovery"] else "apnea"
        print(f"  {h['type']}#{h['id']}{tag}: nadir at t={ni['t_nadir']:.0f}s "
              f"({loc}, delay={ni['delay_from_end']:+.0f}s)")

    # == Baselines ===========================================================
    s_base_values = {}
    for h in all_holds:
        plateau_mask = h["t"] <= 20
        if plateau_mask.sum() > 0:
            s_base_values[h["id"]] = float(np.median(h["spo2"][plateau_mask]))
        else:
            s_base_values[h["id"]] = float(h["spo2"][0])

    # == QC flags ============================================================
    qc_flags = {}
    for h in all_holds:
        ni = nadir_info[h["id"]]
        qc_flags[h["id"]] = ni["t_nadir_apnea"] < h["t_end"] - 2

    # == Changes from v7.06 ==================================================
    print(f"\nKey changes from v7.06:")
    print(f"  1. Config B only (b_s=1 fixed, CV=0.15)")
    print(f"  2. Gamma profile in Stage B: grid [0.8, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 3.5, 4.0]")
    print(f"  3. Gamma bounds widened from [0.8, 2.0] to [{GAMMA_BOUNDS[0]}, {GAMMA_BOUNDS[1]}]")
    print(f"  4. FL#6 washout diagnostic: with/without FL#6, widened tau_washout")
    print(f"  5. Seed robustness check for Stage A: seeds {{42, 123, 456}}")
    print(f"  6. Gamma prior: N({GAMMA_PRIOR_CENTER}, 0.5) [lambda={LAMBDA_GAMMA}] "
          f"(was N(1, 0.15) [lambda=22.2])")

    output_dir = Path(__file__).resolve().parent

    # ========================================================================
    # STEP 1: Stage A (Config B, seed=42)
    # ========================================================================
    print(f"\n{'='*120}")
    print("STEP 1: STAGE A (Config B: b_s=1 fixed, CV=0.15, seed=42)")
    print(f"{'='*120}")

    flat_a, conv_a, loss_a = run_stage_a(fit_holds, nadir_info, s_base_values, seed=42)
    n_holds = len(fit_holds)

    tau_0, b_s, p = flat_a[:3]
    print(f"\n  Stage A globals: tau_0={tau_0:.4f}, b_s={b_s:.4f} [FIXED], p={p:.4f}")
    print(f"    Bounds: tau_0 [{TAU0_BOUNDS[0]}, {TAU0_BOUNDS[1]}]"
          f"{' ** AT BOUND **' if is_at_bound(tau_0, *TAU0_BOUNDS) else ''}")
    print(f"    Bounds: p [{P_BOUNDS[0]}, {P_BOUNDS[1]}]"
          f"{' ** AT BOUND **' if is_at_bound(p, *P_BOUNDS) else ''}")

    k_val = 1.0 / (CV_FIXED * CV_FIXED)
    print(f"    Kernel: k={k_val:.2f}, mean={tau_0:.1f}s, std={tau_0*CV_FIXED:.1f}s")

    delta_offset = 3
    latent_offset = delta_offset + n_holds
    deltas_a = flat_a[delta_offset:delta_offset + n_holds]

    print(f"\n  Per-hold breakdown:")
    for i, h in enumerate(fit_holds):
        d = deltas_a[i]
        eff = max(tau_0 + d, EFF_LAG_MIN)
        lp_start = latent_offset + i * 2
        S_min, v_up = flat_a[lp_start:lp_start + 2]
        bound_str = " *BOUND*" if is_at_bound(d, *DELTA_BOUNDS) else ""
        eff_floor_str = " *FLOOR*" if eff <= EFF_LAG_MIN + 0.01 else ""
        qc_str = " [QC]" if qc_flags[h["id"]] else ""
        print(f"    {h['type']}#{h['id']}: delta={d:+6.2f}, eff_lag={eff:6.2f}, "
              f"B_h={s_base_values[h['id']]:.1f}, S_min={S_min:.1f}, "
              f"v_up={v_up:.2f}{bound_str}{eff_floor_str}{qc_str}")

    eval_a, latent_curves = evaluate_stage_a(flat_a, fit_holds, nadir_info, s_base_values)
    print(f"\n  Stage A fit metrics:")
    for r in eval_a:
        r2n = f"{r['r2_nadir']:.4f}" if r['r2_nadir'] is not None else "N/A"
        r2a = f"{r['r2_apnea']:.4f}" if r['r2_apnea'] is not None else "N/A"
        print(f"    {r['hold_type']}#{r['hold_id']}: R2a={r2a}, R2n={r2n}, "
              f"nadir_err={r['nadir_err']:+.1f}s")

    frozen_sensor = extract_frozen_sensor(flat_a, fit_holds)

    # ========================================================================
    # STEP 2: Seed robustness check
    # ========================================================================
    print(f"\n{'='*120}")
    print("STEP 2: SEED ROBUSTNESS CHECK (seeds: 42, 123, 456)")
    print(f"{'='*120}")

    seed_results = {42: {"tau_0": tau_0, "p": p, "loss": loss_a, "conv": conv_a}}

    for seed in [123, 456]:
        print(f"\n  --- Seed {seed} ---")
        flat_s, conv_s, loss_s = run_stage_a(fit_holds, nadir_info, s_base_values, seed=seed)
        tau_0_s, _, p_s = flat_s[:3]
        seed_results[seed] = {"tau_0": tau_0_s, "p": p_s, "loss": loss_s, "conv": conv_s}
        print(f"  tau_0={tau_0_s:.4f}, p={p_s:.4f}, loss={loss_s:.2f}")

    print(f"\n  Seed Robustness Summary:")
    print(f"  {'Seed':>6s} | {'tau_0':>8s} | {'p':>8s} | {'Loss':>10s} | {'Conv':>5s}")
    print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*5}")
    for seed in [42, 123, 456]:
        sr = seed_results[seed]
        print(f"  {seed:>6d} | {sr['tau_0']:>8.4f} | {sr['p']:>8.4f} | "
              f"{sr['loss']:>10.2f} | {'Y' if sr['conv'] else 'N':>5s}")

    tau_range = max(sr["tau_0"] for sr in seed_results.values()) - min(sr["tau_0"] for sr in seed_results.values())
    p_range = max(sr["p"] for sr in seed_results.values()) - min(sr["p"] for sr in seed_results.values())
    loss_range = max(sr["loss"] for sr in seed_results.values()) - min(sr["loss"] for sr in seed_results.values())
    min_loss = min(sr["loss"] for sr in seed_results.values())
    loss_pct = loss_range / min_loss * 100 if min_loss > 0 else 0

    print(f"\n  Ranges: tau_0={tau_range:.4f}s, p={p_range:.4f}, "
          f"loss={loss_range:.2f} ({loss_pct:.1f}%)")
    seed_robust = tau_range < 1.0 and p_range < 0.1 and loss_pct < 5.0
    print(f"  Seed robust (tau_0<1s, p<0.1, loss<5%): {'PASS' if seed_robust else 'FAIL'}")

    # ========================================================================
    # STEP 3: Gamma profile in Stage B
    # ========================================================================
    print(f"\n{'='*120}")
    print("STEP 3: GAMMA PROFILE (Stage B, 9 grid points)")
    print(f"{'='*120}")

    gamma_grid = [0.8, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 3.5, 4.0]
    print(f"\n  Grid: {gamma_grid}")
    print(f"  Running {len(gamma_grid)} gamma profile points in parallel ({N_WORKERS} workers)...")

    args_list = [
        (gv, fit_holds, nadir_info, frozen_sensor, s_base_values, PERHOLD_BOUNDS)
        for gv in gamma_grid
    ]
    with _mp_ctx.Pool(processes=N_WORKERS) as pool:
        raw_results = pool.map(_gamma_profile_worker, args_list)

    gamma_profile = {}
    for gv, result in raw_results:
        gamma_profile[gv] = result

    print(f"\n  Gamma Profile Results (no gamma prior):")
    print(f"  {'gamma':>8s} | {'Loss':>12s} | {'k_co2':>8s} | {'Conv':>5s}")
    print(f"  {'-'*8}-+-{'-'*12}-+-{'-'*8}-+-{'-'*5}")
    for gv in gamma_grid:
        r = gamma_profile[gv]
        print(f"  {gv:>8.2f} | {r['loss']:>12.2f} | {r['k_co2']:>8.4f} | "
              f"{'Y' if r['success'] else 'N':>5s}")

    # Per-hold tau_washout at each gamma
    print(f"\n  Per-hold tau_washout at each gamma:")
    print(f"  {'gamma':>8s}", end="")
    for h in fit_holds:
        print(f" | {h['type']}#{h['id']:>3d}", end="")
    print()
    for gv in gamma_grid:
        r = gamma_profile[gv]
        print(f"  {gv:>8.2f}", end="")
        for h in fit_holds:
            tw = r["per_hold_ics"][h["id"]]["tau_washout"]
            print(f" | {tw:>8.1f}", end="")
        print()

    # Analyze profile shape
    losses = [gamma_profile[g]["loss"] for g in gamma_grid]
    min_idx = np.argmin(losses)
    min_gamma = gamma_grid[min_idx]
    min_loss = losses[min_idx]
    is_monotone = (all(losses[i] >= losses[i+1] for i in range(len(losses)-1)) or
                   all(losses[i] <= losses[i+1] for i in range(len(losses)-1)))
    is_interior = min_idx > 0 and min_idx < len(gamma_grid) - 1

    print(f"\n  Profile analysis:")
    print(f"    Minimum at gamma={min_gamma:.2f} (loss={min_loss:.2f})")
    print(f"    Interior minimum: {'YES' if is_interior else 'NO'}")
    print(f"    Monotone: {'YES (decreasing)' if is_monotone else 'NO (non-monotone)'}")

    if not is_interior:
        if min_idx == len(gamma_grid) - 1:
            print(f"    --> Loss still decreasing at gamma={gamma_grid[-1]}. "
                  f"ODC parameterization may need rethinking.")
        elif min_idx == 0:
            print(f"    --> Minimum at lower bound gamma={gamma_grid[0]}.")

    # ========================================================================
    # STEP 4: Stage B free fit with widened gamma bounds [0.8, 4.0]
    # ========================================================================
    print(f"\n{'='*120}")
    print("STEP 4: STAGE B FREE FIT (gamma bounds [0.8, 4.0])")
    print(f"{'='*120}")

    flat_b_free, conv_b_free, loss_b_free = run_stage_b(
        fit_holds, nadir_info, frozen_sensor, s_base_values,
        gamma_bounds=GAMMA_BOUNDS,
        label="Stage B (free, gamma [0.8, 4.0])")

    k_co2_free, gamma_free = flat_b_free[:2]
    gamma_at_bound = is_at_bound(gamma_free, *GAMMA_BOUNDS)
    print(f"\n  Result: k_co2={k_co2_free:.4f}, gamma={gamma_free:.4f}"
          f"{' ** AT BOUND **' if gamma_at_bound else ''}")
    print(f"  Loss={loss_b_free:.2f}")

    # Per-hold ICs
    ic_offset_b = 2 + n_holds
    print(f"\n  Per-hold ICs:")
    for i, h in enumerate(fit_holds):
        offset = ic_offset_b + i * N_PH
        tau_washout, paco2_0 = flat_b_free[offset:offset + N_PH]
        pao2_0 = corrected_pao2_0(paco2_0, 0.0)
        tw_bounds = PERHOLD_BOUNDS[h["type"]][0]
        tw_at_bound = is_at_bound(tau_washout, *tw_bounds)
        print(f"    {h['type']}#{h['id']}: tau_w={tau_washout:.1f}"
              f"{' *BOUND*' if tw_at_bound else ''}, "
              f"paco2_0={paco2_0:.1f}, PaO2_0={pao2_0:.1f}")

    eval_b_free = evaluate_stage_b(flat_b_free, fit_holds, nadir_info,
                                    frozen_sensor, s_base_values, all_holds)
    print(f"\n  Stage B fit metrics:")
    for r in eval_b_free:
        if r["is_excluded"]:
            continue
        r2n = f"{r['r2_nadir']:.4f}" if r['r2_nadir'] is not None else "N/A"
        r2a = f"{r['r2_apnea']:.4f}" if r['r2_apnea'] is not None else "N/A"
        print(f"    {r['hold_type']}#{r['hold_id']}: R2a={r2a}, R2n={r2n}, "
              f"nadir_err={r['nadir_err']:+.1f}s")

    # Sponge diagnostics for free fit
    b_bounds = [(0.02, 0.25), GAMMA_BOUNDS]
    b_prior_sigmas = {"k_co2": 0.02, "gamma": 0.5}
    sponge_diagnostics(flat_b_free, b_bounds, b_prior_sigmas,
                        ["k_co2", "gamma"], label="Stage B (free)")

    # ========================================================================
    # STEP 5: FL#6 washout diagnostic
    # ========================================================================
    print(f"\n{'='*120}")
    print("STEP 5: FL#6 WASHOUT DIAGNOSTIC")
    print(f"{'='*120}")

    fl6_configs = {}

    # 5a: All 4 holds, standard bounds (already done in step 4)
    fl6_configs["All 4 holds"] = {
        "gamma": gamma_free,
        "k_co2": k_co2_free,
        "loss": loss_b_free,
        "flat_b": flat_b_free,
    }

    # 5b: Exclude FL#6 (3 holds only)
    fit_holds_no_fl6 = [h for h in fit_holds if h["id"] != 6]
    frozen_sensor_no_fl6 = {
        "tau_0": frozen_sensor["tau_0"],
        "b_s": frozen_sensor["b_s"],
        "p": frozen_sensor["p"],
        "deltas": np.array([frozen_sensor["deltas"][i]
                            for i, h in enumerate(fit_holds) if h["id"] != 6]),
        "cv": frozen_sensor["cv"],
    }

    print(f"\n  5b: Excluding FL#6 (fitting on {[f'{h['type']}#{h['id']}' for h in fit_holds_no_fl6]})")
    flat_b_no_fl6, conv_no_fl6, loss_no_fl6 = run_stage_b(
        fit_holds_no_fl6, nadir_info, frozen_sensor_no_fl6, s_base_values,
        gamma_bounds=GAMMA_BOUNDS,
        label="Stage B (no FL#6)")

    k_co2_no_fl6, gamma_no_fl6 = flat_b_no_fl6[:2]
    print(f"\n  Result: k_co2={k_co2_no_fl6:.4f}, gamma={gamma_no_fl6:.4f}"
          f"{' ** AT BOUND **' if is_at_bound(gamma_no_fl6, *GAMMA_BOUNDS) else ''}")

    fl6_configs["Without FL#6"] = {
        "gamma": gamma_no_fl6,
        "k_co2": k_co2_no_fl6,
        "loss": loss_no_fl6,
        "flat_b": flat_b_no_fl6,
    }

    # 5c: All 4 holds, widened FL tau_washout to 500
    print(f"\n  5c: All 4 holds, FL tau_washout widened to [50, 500]")
    flat_b_wide, conv_wide, loss_wide = run_stage_b(
        fit_holds, nadir_info, frozen_sensor, s_base_values,
        gamma_bounds=GAMMA_BOUNDS,
        perhold_bounds=PERHOLD_BOUNDS_WIDE_WASHOUT,
        label="Stage B (wide washout)")

    k_co2_wide, gamma_wide = flat_b_wide[:2]
    print(f"\n  Result: k_co2={k_co2_wide:.4f}, gamma={gamma_wide:.4f}"
          f"{' ** AT BOUND **' if is_at_bound(gamma_wide, *GAMMA_BOUNDS) else ''}")

    # Check FL#6 tau_washout
    fl6_idx = next(i for i, h in enumerate(fit_holds) if h["id"] == 6)
    ic_offset_wide = 2 + n_holds
    tw_fl6_wide = flat_b_wide[ic_offset_wide + fl6_idx * N_PH]
    tw_bounds_wide = PERHOLD_BOUNDS_WIDE_WASHOUT["FL"][0]
    print(f"  FL#6 tau_washout = {tw_fl6_wide:.1f} "
          f"(bounds [{tw_bounds_wide[0]}, {tw_bounds_wide[1]}])"
          f"{' *BOUND*' if is_at_bound(tw_fl6_wide, *tw_bounds_wide) else ''}")

    fl6_configs["Wide washout"] = {
        "gamma": gamma_wide,
        "k_co2": k_co2_wide,
        "loss": loss_wide,
        "flat_b": flat_b_wide,
    }

    # FL#6 sensitivity analysis
    gamma_change_no_fl6 = abs(gamma_no_fl6 - gamma_free) / max(abs(gamma_free), 1e-6) * 100
    gamma_change_wide = abs(gamma_wide - gamma_free) / max(abs(gamma_free), 1e-6) * 100
    k_co2_change_no_fl6 = abs(k_co2_no_fl6 - k_co2_free) / max(abs(k_co2_free), 1e-6) * 100

    print(f"\n  FL#6 Sensitivity Summary:")
    print(f"  {'Config':<20s} | {'gamma':>8s} | {'k_co2':>8s} | {'Loss':>10s}")
    print(f"  {'-'*20}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}")
    for name, cfg in fl6_configs.items():
        g_flag = " *B*" if is_at_bound(cfg["gamma"], *GAMMA_BOUNDS) else ""
        print(f"  {name:<20s} | {cfg['gamma']:>7.4f}{g_flag} | "
              f"{cfg['k_co2']:>8.4f} | {cfg['loss']:>10.2f}")

    print(f"\n  gamma change (no FL#6): {gamma_change_no_fl6:.1f}%"
          f" {'> 10% - FL#6 DISTORTS' if gamma_change_no_fl6 > 10 else '< 10% - FL#6 OK'}")
    print(f"  k_co2 change (no FL#6): {k_co2_change_no_fl6:.1f}%")
    print(f"  gamma change (wide washout): {gamma_change_wide:.1f}%")

    # ========================================================================
    # STEP 6: Weak-lag diagnostic on best Stage B
    # ========================================================================
    print(f"\n{'='*120}")
    print("STEP 6: WEAK-LAG DIAGNOSTIC (best Stage B config)")
    print(f"{'='*120}")

    flat_b_wl, conv_wl = run_stage_b_weak_lag(
        fit_holds, nadir_info, frozen_sensor, s_base_values,
        gamma_bounds=GAMMA_BOUNDS)

    k_co2_wl, gamma_wl = flat_b_wl[:2]
    print(f"\n  Frozen vs Weak-lag:")
    print(f"    k_co2: frozen={k_co2_free:.4f}, weak={k_co2_wl:.4f}, "
          f"diff={abs(k_co2_wl - k_co2_free) / max(abs(k_co2_free), 1e-6) * 100:.1f}%")
    print(f"    gamma: frozen={gamma_free:.4f}, weak={gamma_wl:.4f}, "
          f"diff={abs(gamma_wl - gamma_free) / max(abs(gamma_free), 1e-6) * 100:.1f}%")

    # ========================================================================
    # STEP 7: Stage A sponge diagnostics
    # ========================================================================
    print(f"\n{'='*120}")
    print("STEP 7: SPONGE DIAGNOSTICS")
    print(f"{'='*120}")

    a_bounds_list = [TAU0_BOUNDS, (0.5, 2.0), P_BOUNDS]
    a_prior_sigmas = {"tau_0": 0.4, "b_s": 0.1, "p": 0.35}
    sponge_diagnostics(flat_a, a_bounds_list, a_prior_sigmas,
                        ["tau_0", "b_s", "p"], label="Stage A")

    b_bounds_list = [(0.02, 0.25), GAMMA_BOUNDS]
    b_prior_sigmas = {"k_co2": 0.02, "gamma": 0.5}
    sponge_diagnostics(flat_b_free, b_bounds_list, b_prior_sigmas,
                        ["k_co2", "gamma"], label="Stage B (free)")

    # ========================================================================
    # COMPARISON TABLE
    # ========================================================================
    print(f"\n{'='*120}")
    print("SUMMARY COMPARISON TABLE")
    print(f"{'='*120}")

    print(f"\n  {'Metric':<35s} | {'v7.06 (B)':>12s} | {'v7.07 (free)':>12s}")
    print(f"  {'-'*35}-+-{'-'*12}-+-{'-'*12}")
    print(f"  {'gamma bounds':<35s} | {'[0.8, 2.0]':>12s} | {f'[{GAMMA_BOUNDS[0]}, {GAMMA_BOUNDS[1]}]':>12s}")
    print(f"  {'gamma prior':<35s} | {'N(1,0.15)':>12s} | {f'N({GAMMA_PRIOR_CENTER},{0.5})':>12s}")
    print(f"  {'gamma lambda':<35s} | {'22.2':>12s} | {f'{LAMBDA_GAMMA}':>12s}")
    print(f"  {'gamma value':<35s} | {'2.00 *BOUND*':>12s} | {f'{gamma_free:.4f}':>12s}")
    print(f"  {'k_co2':<35s} | {'(from v7.06)':>12s} | {k_co2_free:>12.4f}")
    print(f"  {'gamma at bound':<35s} | {'YES':>12s} | {'YES' if gamma_at_bound else 'NO':>12s}")

    # ========================================================================
    # DECISION CRITERIA
    # ========================================================================
    print(f"\n{'='*120}")
    print("DECISION CRITERIA")
    print(f"{'='*120}")

    print(f"\n  1. Gamma profile interior minimum at gamma_opt < 4.0:")
    if is_interior:
        print(f"     YES - gamma_opt = {min_gamma:.2f}")
        print(f"     Action: Set bounds to [{GAMMA_BOUNDS[0]}, {max(min_gamma*1.5, 4.0):.1f}], "
              f"use {min_gamma:.2f} as prior center")
    else:
        if min_idx == len(gamma_grid) - 1:
            print(f"     NO - loss still decreasing at gamma={gamma_grid[-1]}")
            print(f"     Action: ODC parameterization may need rethinking (Hill-form in v7.08)")
        else:
            print(f"     EDGE - minimum at gamma={min_gamma:.2f}")

    print(f"\n  2. Gamma profile monotonically decreasing through 4.0:")
    if is_monotone and min_idx == len(gamma_grid) - 1:
        print(f"     YES - monotonically decreasing")
        print(f"     Action: Investigate Hill-form ODC in v7.08")
    else:
        print(f"     NO - profile is non-monotone")

    print(f"\n  3. FL#6 exclusion changes gamma by >10%:")
    if gamma_change_no_fl6 > 10:
        print(f"     YES - change = {gamma_change_no_fl6:.1f}%")
        print(f"     Action: FL#6 washout distorting; widen bounds or exclude from Stage B")
    else:
        print(f"     NO - change = {gamma_change_no_fl6:.1f}%")

    print(f"\n  4. Seed check shows divergent basins:")
    if seed_robust:
        print(f"     NO - all seeds in same basin (PASS)")
    else:
        print(f"     YES - seeds diverge")
        print(f"     Action: Add multi-start protocol to Stage A")

    print(f"\n  5. Free-fit gamma interior to bounds:")
    if not gamma_at_bound:
        print(f"     YES - gamma={gamma_free:.4f} is interior to [{GAMMA_BOUNDS[0]}, {GAMMA_BOUNDS[1]}]")
        print(f"     Gamma problem RESOLVED.")
    else:
        print(f"     NO - gamma={gamma_free:.4f} at bound")
        print(f"     Gamma still pegging. Further investigation needed.")

    # ========================================================================
    # PLOTS
    # ========================================================================
    print(f"\n{'='*120}")
    print("GENERATING PLOTS")
    print(f"{'='*120}")

    plot_gamma_profile(gamma_profile, output_dir / "exp_v7_07_gamma_profile.png")

    plot_stage_b_fits(eval_b_free, all_holds, nadir_info,
                       f"Stage B (gamma={gamma_free:.2f}, k_co2={k_co2_free:.4f})",
                       output_dir / "exp_v7_07_stage_b.png")

    plot_fl6_comparison(fl6_configs, output_dir / "exp_v7_07_fl6_sensitivity.png")

    # ========================================================================
    # DONE
    # ========================================================================
    print(f"\n{'='*120}")
    print("DONE")
    print(f"{'='*120}")

    print(f"\nOutput files:")
    print(f"  {output_dir / 'exp_v7_07_gamma_profile.png'}")
    print(f"  {output_dir / 'exp_v7_07_stage_b.png'}")
    print(f"  {output_dir / 'exp_v7_07_fl6_sensitivity.png'}")


if __name__ == "__main__":
    main()
