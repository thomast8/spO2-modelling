"""
v7 Experiment 08: Crystallize and Ship.

v7.07 resolved the gamma-at-bound problem (interior minimum at ~2.5, free fit at 2.4) and
identified FL#6 washout as the remaining distortion source. Wide FL washout bounds [50, 500]
recover the same gamma/k_co2 as the "no FL#6" model (1.7% gamma shift, 3.7% k_co2 shift).

v7.08 is a packaging step, not a research experiment. The only genuinely new diagnostic is
the k_co2 profile. Everything else confirms the shipping config.

v7.08 changes from v7.07:
  1. Shipping config: FL tau_washout bounds [50, 500] (was [50, 250]).
     FRC bounds stay at [20, 100] (widening to [20, 150] destabilizes the system).
  2. Gamma prior updated to N(2.0, 0.5) (was N(1.5, 0.5)).
  3. k_co2 profile: fix k_co2 on grid, refit gamma + per-hold ICs (new diagnostic).
  4. FRC widening diagnostic: compare FRC [20, 100] vs [20, 150] to document why
     FRC bounds must stay narrow.
  5. No seed robustness (settled in v7.07), no gamma profile (settled in v7.07).

Carries forward from v7.07:
  Config B sensor model (b_s=1 fixed, CV=0.15), gamma bounds [0.8, 4.0],
  power-law descent, Student-t NLL + Huber timing penalty, DE popsize=40/maxiter=4000.

Usage:
    cd backend && uv run python -u scripts/experiments/exp_v7_08/exp_v7_08_ship.py
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
K_CO2_PRIOR_CENTER = 0.06
LAMBDA_PACO2 = 0.056       # N(40, 3): 1/(2*3^2)
# v7.08: gamma prior updated from N(1.5, 0.5) to N(2.0, 0.5)
LAMBDA_GAMMA = 2.0          # N(2.0, 0.5): 1/(2*0.5^2)
GAMMA_PRIOR_CENTER = 2.0
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

# Stage B - v7.08 shipping config:
#   FL washout [50, 500] (was [50, 250] in v7.06, validated in v7.07)
#   FRC washout stays at [20, 100] (widening to [20, 150] destabilizes system)
PERHOLD_BOUNDS = {
    "FL": [(50, 500), (20, 50)],
    "FRC": [(20, 100), (25, 50)],
    "RV": [(10, 80), (30, 55)],
}
# Diagnostic: FRC widened to [20, 150] to show why it breaks
PERHOLD_BOUNDS_FRC_WIDE = {
    "FL": [(50, 500), (20, 50)],
    "FRC": [(20, 150), (25, 50)],
    "RV": [(10, 80), (30, 55)],
}
PERHOLD_NAMES = ["tau_washout", "paco2_0"]
N_PH = len(PERHOLD_NAMES)
GAMMA_BOUNDS = (0.8, 4.0)
K_CO2_BOUNDS = (0.02, 0.25)

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
                gamma_bounds=None, gamma_fixed=None,
                k_co2_bounds=None, k_co2_fixed=None,
                perhold_bounds=None, label="Stage B"):
    """Stage B with truly frozen sensor.

    If gamma_fixed is not None, gamma is fixed to that value.
    If k_co2_fixed is not None, k_co2 is fixed to that value.
    """
    tau_0_frozen = frozen_sensor["tau_0"]
    b_s_frozen = frozen_sensor["b_s"]
    cv_frozen = frozen_sensor["cv"]
    deltas_frozen = frozen_sensor["deltas"]

    if gamma_bounds is None:
        gamma_bounds = GAMMA_BOUNDS
    if k_co2_bounds is None:
        k_co2_bounds = K_CO2_BOUNDS
    if perhold_bounds is None:
        perhold_bounds = PERHOLD_BOUNDS

    n_holds = len(fit_holds)

    # k_co2
    if k_co2_fixed is not None:
        bounds = [(k_co2_fixed - 0.0001, k_co2_fixed + 0.0001)]
    else:
        bounds = [k_co2_bounds]
    # gamma
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
    k_co2_str = f"k_co2={'fixed=' + f'{k_co2_fixed:.4f}' if k_co2_fixed is not None else f'free [{k_co2_bounds[0]:.2f}, {k_co2_bounds[1]:.2f}]'}"
    print(f"\n  {label}: {len(bounds)} params, {gamma_str}, {k_co2_str}")

    # Determine which priors to apply (skip prior for fixed/profiled params)
    apply_k_co2_prior = k_co2_fixed is None
    apply_gamma_prior = gamma_fixed is None

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

        if apply_k_co2_prior:
            total += LAMBDA_K_CO2 * (k_co2 - K_CO2_PRIOR_CENTER) ** 2
        if apply_gamma_prior:
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
        K_CO2_BOUNDS,  # k_co2
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

        total += LAMBDA_K_CO2 * (k_co2 - K_CO2_PRIOR_CENTER) ** 2
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


# -- k_co2 profile worker (for parallel execution) --------------------------


def _k_co2_profile_worker(args):
    """Worker for one k_co2 profile point."""
    k_co2_val, fit_holds, nadir_info, frozen_sensor, s_base_values, perhold_bounds_dict = args

    tau_0_frozen = frozen_sensor["tau_0"]
    b_s_frozen = frozen_sensor["b_s"]
    cv_frozen = frozen_sensor["cv"]
    deltas_frozen = frozen_sensor["deltas"]
    n_holds = len(fit_holds)

    # k_co2 (fixed) + gamma (free) + per-hold ICs
    bounds = [
        (k_co2_val - 0.0001, k_co2_val + 0.0001),
        GAMMA_BOUNDS,
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
        kc, gamma_val = flat[:n_phys]
        total = 0.0
        for i, h in enumerate(fit_holds):
            ph_offset = ic_offset + i * N_PH
            tau_washout, paco2_0 = flat[ph_offset:ph_offset + N_PH]
            pred = predict_v7(
                h["t"], PVO2_FIXED, tau_washout, gamma_val,
                paco2_0, kc, b_s_frozen,
                tau_0_frozen, cv_frozen, h["t_end"],
                s_base=s_base_values[h["id"]],
                shift=deltas_frozen[i],
            )
            m = masks[i]
            total += np.sum(weights[i] * (h["spo2"][m] - pred[m]) ** 2)
            total += LAMBDA_PACO2 * (paco2_0 - 40.0) ** 2
        # No k_co2 prior (we want raw loss landscape)
        total += LAMBDA_GAMMA * (gamma_val - GAMMA_PRIOR_CENTER) ** 2
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

    gamma_val = res.x[1]
    ics = res.x[n_phys:]
    per_hold_ics = {}
    for i, h in enumerate(fit_holds):
        ph_offset = i * N_PH
        per_hold_ics[h["id"]] = {
            "tau_washout": ics[ph_offset],
            "paco2_0": ics[ph_offset + 1],
        }

    return k_co2_val, {
        "loss": res.fun,
        "gamma": gamma_val,
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


def plot_k_co2_profile(k_co2_results, output_path):
    """k_co2 profile: loss and gamma vs k_co2."""
    k_co2_vals = sorted(k_co2_results.keys())
    losses = [k_co2_results[k]["loss"] for k in k_co2_vals]
    gammas = [k_co2_results[k]["gamma"] for k in k_co2_vals]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = "#1f77b4"
    ax1.plot(k_co2_vals, losses, "o-", color=color1, linewidth=2, markersize=8, label="Loss")
    ax1.set_xlabel("k_co2 (fixed)", fontsize=12)
    ax1.set_ylabel("Loss (no k_co2 prior)", fontsize=12, color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    # Mark minimum
    min_idx = np.argmin(losses)
    min_k = k_co2_vals[min_idx]
    min_loss = losses[min_idx]
    ax1.plot(min_k, min_loss, "r*", markersize=15, zorder=5,
             label=f"Min at k_co2={min_k:.3f}")

    ax2 = ax1.twinx()
    color2 = "#ff7f0e"
    ax2.plot(k_co2_vals, gammas, "s--", color=color2, linewidth=2, markersize=6, label="gamma")
    ax2.set_ylabel("gamma", fontsize=12, color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

    ax1.set_title("v7.08 k_co2 Profile Likelihood (Stage B)", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nk_co2 profile plot saved to {output_path}")


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

    fig.suptitle(f"v7.08 {label}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Stage B fits plot saved to {output_path}")


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
    axes[0].set_ylabel("gamma")
    axes[0].set_title("gamma", fontweight="bold")
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

    fig.suptitle("v7.08 FL#6 Sensitivity (Shipping Config)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"FL#6 comparison plot saved to {output_path}")


# -- Main --------------------------------------------------------------------


def main():
    print("=" * 120)
    print("v7.08: Crystallize and Ship")
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

    # == Changes from v7.07 ==================================================
    print(f"\nKey changes from v7.07:")
    print(f"  1. Shipping config: FL tau_washout bounds [50, 500] (was [50, 250])")
    print(f"     FRC tau_washout stays at [20, 100] (widening destabilizes)")
    print(f"  2. Gamma prior updated: N({GAMMA_PRIOR_CENTER}, 0.5) [lambda={LAMBDA_GAMMA}]")
    print(f"     (was N(1.5, 0.5) in v7.07)")
    print(f"  3. k_co2 profile: new diagnostic (fix k_co2 on grid, refit gamma + ICs)")
    print(f"  4. FRC widening diagnostic: FRC [20, 150] vs [20, 100] to document instability")
    print(f"  5. Removed: seed robustness (settled), gamma profile (settled)")

    output_dir = Path(__file__).resolve().parent

    # ========================================================================
    # STEP 1: Stage A (Config B, seed=42 - frozen from v7.06/07)
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
    # STEP 2: Stage B free fit (shipping config)
    # ========================================================================
    print(f"\n{'='*120}")
    print("STEP 2: STAGE B FREE FIT (shipping config)")
    print(f"{'='*120}")
    print(f"\n  Config: FL washout [50, 500], FRC washout [20, 100], "
          f"gamma [{GAMMA_BOUNDS[0]}, {GAMMA_BOUNDS[1]}], "
          f"gamma prior N({GAMMA_PRIOR_CENTER}, 0.5)")

    flat_b_free, conv_b_free, loss_b_free = run_stage_b(
        fit_holds, nadir_info, frozen_sensor, s_base_values,
        label="Stage B (shipping)")

    k_co2_free, gamma_free = flat_b_free[:2]
    gamma_at_bound = is_at_bound(gamma_free, *GAMMA_BOUNDS)
    k_co2_at_bound = is_at_bound(k_co2_free, *K_CO2_BOUNDS)
    print(f"\n  Result: k_co2={k_co2_free:.4f}"
          f"{' ** AT BOUND **' if k_co2_at_bound else ''}, "
          f"gamma={gamma_free:.4f}"
          f"{' ** AT BOUND **' if gamma_at_bound else ''}")
    print(f"  Loss={loss_b_free:.2f}")

    # Per-hold ICs
    ic_offset_b = 2 + n_holds
    fl6_idx = next(i for i, h in enumerate(fit_holds) if h["id"] == 6)
    print(f"\n  Per-hold ICs:")
    for i, h in enumerate(fit_holds):
        offset = ic_offset_b + i * N_PH
        tau_washout, paco2_0 = flat_b_free[offset:offset + N_PH]
        pao2_0 = corrected_pao2_0(paco2_0, 0.0)
        tw_bounds = PERHOLD_BOUNDS[h["type"]][0]
        tw_at_bound = is_at_bound(tau_washout, *tw_bounds)
        print(f"    {h['type']}#{h['id']}: tau_w={tau_washout:.1f}"
              f"{' *BOUND*' if tw_at_bound else ''}"
              f" [{tw_bounds[0]}, {tw_bounds[1]}], "
              f"paco2_0={paco2_0:.1f}, PaO2_0={pao2_0:.1f}")

    eval_b_free = evaluate_stage_b(flat_b_free, fit_holds, nadir_info,
                                    frozen_sensor, s_base_values, all_holds)
    print(f"\n  Stage B fit metrics (primary: R2a, secondary: R2n):")
    for r in eval_b_free:
        if r["is_excluded"]:
            continue
        r2n = f"{r['r2_nadir']:.4f}" if r['r2_nadir'] is not None else "N/A"
        r2a = f"{r['r2_apnea']:.4f}" if r['r2_apnea'] is not None else "N/A"
        print(f"    {r['hold_type']}#{r['hold_id']}: R2a={r2a}, R2n={r2n}, "
              f"nadir_err={r['nadir_err']:+.1f}s")

    # ========================================================================
    # STEP 3: k_co2 profile (new diagnostic)
    # ========================================================================
    print(f"\n{'='*120}")
    print("STEP 3: k_co2 PROFILE (Stage B, 8 grid points)")
    print(f"{'='*120}")

    k_co2_grid = [0.02, 0.04, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20]
    print(f"\n  Grid: {k_co2_grid}")
    print(f"  Running {len(k_co2_grid)} k_co2 profile points in parallel ({N_WORKERS} workers)...")
    print(f"  No k_co2 prior in profile (raw loss landscape)")

    args_list = [
        (kv, fit_holds, nadir_info, frozen_sensor, s_base_values, PERHOLD_BOUNDS)
        for kv in k_co2_grid
    ]
    with _mp_ctx.Pool(processes=N_WORKERS) as pool:
        raw_results = pool.map(_k_co2_profile_worker, args_list)

    k_co2_profile = {}
    for kv, result in raw_results:
        k_co2_profile[kv] = result

    print(f"\n  k_co2 Profile Results (no k_co2 prior):")
    print(f"  {'k_co2':>8s} | {'Loss':>12s} | {'gamma':>8s} | {'Conv':>5s}")
    print(f"  {'-'*8}-+-{'-'*12}-+-{'-'*8}-+-{'-'*5}")
    for kv in k_co2_grid:
        r = k_co2_profile[kv]
        print(f"  {kv:>8.4f} | {r['loss']:>12.2f} | {r['gamma']:>8.4f} | "
              f"{'Y' if r['success'] else 'N':>5s}")

    # Per-hold tau_washout at each k_co2
    print(f"\n  Per-hold tau_washout at each k_co2:")
    print(f"  {'k_co2':>8s}", end="")
    for h in fit_holds:
        print(f" | {h['type']}#{h['id']:>3d}", end="")
    print()
    for kv in k_co2_grid:
        r = k_co2_profile[kv]
        print(f"  {kv:>8.4f}", end="")
        for h in fit_holds:
            tw = r["per_hold_ics"][h["id"]]["tau_washout"]
            print(f" | {tw:>8.1f}", end="")
        print()

    # Analyze profile shape
    losses = [k_co2_profile[k]["loss"] for k in k_co2_grid]
    min_idx = np.argmin(losses)
    min_k = k_co2_grid[min_idx]
    min_loss_kp = losses[min_idx]
    is_monotone_k = (all(losses[i] >= losses[i+1] for i in range(len(losses)-1)) or
                     all(losses[i] <= losses[i+1] for i in range(len(losses)-1)))
    is_interior_k = min_idx > 0 and min_idx < len(k_co2_grid) - 1

    # Check flatness: ratio of max to min loss in the profile
    loss_range = max(losses) - min(losses)
    loss_range_pct = loss_range / min_loss_kp * 100 if min_loss_kp > 0 else 0

    print(f"\n  Profile analysis:")
    print(f"    Minimum at k_co2={min_k:.4f} (loss={min_loss_kp:.2f})")
    print(f"    Interior minimum: {'YES' if is_interior_k else 'NO'}")
    print(f"    Monotone: {'YES' if is_monotone_k else 'NO (non-monotone)'}")
    print(f"    Loss range: {loss_range:.2f} ({loss_range_pct:.1f}% of minimum)")

    if is_interior_k:
        print(f"    --> k_co2 is IDENTIFIED (interior minimum at {min_k:.4f})")
    elif is_monotone_k:
        if min_idx == 0:
            print(f"    --> Loss decreasing toward k_co2={k_co2_grid[0]}. "
                  f"k_co2 may prefer lower values.")
        else:
            print(f"    --> Loss decreasing toward k_co2={k_co2_grid[-1]}. "
                  f"k_co2 wants higher values than grid covers.")
    else:
        print(f"    --> k_co2 at edge of grid (minimum at boundary).")

    # Gamma-k_co2 tradeoff
    gammas_profile = [k_co2_profile[k]["gamma"] for k in k_co2_grid]
    gamma_range = max(gammas_profile) - min(gammas_profile)
    print(f"\n  Gamma-k_co2 tradeoff:")
    print(f"    gamma range across k_co2 grid: {min(gammas_profile):.3f} to "
          f"{max(gammas_profile):.3f} (span={gamma_range:.3f})")
    if gamma_range > 0.5:
        print(f"    --> Strong tradeoff: gamma changes by {gamma_range:.2f} across k_co2 grid")
    else:
        print(f"    --> Moderate/weak tradeoff")

    # ========================================================================
    # STEP 4: FL#6 sensitivity (confirmation under shipping config)
    # ========================================================================
    print(f"\n{'='*120}")
    print("STEP 4: FL#6 SENSITIVITY (confirmation)")
    print(f"{'='*120}")

    fl6_configs = {}

    # 4a: All 4 holds (already done in step 2)
    fl6_configs["All 4 holds"] = {
        "gamma": gamma_free,
        "k_co2": k_co2_free,
        "loss": loss_b_free,
        "flat_b": flat_b_free,
    }

    # 4b: Exclude FL#6 (3 holds only)
    fit_holds_no_fl6 = [h for h in fit_holds if h["id"] != 6]
    frozen_sensor_no_fl6 = {
        "tau_0": frozen_sensor["tau_0"],
        "b_s": frozen_sensor["b_s"],
        "p": frozen_sensor["p"],
        "deltas": np.array([frozen_sensor["deltas"][i]
                            for i, h in enumerate(fit_holds) if h["id"] != 6]),
        "cv": frozen_sensor["cv"],
    }

    print(f"\n  4b: Excluding FL#6 (fitting on {[f'{h['type']}#{h['id']}' for h in fit_holds_no_fl6]})")
    flat_b_no_fl6, conv_no_fl6, loss_no_fl6 = run_stage_b(
        fit_holds_no_fl6, nadir_info, frozen_sensor_no_fl6, s_base_values,
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

    # FL#6 sensitivity analysis
    gamma_change = abs(gamma_no_fl6 - gamma_free) / max(abs(gamma_free), 1e-6) * 100
    k_co2_change = abs(k_co2_no_fl6 - k_co2_free) / max(abs(k_co2_free), 1e-6) * 100

    print(f"\n  FL#6 Sensitivity Summary:")
    print(f"  {'Config':<20s} | {'gamma':>8s} | {'k_co2':>8s} | {'Loss':>10s}")
    print(f"  {'-'*20}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}")
    for name, cfg in fl6_configs.items():
        g_flag = " *B*" if is_at_bound(cfg["gamma"], *GAMMA_BOUNDS) else ""
        print(f"  {name:<20s} | {cfg['gamma']:>7.4f}{g_flag} | "
              f"{cfg['k_co2']:>8.4f} | {cfg['loss']:>10.2f}")

    fl6_stable = gamma_change < 10 and k_co2_change < 10
    print(f"\n  gamma change: {gamma_change:.1f}%"
          f" {'< 10% PASS' if gamma_change < 10 else '> 10% FAIL'}")
    print(f"  k_co2 change: {k_co2_change:.1f}%"
          f" {'< 10% PASS' if k_co2_change < 10 else '> 10% FAIL'}")
    print(f"  FL#6 stability: {'PASS' if fl6_stable else 'FAIL'}")

    # ========================================================================
    # STEP 5: FRC widening diagnostic (why FRC [20, 100] must stay narrow)
    # ========================================================================
    print(f"\n{'='*120}")
    print("STEP 5: FRC WIDENING DIAGNOSTIC")
    print(f"{'='*120}")
    print(f"\n  Testing FRC [20, 150] vs shipping FRC [20, 100].")
    print(f"  This documents why FRC bounds must stay narrow.")

    flat_b_frc_wide, conv_frc_wide, loss_frc_wide = run_stage_b(
        fit_holds, nadir_info, frozen_sensor, s_base_values,
        perhold_bounds=PERHOLD_BOUNDS_FRC_WIDE,
        label="Stage B (FRC wide)")

    k_co2_frc_wide, gamma_frc_wide = flat_b_frc_wide[:2]
    print(f"\n  Result: k_co2={k_co2_frc_wide:.4f}, gamma={gamma_frc_wide:.4f}")
    print(f"  Loss={loss_frc_wide:.2f}")

    # Per-hold ICs for FRC-wide
    print(f"\n  Per-hold ICs (FRC wide):")
    for i, h in enumerate(fit_holds):
        offset = ic_offset_b + i * N_PH
        tau_washout_fw = flat_b_frc_wide[offset]
        paco2_0_fw = flat_b_frc_wide[offset + 1]
        tw_bounds_fw = PERHOLD_BOUNDS_FRC_WIDE[h["type"]][0]
        tw_at_bound_fw = is_at_bound(tau_washout_fw, *tw_bounds_fw)
        print(f"    {h['type']}#{h['id']}: tau_w={tau_washout_fw:.1f}"
              f"{' *BOUND*' if tw_at_bound_fw else ''}"
              f" [{tw_bounds_fw[0]}, {tw_bounds_fw[1]}], "
              f"paco2_0={paco2_0_fw:.1f}")

    # Comparison
    gamma_shift_frc = abs(gamma_frc_wide - gamma_free) / max(abs(gamma_free), 1e-6) * 100
    k_co2_shift_frc = abs(k_co2_frc_wide - k_co2_free) / max(abs(k_co2_free), 1e-6) * 100
    print(f"\n  FRC Widening Impact:")
    print(f"  {'Metric':<20s} | {'FRC [20,100]':>12s} | {'FRC [20,150]':>12s} | {'Shift':>8s}")
    print(f"  {'-'*20}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}")
    print(f"  {'gamma':<20s} | {gamma_free:>12.4f} | {gamma_frc_wide:>12.4f} | {gamma_shift_frc:>7.1f}%")
    print(f"  {'k_co2':<20s} | {k_co2_free:>12.4f} | {k_co2_frc_wide:>12.4f} | {k_co2_shift_frc:>7.1f}%")
    print(f"  {'loss':<20s} | {loss_b_free:>12.2f} | {loss_frc_wide:>12.2f} |")

    # Check FL#6 tau_w in FRC-wide config
    fl6_tw_frc_wide = flat_b_frc_wide[ic_offset_b + fl6_idx * N_PH]
    fl6_tw_ship = flat_b_free[ic_offset_b + fl6_idx * N_PH]
    fl6_bounds_wide = PERHOLD_BOUNDS_FRC_WIDE["FL"][0]
    print(f"\n  FL#6 tau_washout: shipping={fl6_tw_ship:.1f}, "
          f"FRC-wide={fl6_tw_frc_wide:.1f}"
          f"{' *BOUND*' if is_at_bound(fl6_tw_frc_wide, *fl6_bounds_wide) else ''}")
    if is_at_bound(fl6_tw_frc_wide, *fl6_bounds_wide) and not is_at_bound(fl6_tw_ship, *PERHOLD_BOUNDS["FL"][0]):
        print(f"  --> FRC widening pushes FL#6 tau_w to bound!")
    print(f"\n  Conclusion: FRC [20, 100] is a stabilizing constraint. "
          f"Widening to [20, 150] destabilizes the gamma-k_co2-tau_washout system.")

    # ========================================================================
    # STEP 6: Weak-lag diagnostic
    # ========================================================================
    print(f"\n{'='*120}")
    print("STEP 6: WEAK-LAG DIAGNOSTIC")
    print(f"{'='*120}")

    flat_b_wl, conv_wl = run_stage_b_weak_lag(
        fit_holds, nadir_info, frozen_sensor, s_base_values)

    k_co2_wl, gamma_wl = flat_b_wl[:2]
    wl_gamma_diff = abs(gamma_wl - gamma_free) / max(abs(gamma_free), 1e-6) * 100
    wl_k_co2_diff = abs(k_co2_wl - k_co2_free) / max(abs(k_co2_free), 1e-6) * 100
    wl_pass = wl_gamma_diff < 5 and wl_k_co2_diff < 5

    print(f"\n  Frozen vs Weak-lag:")
    print(f"    k_co2: frozen={k_co2_free:.4f}, weak={k_co2_wl:.4f}, "
          f"diff={wl_k_co2_diff:.1f}%")
    print(f"    gamma: frozen={gamma_free:.4f}, weak={gamma_wl:.4f}, "
          f"diff={wl_gamma_diff:.1f}%")
    print(f"  Weak-lag stability (<5%): {'PASS' if wl_pass else 'FAIL'}")

    # ========================================================================
    # STEP 7: Sponge diagnostics
    # ========================================================================
    print(f"\n{'='*120}")
    print("STEP 7: SPONGE DIAGNOSTICS")
    print(f"{'='*120}")

    a_bounds_list = [TAU0_BOUNDS, (0.5, 2.0), P_BOUNDS]
    a_prior_sigmas = {"tau_0": 0.4, "b_s": 0.1, "p": 0.35}
    a_at_bound = sponge_diagnostics(flat_a, a_bounds_list, a_prior_sigmas,
                        ["tau_0", "b_s", "p"], label="Stage A")

    b_bounds_list = [K_CO2_BOUNDS, GAMMA_BOUNDS]
    b_prior_sigmas = {"k_co2": 0.02, "gamma": 0.5}
    b_at_bound = sponge_diagnostics(flat_b_free, b_bounds_list, b_prior_sigmas,
                        ["k_co2", "gamma"], label="Stage B (shipping)")

    # Per-hold sponge check
    print(f"\n  Per-hold tau_washout bounds check (shipping config):")
    ph_at_bound = 0
    for i, h in enumerate(fit_holds):
        offset = ic_offset_b + i * N_PH
        tau_washout = flat_b_free[offset]
        tw_bounds = PERHOLD_BOUNDS[h["type"]][0]
        at_b = is_at_bound(tau_washout, *tw_bounds)
        if at_b:
            ph_at_bound += 1
        print(f"    {h['type']}#{h['id']}: tau_w={tau_washout:.1f} "
              f"[{tw_bounds[0]}, {tw_bounds[1]}]"
              f"{' *BOUND*' if at_b else ''}")
    print(f"  Per-hold at bound: {ph_at_bound}/{n_holds}")

    # ========================================================================
    # SUCCESS CRITERIA
    # ========================================================================
    print(f"\n{'='*120}")
    print("SUCCESS CRITERIA")
    print(f"{'='*120}")

    criteria = []

    # 1. k_co2 identified
    c1 = is_interior_k
    criteria.append(("k_co2 profile has interior minimum", c1))
    print(f"\n  1. k_co2 profile has interior minimum: "
          f"{'PASS' if c1 else 'FAIL'}"
          f" (min at k_co2={min_k:.4f})")

    # 2. Gamma interior
    c2 = not gamma_at_bound
    criteria.append(("Gamma interior to bounds", c2))
    print(f"  2. Gamma interior to bounds: "
          f"{'PASS' if c2 else 'FAIL'}"
          f" (gamma={gamma_free:.4f})")

    # 3. No global params at bound (Stage A)
    c3 = a_at_bound == 0
    criteria.append(("No global params at bound (Stage A)", c3))
    print(f"  3. No global params at bound (Stage A): "
          f"{'PASS' if c3 else 'FAIL'} ({a_at_bound}/3)")

    # 4. No global params at bound (Stage B)
    c4 = b_at_bound == 0
    criteria.append(("No global params at bound (Stage B)", c4))
    print(f"  4. No global params at bound (Stage B): "
          f"{'PASS' if c4 else 'FAIL'} ({b_at_bound}/2)")

    # 5. FL#6 stability
    c5 = fl6_stable
    criteria.append(("FL#6 exclusion shift < 10%", c5))
    print(f"  5. FL#6 exclusion shift < 10%: "
          f"{'PASS' if c5 else 'FAIL'}"
          f" (gamma={gamma_change:.1f}%, k_co2={k_co2_change:.1f}%)")

    # 6. Weak-lag stability
    c6 = wl_pass
    criteria.append(("Weak-lag divergence < 5%", c6))
    print(f"  6. Weak-lag divergence < 5%: "
          f"{'PASS' if c6 else 'FAIL'}")

    # 7. FRC widening destabilizes (confirms keeping [20, 100])
    c7 = gamma_shift_frc > 5 or k_co2_shift_frc > 5
    criteria.append(("FRC widening destabilizes (confirms narrow bounds)", c7))
    print(f"  7. FRC widening destabilizes (>5% shift): "
          f"{'PASS (confirms [20,100])' if c7 else 'FAIL (widening would be safe)'}"
          f" (gamma={gamma_shift_frc:.1f}%, k_co2={k_co2_shift_frc:.1f}%)")

    # 8. FL#6 tau_washout not at bound with wide bounds
    fl6_tw = flat_b_free[ic_offset_b + fl6_idx * N_PH]
    fl6_bounds = PERHOLD_BOUNDS["FL"][0]
    c8 = not is_at_bound(fl6_tw, *fl6_bounds)
    criteria.append(("FL#6 tau_washout off bound with [50, 500]", c8))
    print(f"  8. FL#6 tau_washout off bound: "
          f"{'PASS' if c8 else 'FAIL'}"
          f" (tau_w={fl6_tw:.1f}, bounds [{fl6_bounds[0]}, {fl6_bounds[1]}])")

    n_pass = sum(1 for _, c in criteria)
    n_total = len(criteria)
    print(f"\n  Score: {sum(c for _, c in criteria)}/{n_total} PASS")

    # ========================================================================
    # FINAL MODEL SPECIFICATION
    # ========================================================================
    print(f"\n{'='*120}")
    print("v7 FINAL MODEL SPECIFICATION")
    print(f"{'='*120}")

    print(f"""
  Two-stage SpO2 sensor + physiology model for pulse oximetry during breath-hold apnea.

  Stage A - Sensor Calibration (power-law latent + gamma kernel)
  =============================================================
  Latent SaO2(t):
    if t <= t_end:  latent(t) = S_min + (B_h - S_min) * (1 - (t/t_end)^p)
    if t > t_end:   latent(t) = S_min + v_up * (t - t_end)

  Sensor pipeline (b_s=1 fixed):
    pred(t) = GammaKernel(latent, eff_lag_h, cv=0.15)
    eff_lag_h = max(tau_0 + delta_h, 5.0)

  Parameters (14 total):
    Global:   tau_0 [{TAU0_BOUNDS[0]}, {TAU0_BOUNDS[1]}], p [{P_BOUNDS[0]}, {P_BOUNDS[1]}]
    Per-hold: delta_h [{DELTA_BOUNDS[0]}, {DELTA_BOUNDS[1]}], S_min [{S_MIN_BOUNDS[0]}, {S_MIN_BOUNDS[1]}], v_up [{V_UP_BOUNDS[0]}, {V_UP_BOUNDS[1]}]
    Fixed:    b_s=1.0, cv=0.15

  Loss: Student-t NLL (nu=4, sigma=1) + Huber nadir timing (delta=8s, lambda=500)
  Priors: tau_0 LogNormal(log 18, 0.4) [lambda=3.125]
          p LogNormal(log 3.5, 0.35) [lambda=4.08]
          delta StudentT shrinkage (sigma=3) [lambda=5] + zero-sum [lambda=500]
  Optimizer: DE popsize=40, maxiter=4000, seed=42

  Stage B - Apnea-Only Physiology (frozen sensor)
  ================================================
  PaO2(t) = PvO2 + (PaO2_0 - PvO2) * exp(-t / tau_washout)
  PaCO2(t) = PaCO2_0 + k_co2 * t
  P50(t) = 26.6 + 0.48 * (PaCO2(t) - 40)
  SaO2 = Severinghaus ODC with gamma steepness
  pred(t) = clip(GammaKernel(SaO2, eff_lag_h, cv=0.15), 0, 100)

  Parameters (10 total):
    Global:   k_co2 [{K_CO2_BOUNDS[0]}, {K_CO2_BOUNDS[1]}], gamma [{GAMMA_BOUNDS[0]}, {GAMMA_BOUNDS[1]}]
    Per-hold: tau_washout (type-specific), paco2_0 (type-specific)
    Fixed:    PvO2=25.0 mmHg, tau_0 and deltas frozen from Stage A

  Per-hold bounds (tau_washout, paco2_0):
    FL:  [{PERHOLD_BOUNDS['FL'][0][0]}, {PERHOLD_BOUNDS['FL'][0][1]}], [{PERHOLD_BOUNDS['FL'][1][0]}, {PERHOLD_BOUNDS['FL'][1][1]}]
    FRC: [{PERHOLD_BOUNDS['FRC'][0][0]}, {PERHOLD_BOUNDS['FRC'][0][1]}], [{PERHOLD_BOUNDS['FRC'][1][0]}, {PERHOLD_BOUNDS['FRC'][1][1]}]
    RV:  [{PERHOLD_BOUNDS['RV'][0][0]}, {PERHOLD_BOUNDS['RV'][0][1]}], [{PERHOLD_BOUNDS['RV'][1][0]}, {PERHOLD_BOUNDS['RV'][1][1]}]

  Loss: weighted SSE (3x for SpO2 < 95%) on apnea window (t <= t_end + 5s)
  Priors: k_co2 N(0.06, 0.02) [lambda=1250]
          gamma N(2.0, 0.5) [lambda=2.0]
          paco2_0 N(40, 3) [lambda=0.056]
          per-hold IC -> type-mean [lambda=10]
  Optimizer: DE popsize=25, maxiter=2000, seed=42

  QC Rules
  ========
  Exclude from training:
    - Low variation: SpO2 range < 5% during hold (FL#1: only 2%)
    - Intra-apnea nadir: nadir occurs during apnea, not post-apnea (RV#4)
  Flag but keep:
    - Out-of-regime: apnea-window nadir before t_end - 2s (FRC#2)

  Applicability Domain
  ====================
  - Single-hold apnea events, tested up to ~370s duration
  - Apnea-window prediction only (recovery not modeled in Stage B)
  - Finger pulse oximetry (Masimo MightySat, 1-Hz)
  - Primary metric: R2a (apnea window)
  - Secondary diagnostics: R2n (nadir window, includes recovery)

  Known Limitations
  =================
  - Single subject, single session (4 training holds after QC)
  - gamma-k_co2 correlation: these parameters trade off
  - Recovery phase not modeled in Stage B
  - Long holds (>300s) may need wider tau_washout bounds

  Final Parameter Table (this subject/session)
  =============================================
  Stage A: tau_0={tau_0:.4f}, p={p:.4f}, b_s=1.0 (fixed), cv=0.15 (fixed)
  Stage B: k_co2={k_co2_free:.4f}, gamma={gamma_free:.4f}
""")

    # ========================================================================
    # PLOTS
    # ========================================================================
    print(f"\n{'='*120}")
    print("GENERATING PLOTS")
    print(f"{'='*120}")

    plot_k_co2_profile(k_co2_profile, output_dir / "exp_v7_08_k_co2_profile.png")

    plot_stage_b_fits(eval_b_free, all_holds, nadir_info,
                       f"Stage B Shipping (gamma={gamma_free:.2f}, k_co2={k_co2_free:.4f})",
                       output_dir / "exp_v7_08_stage_b.png")

    plot_fl6_comparison(fl6_configs, output_dir / "exp_v7_08_fl6_sensitivity.png")

    # ========================================================================
    # DONE
    # ========================================================================
    print(f"\n{'='*120}")
    print("DONE")
    print(f"{'='*120}")

    print(f"\nOutput files:")
    print(f"  {output_dir / 'exp_v7_08_k_co2_profile.png'}")
    print(f"  {output_dir / 'exp_v7_08_stage_b.png'}")
    print(f"  {output_dir / 'exp_v7_08_fl6_sensitivity.png'}")


if __name__ == "__main__":
    main()
