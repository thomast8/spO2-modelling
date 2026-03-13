"""
v7 Experiment 06: b_s Ablation Study.

v7.05 established the power-law descent + baseline-locked sensor model as structurally
sound: baseline ambiguity solved, shape-lag degeneracy largely broken, tau_0 and p
identifiable. But b_s = 1.75 remains interpretively problematic for a "smoothing + delay"
sensor model.

v7.06 changes from v7.05:
  1. Exclude RV#4 from calibration (train on {FRC#2, RV#3, FRC#5, FL#6}).
     RV#4 has fundamentally different regime (nadir during apnea, -43% influence).
     Report as stress-test only.
  2. b_s ablation: 3 configurations run on the same 4-hold training set:
     - Config A: b_s free, CV=0.15 (baseline without RV#4)
     - Config B: b_s fixed=1, CV=0.15
     - Config C: b_s free, CV=0.10
  3. Recenter p prior: LogNormal(log 3.5, 0.35) instead of LogNormal(log 2, 0.35).
     Profile minimum was at p=4.0; old prior actively pulled away from optimum.
  4. Widen gamma bounds in Stage B to [0.8, 2.0].
  5. True-frozen Stage B: freeze both tau_0 and per-hold delta_h from Stage A.
     Weak-lag as diagnostic with tight prior (sigma=2s around Stage A delta).
  6. RV#4 stress test: infer delta + latent on frozen globals, report R2.

Carries forward from v7.05:
  Power-law descent, baseline-corrected measurement, nadir-window fitting,
  Student-t NLL + Huber timing penalty, LOHO-Inference, saturation diagnostic,
  QC flagging, DE popsize=40/maxiter=4000.

Usage:
    cd backend && uv run python -u scripts/experiments/exp_v7_06/exp_v7_06_ablation.py
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
from scipy.optimize import differential_evolution, minimize
from scipy.special import gammainc

N_WORKERS = min(max(1, os.cpu_count() - 1), 8)  # Cap at 8 to avoid pool deadlocks


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


# ── Paths and constants ─────────────────────────────────────────────────────

DB_PATH = Path(__file__).resolve().parents[4] / "data" / "spo2.db"

P50_BASE = 26.6
P_EQ = 100.0
PACO2_NORMAL = 40.0
TAU_CLEAR_FIXED = 30.0
FIO2_PB_PH2O = 149.2  # FiO2 * (PB - PH2O) = 0.2093 * (760 - 47)
RQ = 0.8

# FL#1 excluded (only 2% SpO2 variation), RV#4 excluded (nadir during apnea, -43% influence)
EXCLUDED_IDS = {1, 4}

# ── Ablation configurations ─────────────────────────────────────────────────

CONFIGS = {
    "A": {"label": "b_s free, CV=0.15", "bs_fixed": None, "cv": 0.15},
    "B": {"label": "b_s=1 fixed, CV=0.15", "bs_fixed": 1.0, "cv": 0.15},
    "C": {"label": "b_s free, CV=0.10", "bs_fixed": None, "cv": 0.10},
}

# ── Stage A: Sensor regularization (MAP-correct: lambda = 1/(2*sigma^2)) ───

LAMBDA_TAU0 = 3.125        # LogNormal(log 18, 0.4): 1/(2*0.4^2) = 3.125
LAMBDA_DELTA = 5.0         # StudentT-like shrinkage, sigma ~3s
LAMBDA_ZEROSUM = 500.0     # Zero-sum on deltas
LAMBDA_GAIN = 50.0         # N(1, 0.1): 1/(2*0.1^2) = 50
LAMBDA_P = 4.08            # LogNormal(log 3.5, 0.35): 1/(2*0.35^2) = 4.08
LAMBDA_NADIR = 500.0       # Huber timing penalty (delta=8s)
EFF_LAG_MIN = 5.0          # Minimum effective lag floor (seconds)

# v7.06: p prior recentered from 2.0 to 3.5
P_PRIOR_CENTER = 3.5

# ── Stage B: Physiology regularization ──────────────────────────────────────

PVO2_FIXED = 25.0
LAMBDA_K_CO2 = 1250.0      # N(0.06, 0.02): 1/(2*0.02^2) = 1250
LAMBDA_PACO2 = 0.056       # N(40, 3): 1/(2*3^2) = 0.056
LAMBDA_GAMMA = 22.2        # N(1, 0.15): 1/(2*0.15^2) = 22.2
LAMBDA_REG = 10.0          # Per-hold IC -> type-mean

# ── Shared constants ────────────────────────────────────────────────────────

TAU0_PRIOR_CENTER = 18.0
NADIR_WINDOW_AFTER = 45

# ── Bounds ──────────────────────────────────────────────────────────────────

# Stage A
TAU0_BOUNDS = (5, 45)
GAIN_BOUNDS = (0.5, 2.0)
P_BOUNDS = (1.0, 5.0)
DELTA_BOUNDS = (-20, 20)

# Power-law latent
S_MIN_BOUNDS = (30, 100)
V_UP_BOUNDS = (0.0, 3.0)

# Stage B - v7.06: gamma widened from [0.8, 1.3] to [0.8, 2.0]
PERHOLD_BOUNDS = {
    "FL": [(50, 250), (20, 50)],
    "FRC": [(20, 100), (25, 50)],
    "RV": [(10, 80), (30, 55)],
}
PERHOLD_NAMES = ["tau_washout", "paco2_0"]
N_PH = len(PERHOLD_NAMES)
GAMMA_BOUNDS = (0.8, 2.0)

# Student-t NLL parameters
NU_STUDENT = 4.0
SIGMA_STUDENT = 1.0


# ── Data loading ────────────────────────────────────────────────────────────


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


# ── Physiology functions ────────────────────────────────────────────────────


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


# ── Smooth Discrete Gamma Kernel ───────────────────────────────────────────


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


# ── Power-law latent template ──────────────────────────────────────────────


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


# ── Student-t NLL loss ─────────────────────────────────────────────────────


def student_t_nll(residuals, nu=NU_STUDENT, sigma=SIGMA_STUDENT):
    return (nu + 1.0) / 2.0 * np.sum(np.log1p(residuals**2 / (nu * sigma**2)))


# ── Nadir + loss helpers ───────────────────────────────────────────────────


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


# ── Metrics ─────────────────────────────────────────────────────────────────


def compute_r2(obs, pred):
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def compute_rmse(obs, pred):
    return float(np.sqrt(np.mean((obs - pred) ** 2)))


def is_at_bound(val, lo, hi, tol=1e-3):
    return abs(val - lo) < tol or abs(val - hi) < tol


# ── Stage A: Sensor-First Calibration ───────────────────────────────────────


def run_stage_a(fit_holds, nadir_info, s_base_values, cv_fixed, bs_fixed=None):
    """Stage A: Sensor calibration with power-law latent.

    If bs_fixed is not None, b_s is fixed to that value (not optimized).
    """
    n_holds = len(fit_holds)

    bounds = [TAU0_BOUNDS]
    global_names = ["tau_0"]
    if bs_fixed is None:
        bounds.append(GAIN_BOUNDS)
        global_names.append("b_s")
    bounds.append(P_BOUNDS)
    global_names.append("p")
    n_global = len(bounds)

    # Per-hold deltas
    delta_offset = n_global
    for _ in fit_holds:
        bounds.append(DELTA_BOUNDS)

    # Per-hold latent params: S_min, v_up
    latent_offset = delta_offset + n_holds
    n_latent_per_hold = 2
    for _ in fit_holds:
        bounds.append(S_MIN_BOUNDS)
        bounds.append(V_UP_BOUNDS)

    n_total = len(bounds)
    bs_label = f"b_s={'free' if bs_fixed is None else f'{bs_fixed:.1f} (fixed)'}"

    print(f"\n  Stage A: {n_total} params ({n_global} global + {n_holds} delta + "
          f"{n_latent_per_hold}x{n_holds} latent)")
    print(f"  {bs_label}, cv={cv_fixed}")

    masks = [nadir_window_mask(h["t"], h["t_end"]) for h in fit_holds]
    nadir_ts = [nadir_info[h["id"]]["t_nadir"] for h in fit_holds]
    t_grids = [np.arange(0, h["t"][-1] + 1, 1.0) for h in fit_holds]

    def _unpack_globals(flat):
        idx = 0
        tau_0 = flat[idx]; idx += 1
        if bs_fixed is None:
            b_s = flat[idx]; idx += 1
        else:
            b_s = bs_fixed
        p = flat[idx]
        return tau_0, b_s, p

    def objective(flat):
        tau_0, b_s, p = _unpack_globals(flat)
        deltas = flat[delta_offset:delta_offset + n_holds]
        total = 0.0

        for i, h in enumerate(fit_holds):
            lp_start = latent_offset + i * n_latent_per_hold
            S_min, v_up = flat[lp_start:lp_start + n_latent_per_hold]
            S_start = s_base_values[h["id"]]
            t_1hz = t_grids[i]
            latent = build_powerlaw_latent(t_1hz, h["t_end"], S_start, S_min, v_up, p)
            eff_lag = max(tau_0 + deltas[i], EFF_LAG_MIN)
            filtered = apply_gamma_kernel(latent, eff_lag, cv_fixed)
            B_h = s_base_values[h["id"]]
            pred_1hz = B_h + b_s * (filtered - B_h)
            pred_at_obs = np.interp(h["t"], t_1hz, pred_1hz)
            pred_display = np.clip(pred_at_obs, 0.0, 100.0)
            m = masks[i]
            residuals = h["spo2"][m] - pred_at_obs[m]
            total += student_t_nll(residuals)
            total += nadir_timing_penalty_huber(h["t"][m], pred_display[m], nadir_ts[i])

        # Priors
        total += LAMBDA_TAU0 * (np.log(max(tau_0, 1.0)) - np.log(TAU0_PRIOR_CENTER)) ** 2
        total += LAMBDA_DELTA * np.sum(np.log1p(deltas**2 / 9.0))
        total += LAMBDA_ZEROSUM * np.sum(deltas) ** 2
        if bs_fixed is None:
            total += LAMBDA_GAIN * (b_s - 1.0) ** 2
        # v7.06: p prior centered at 3.5 instead of 2.0
        total += LAMBDA_P * (np.log(max(p, 0.1)) - np.log(P_PRIOR_CENTER)) ** 2

        return total

    maxiter_a = 4000
    res = differential_evolution(
        objective, bounds, maxiter=maxiter_a, seed=42, tol=1e-10,
        polish=True, popsize=40, mutation=(0.5, 1.5), recombination=0.9,
        callback=make_de_callback("Stage A", maxiter_a),
    )
    print(f"\n  Converged: {res.success}, fun={res.fun:.2f}, nfev={res.nfev}")

    # Reconstruct full parameter vector with b_s included at index 1
    if bs_fixed is not None:
        full = np.empty(len(res.x) + 1)
        full[0] = res.x[0]       # tau_0
        full[1] = bs_fixed        # b_s (fixed)
        full[2:] = res.x[1:]     # p, deltas, latent params
    else:
        full = res.x.copy()

    return full, res.success, res.fun


def evaluate_stage_a(flat_a, fit_holds, nadir_info, s_base_values, cv_fixed):
    """Evaluate Stage A results. flat_a always has b_s at index 1."""
    n_holds = len(fit_holds)
    n_global = 3  # tau_0, b_s, p (always in full vector)
    delta_offset = n_global
    latent_offset = delta_offset + n_holds
    n_latent_per_hold = 2

    tau_0, b_s, p = flat_a[:n_global]
    deltas = flat_a[delta_offset:delta_offset + n_holds]

    results = []
    latent_curves = []

    for i, h in enumerate(fit_holds):
        lp_start = latent_offset + i * n_latent_per_hold
        S_min, v_up = flat_a[lp_start:lp_start + n_latent_per_hold]
        S_start = s_base_values[h["id"]]

        t_1hz = np.arange(0, h["t"][-1] + 1, 1.0)
        latent = build_powerlaw_latent(t_1hz, h["t_end"], S_start, S_min, v_up, p)
        eff_lag = max(tau_0 + deltas[i], EFF_LAG_MIN)
        filtered = apply_gamma_kernel(latent, eff_lag, cv_fixed)
        B_h = s_base_values[h["id"]]
        pred_1hz = B_h + b_s * (filtered - B_h)
        pred_at_obs = np.interp(h["t"], t_1hz, pred_1hz)
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
            "variant": "A:sensor",
            "hold_id": h["id"],
            "hold_type": h["type"],
            "r2_full": r2_full,
            "r2_apnea": r2_apnea,
            "r2_nadir": r2_nadir,
            "pred_full": pred_at_obs,
            "nadir_err": nadir_err,
            "t_nadir_pred": t_nadir_pred,
            "is_excluded": h["id"] in EXCLUDED_IDS,
            "effective_lag": eff_lag,
            "delta": deltas[i],
        })
        latent_curves.append({
            "hold_id": h["id"],
            "t_1hz": t_1hz,
            "latent": latent,
            "filtered": pred_1hz,
            "S_start": S_start,
            "S_min": S_min,
            "v_up": v_up,
        })

    return results, latent_curves


def extract_frozen_sensor(flat_a, fit_holds, cv_fixed):
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
        "cv": cv_fixed,
    }


# ── Stage B: Physiology (apnea-only, frozen sensor) ────────────────────────


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


def run_stage_b_true_frozen(fit_holds, nadir_info, frozen_sensor, s_base_values,
                            label="Stage B (true-frozen)"):
    """Stage B with truly frozen sensor: tau_0 AND deltas from Stage A are frozen.

    No delta re-fitting allowed. Only physiology params + per-hold ICs are free.
    """
    tau_0_frozen = frozen_sensor["tau_0"]
    b_s_frozen = frozen_sensor["b_s"]
    cv_frozen = frozen_sensor["cv"]
    deltas_frozen = frozen_sensor["deltas"]

    n_holds = len(fit_holds)

    bounds = [
        (0.02, 0.25),  # k_co2
        GAMMA_BOUNDS,  # gamma [0.8, 2.0]
    ]
    n_phys = len(bounds)

    # Per-hold ICs only (no deltas)
    ic_offset = n_phys
    for h in fit_holds:
        bounds.extend(PERHOLD_BOUNDS[h["type"]])
    n_total = len(bounds)

    type_groups = {}
    for i, h in enumerate(fit_holds):
        type_groups.setdefault(h["type"], []).append(i)

    apnea_window = 5
    masks = [h["t"] <= h["t_end"] + apnea_window for h in fit_holds]
    weights = [np.where(h["spo2"][m] < 95, 3.0, 1.0) for h, m in zip(fit_holds, masks)]

    print(f"\n  {label}: {n_total} params ({n_phys} physiology + "
          f"{N_PH}x{n_holds} per-hold ICs, NO deltas)")
    print(f"  Frozen sensor: tau_0={tau_0_frozen:.2f}, cv={cv_frozen:.3f}, "
          f"b_s={b_s_frozen:.4f}")
    print(f"  Frozen deltas: {[f'{d:+.2f}' for d in deltas_frozen]}")

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
        total += LAMBDA_GAMMA * (gamma_val - 1.0) ** 2

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

    # Reconstruct full flat_b with frozen deltas inserted for evaluate_stage_b compatibility
    k_co2, gamma_val = res.x[:n_phys]
    ics = res.x[n_phys:]
    full_b = np.concatenate([
        [k_co2, gamma_val],
        deltas_frozen,
        ics,
    ])
    return full_b, res.success


def run_stage_b_weak_lag(fit_holds, nadir_info, frozen_sensor, s_base_values,
                         label="Stage B (weak-lag)"):
    """Stage B with weak lag prior: delta free with N(delta_stageA, sigma=2s)."""
    tau_0_frozen = frozen_sensor["tau_0"]
    b_s_frozen = frozen_sensor["b_s"]
    cv_frozen = frozen_sensor["cv"]
    delta_a_values = frozen_sensor["deltas"]

    n_holds = len(fit_holds)
    bounds = [
        (0.02, 0.25),  # k_co2
        GAMMA_BOUNDS,  # gamma [0.8, 2.0]
    ]
    n_phys = len(bounds)

    delta_offset = n_phys
    for _ in fit_holds:
        bounds.append(DELTA_BOUNDS)

    ic_offset = delta_offset + n_holds
    for h in fit_holds:
        bounds.extend(PERHOLD_BOUNDS[h["type"]])

    type_groups = {}
    for i, h in enumerate(fit_holds):
        type_groups.setdefault(h["type"], []).append(i)

    apnea_window = 5
    masks = [h["t"] <= h["t_end"] + apnea_window for h in fit_holds]
    weights = [np.where(h["spo2"][m] < 95, 3.0, 1.0) for h, m in zip(fit_holds, masks)]

    # v7.06: tight prior sigma=2s (was 3s in v7.05)
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
        total += LAMBDA_GAMMA * (gamma_val - 1.0) ** 2

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

        rec = {
            "variant": "B:physiology",
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
        }
        results.append(rec)
    return results


# ── LOHO: Leave-One-Hold-Out for Stage A ────────────────────────────────────


def _loho_worker(args):
    """Worker for one LOHO fold."""
    leave_idx, fit_holds, nadir_info, s_base_values, cv_fixed, bs_fixed = args
    left_out = fit_holds[leave_idx]
    train_holds = [h for i, h in enumerate(fit_holds) if i != leave_idx]
    n_train = len(train_holds)

    # Phase 1: Train on train holds
    bounds = [TAU0_BOUNDS]
    if bs_fixed is None:
        bounds.append(GAIN_BOUNDS)
    bounds.append(P_BOUNDS)
    n_global = len(bounds)

    delta_offset = n_global
    for _ in train_holds:
        bounds.append(DELTA_BOUNDS)

    latent_offset = delta_offset + n_train
    for _ in train_holds:
        bounds.extend([S_MIN_BOUNDS, V_UP_BOUNDS])

    masks_train = [nadir_window_mask(h["t"], h["t_end"]) for h in train_holds]
    nadir_ts_train = [nadir_info[h["id"]]["t_nadir"] for h in train_holds]
    t_grids_train = [np.arange(0, h["t"][-1] + 1, 1.0) for h in train_holds]

    def _unpack(flat):
        idx = 0
        tau_0 = flat[idx]; idx += 1
        if bs_fixed is None:
            b_s = flat[idx]; idx += 1
        else:
            b_s = bs_fixed
        p = flat[idx]
        return tau_0, b_s, p

    def train_objective(flat):
        tau_0, b_s, p = _unpack(flat)
        deltas = flat[delta_offset:delta_offset + n_train]
        total = 0.0

        for i, h in enumerate(train_holds):
            lp_start = latent_offset + i * 2
            S_min, v_up = flat[lp_start:lp_start + 2]
            S_start = s_base_values[h["id"]]
            t_1hz = t_grids_train[i]
            latent = build_powerlaw_latent(t_1hz, h["t_end"], S_start, S_min, v_up, p)
            eff_lag = max(tau_0 + deltas[i], EFF_LAG_MIN)
            filtered = apply_gamma_kernel(latent, eff_lag, cv_fixed)
            B_h = s_base_values[h["id"]]
            pred_1hz = B_h + b_s * (filtered - B_h)
            pred_at_obs = np.interp(h["t"], t_1hz, pred_1hz)
            pred_display = np.clip(pred_at_obs, 0.0, 100.0)
            m = masks_train[i]
            total += student_t_nll(h["spo2"][m] - pred_at_obs[m])
            total += nadir_timing_penalty_huber(h["t"][m], pred_display[m], nadir_ts_train[i])

        total += LAMBDA_TAU0 * (np.log(max(tau_0, 1.0)) - np.log(TAU0_PRIOR_CENTER)) ** 2
        total += LAMBDA_DELTA * np.sum(np.log1p(deltas**2 / 9.0))
        total += LAMBDA_ZEROSUM * np.sum(deltas) ** 2
        if bs_fixed is None:
            total += LAMBDA_GAIN * (b_s - 1.0) ** 2
        total += LAMBDA_P * (np.log(max(p, 0.1)) - np.log(P_PRIOR_CENTER)) ** 2
        return total

    res = differential_evolution(
        train_objective, bounds, maxiter=2000, seed=42, tol=1e-10,
        polish=True, popsize=25, mutation=(0.5, 1.5), recombination=0.9,
    )

    # Phase 2: Held-out
    tau_0_fit, b_s_fit, p_fit = _unpack(res.x)

    eff_lag_ho = max(tau_0_fit, EFF_LAG_MIN)
    t_1hz_ho = np.arange(0, left_out["t"][-1] + 1, 1.0)
    mask_ho = nadir_window_mask(left_out["t"], left_out["t_end"])
    nadir_t_ho = nadir_info[left_out["id"]]["t_nadir"]
    S_start_ho = s_base_values[left_out["id"]]
    B_h_ho = S_start_ho

    # Phase 2a: LOHO-Global (2D: S_min, v_up, no delta)
    ho_bounds = [S_MIN_BOUNDS, V_UP_BOUNDS]

    def ho_objective(x):
        S_min, v_up = x
        latent = build_powerlaw_latent(t_1hz_ho, left_out["t_end"], S_start_ho, S_min, v_up, p_fit)
        filtered = apply_gamma_kernel(latent, eff_lag_ho, cv_fixed)
        pred_1hz = B_h_ho + b_s_fit * (filtered - B_h_ho)
        pred_at_obs = np.interp(left_out["t"], t_1hz_ho, pred_1hz)
        pred_display = np.clip(pred_at_obs, 0.0, 100.0)
        m = mask_ho
        total = student_t_nll(left_out["spo2"][m] - pred_at_obs[m])
        total += nadir_timing_penalty_huber(left_out["t"][m], pred_display[m], nadir_t_ho)
        return total

    ho_res = differential_evolution(
        ho_objective, ho_bounds, maxiter=500, seed=42 + leave_idx, tol=1e-10,
        polish=True, popsize=20, mutation=(0.5, 1.5), recombination=0.9,
    )
    ho_S_min, ho_v_up = ho_res.x

    latent_ho = build_powerlaw_latent(
        t_1hz_ho, left_out["t_end"], S_start_ho, ho_S_min, ho_v_up, p_fit)
    filtered_ho = apply_gamma_kernel(latent_ho, eff_lag_ho, cv_fixed)
    pred_ho_1hz = B_h_ho + b_s_fit * (filtered_ho - B_h_ho)
    pred_ho = np.interp(left_out["t"], t_1hz_ho, pred_ho_1hz)
    pred_ho = np.clip(pred_ho, 0.0, 100.0)

    r2_ho = compute_r2(left_out["spo2"][mask_ho], pred_ho[mask_ho]) if mask_ho.sum() > 3 else None
    rmse_ho = compute_rmse(left_out["spo2"][mask_ho], pred_ho[mask_ho]) if mask_ho.sum() > 3 else None

    t_nadir_obs = nadir_info[left_out["id"]]["t_nadir"]
    t_nadir_pred = left_out["t"][np.argmin(pred_ho)]
    nadir_err = t_nadir_pred - t_nadir_obs

    # Phase 2b: LOHO-Inference (3D: S_min, v_up, delta_ho)
    ho_bounds_inf = [S_MIN_BOUNDS, V_UP_BOUNDS, DELTA_BOUNDS]

    def ho_inference_objective(x):
        S_min, v_up, delta_ho = x
        latent = build_powerlaw_latent(t_1hz_ho, left_out["t_end"], S_start_ho, S_min, v_up, p_fit)
        eff_lag_inf = max(tau_0_fit + delta_ho, EFF_LAG_MIN)
        filtered = apply_gamma_kernel(latent, eff_lag_inf, cv_fixed)
        pred_1hz = B_h_ho + b_s_fit * (filtered - B_h_ho)
        pred_at_obs = np.interp(left_out["t"], t_1hz_ho, pred_1hz)
        pred_display = np.clip(pred_at_obs, 0.0, 100.0)
        m = mask_ho
        total = student_t_nll(left_out["spo2"][m] - pred_at_obs[m])
        total += nadir_timing_penalty_huber(left_out["t"][m], pred_display[m], nadir_t_ho)
        total += LAMBDA_DELTA * np.log1p(delta_ho**2 / 9.0)
        return total

    ho_res_inf = differential_evolution(
        ho_inference_objective, ho_bounds_inf, maxiter=500, seed=42 + leave_idx, tol=1e-10,
        polish=True, popsize=20, mutation=(0.5, 1.5), recombination=0.9,
    )
    inf_S_min, inf_v_up, inf_delta = ho_res_inf.x
    eff_lag_inf = max(tau_0_fit + inf_delta, EFF_LAG_MIN)

    latent_inf = build_powerlaw_latent(
        t_1hz_ho, left_out["t_end"], S_start_ho, inf_S_min, inf_v_up, p_fit)
    filtered_inf = apply_gamma_kernel(latent_inf, eff_lag_inf, cv_fixed)
    pred_inf_1hz = B_h_ho + b_s_fit * (filtered_inf - B_h_ho)
    pred_inf = np.interp(left_out["t"], t_1hz_ho, pred_inf_1hz)
    pred_inf = np.clip(pred_inf, 0.0, 100.0)

    r2_inf = compute_r2(left_out["spo2"][mask_ho], pred_inf[mask_ho]) if mask_ho.sum() > 3 else None
    rmse_inf = compute_rmse(left_out["spo2"][mask_ho], pred_inf[mask_ho]) if mask_ho.sum() > 3 else None
    t_nadir_pred_inf = left_out["t"][np.argmin(pred_inf)]
    nadir_err_inf = t_nadir_pred_inf - t_nadir_obs

    return {
        "hold_id": left_out["id"],
        "hold_type": left_out["type"],
        "r2": r2_ho,
        "rmse": rmse_ho,
        "nadir_err": nadir_err,
        "eff_lag": eff_lag_ho,
        "pred": pred_ho,
        "converged": res.success,
        "loss": res.fun,
        "ho_loss": ho_res.fun,
        "r2_inf": r2_inf,
        "rmse_inf": rmse_inf,
        "nadir_err_inf": nadir_err_inf,
        "eff_lag_inf": eff_lag_inf,
        "inf_delta": inf_delta,
        "inf_loss": ho_res_inf.fun,
    }


def run_stage_a_loho(fit_holds, nadir_info, s_base_values, cv_fixed, bs_fixed):
    n_holds = len(fit_holds)
    print(f"\n  Running {n_holds} LOHO folds in parallel ({N_WORKERS} workers)...")

    args_list = [(i, fit_holds, nadir_info, s_base_values, cv_fixed, bs_fixed)
                 for i in range(n_holds)]

    with _mp_ctx.Pool(processes=min(N_WORKERS, n_holds)) as pool:
        loho_results = pool.map(_loho_worker, args_list)

    for r in loho_results:
        print(f"    Held-out {r['hold_type']}#{r['hold_id']}:")
        print(f"      Global:    R2={r['r2']:.4f}, RMSE={r['rmse']:.2f}, "
              f"timing_err={r['nadir_err']:+.1f}s, eff_lag={r['eff_lag']:.1f}s")
        r2_inf_str = f"{r['r2_inf']:.4f}" if r['r2_inf'] is not None else "N/A"
        rmse_inf_str = f"{r['rmse_inf']:.2f}" if r['rmse_inf'] is not None else "N/A"
        print(f"      Inference: R2={r2_inf_str}, RMSE={rmse_inf_str}, "
              f"timing_err={r['nadir_err_inf']:+.1f}s, eff_lag={r['eff_lag_inf']:.1f}s, "
              f"delta={r['inf_delta']:+.1f}s")

    return loho_results


# ── Profile likelihood ──────────────────────────────────────────────────────


def _profile_tau_worker(args):
    tau_fixed, fit_holds, nadir_info, s_base_values, cv_fixed, bs_fixed = args
    n_holds = len(fit_holds)
    masks = [nadir_window_mask(h["t"], h["t_end"]) for h in fit_holds]
    nadir_ts = [nadir_info[h["id"]]["t_nadir"] for h in fit_holds]
    t_grids = [np.arange(0, h["t"][-1] + 1, 1.0) for h in fit_holds]

    bounds = [(tau_fixed - 0.01, tau_fixed + 0.01)]
    if bs_fixed is None:
        bounds.append(GAIN_BOUNDS)
    bounds.append(P_BOUNDS)
    n_global = len(bounds)
    delta_offset = n_global
    for _ in fit_holds:
        bounds.append(DELTA_BOUNDS)
    latent_offset = delta_offset + n_holds
    for _ in fit_holds:
        bounds.extend([S_MIN_BOUNDS, V_UP_BOUNDS])

    def _unpack(flat):
        idx = 0
        tau_0 = flat[idx]; idx += 1
        if bs_fixed is None:
            b_s = flat[idx]; idx += 1
        else:
            b_s = bs_fixed
        p = flat[idx]
        return tau_0, b_s, p

    def objective(flat):
        tau_0, b_s, p = _unpack(flat)
        deltas = flat[delta_offset:delta_offset + n_holds]
        total = 0.0
        for i, h in enumerate(fit_holds):
            lp_start = latent_offset + i * 2
            S_min, v_up = flat[lp_start:lp_start + 2]
            S_start = s_base_values[h["id"]]
            t_1hz = t_grids[i]
            latent = build_powerlaw_latent(t_1hz, h["t_end"], S_start, S_min, v_up, p)
            eff_lag = max(tau_0 + deltas[i], EFF_LAG_MIN)
            filtered = apply_gamma_kernel(latent, eff_lag, cv_fixed)
            B_h = s_base_values[h["id"]]
            pred_1hz = B_h + b_s * (filtered - B_h)
            pred_at_obs = np.interp(h["t"], t_1hz, pred_1hz)
            pred_display = np.clip(pred_at_obs, 0.0, 100.0)
            m = masks[i]
            total += student_t_nll(h["spo2"][m] - pred_at_obs[m])
            total += nadir_timing_penalty_huber(h["t"][m], pred_display[m], nadir_ts[i])
        total += LAMBDA_TAU0 * (np.log(max(tau_0, 1.0)) - np.log(TAU0_PRIOR_CENTER)) ** 2
        total += LAMBDA_DELTA * np.sum(np.log1p(deltas**2 / 9.0))
        total += LAMBDA_ZEROSUM * np.sum(deltas) ** 2
        if bs_fixed is None:
            total += LAMBDA_GAIN * (b_s - 1.0) ** 2
        total += LAMBDA_P * (np.log(max(p, 0.1)) - np.log(P_PRIOR_CENTER)) ** 2
        return total

    res = differential_evolution(
        objective, bounds, maxiter=2000, seed=42, tol=1e-10,
        polish=True, popsize=25, mutation=(0.5, 1.5), recombination=0.9,
    )
    # Return full vector with b_s inserted if it was fixed
    if bs_fixed is not None:
        full = np.empty(len(res.x) + 1)
        full[0] = res.x[0]
        full[1] = bs_fixed
        full[2:] = res.x[1:]
    else:
        full = res.x.copy()
    return tau_fixed, {"flat": full, "loss": res.fun, "success": res.success}


def _profile_p_worker(args):
    p_fixed, fit_holds, nadir_info, s_base_values, cv_fixed, bs_fixed = args
    n_holds = len(fit_holds)
    masks = [nadir_window_mask(h["t"], h["t_end"]) for h in fit_holds]
    nadir_ts = [nadir_info[h["id"]]["t_nadir"] for h in fit_holds]
    t_grids = [np.arange(0, h["t"][-1] + 1, 1.0) for h in fit_holds]

    bounds = [TAU0_BOUNDS]
    if bs_fixed is None:
        bounds.append(GAIN_BOUNDS)
    bounds.append((p_fixed - 0.01, p_fixed + 0.01))
    n_global = len(bounds)
    delta_offset = n_global
    for _ in fit_holds:
        bounds.append(DELTA_BOUNDS)
    latent_offset = delta_offset + n_holds
    for _ in fit_holds:
        bounds.extend([S_MIN_BOUNDS, V_UP_BOUNDS])

    def _unpack(flat):
        idx = 0
        tau_0 = flat[idx]; idx += 1
        if bs_fixed is None:
            b_s = flat[idx]; idx += 1
        else:
            b_s = bs_fixed
        p = flat[idx]
        return tau_0, b_s, p

    def objective(flat):
        tau_0, b_s, p = _unpack(flat)
        deltas = flat[delta_offset:delta_offset + n_holds]
        total = 0.0
        for i, h in enumerate(fit_holds):
            lp_start = latent_offset + i * 2
            S_min, v_up = flat[lp_start:lp_start + 2]
            S_start = s_base_values[h["id"]]
            t_1hz = t_grids[i]
            latent = build_powerlaw_latent(t_1hz, h["t_end"], S_start, S_min, v_up, p)
            eff_lag = max(tau_0 + deltas[i], EFF_LAG_MIN)
            filtered = apply_gamma_kernel(latent, eff_lag, cv_fixed)
            B_h = s_base_values[h["id"]]
            pred_1hz = B_h + b_s * (filtered - B_h)
            pred_at_obs = np.interp(h["t"], t_1hz, pred_1hz)
            pred_display = np.clip(pred_at_obs, 0.0, 100.0)
            m = masks[i]
            total += student_t_nll(h["spo2"][m] - pred_at_obs[m])
            total += nadir_timing_penalty_huber(h["t"][m], pred_display[m], nadir_ts[i])
        total += LAMBDA_TAU0 * (np.log(max(tau_0, 1.0)) - np.log(TAU0_PRIOR_CENTER)) ** 2
        total += LAMBDA_DELTA * np.sum(np.log1p(deltas**2 / 9.0))
        total += LAMBDA_ZEROSUM * np.sum(deltas) ** 2
        if bs_fixed is None:
            total += LAMBDA_GAIN * (b_s - 1.0) ** 2
        total += LAMBDA_P * (np.log(max(p, 0.1)) - np.log(P_PRIOR_CENTER)) ** 2
        return total

    res = differential_evolution(
        objective, bounds, maxiter=2000, seed=42, tol=1e-10,
        polish=True, popsize=25, mutation=(0.5, 1.5), recombination=0.9,
    )
    if bs_fixed is not None:
        full = np.empty(len(res.x) + 1)
        full[0] = res.x[0]
        full[1] = bs_fixed
        full[2:] = res.x[1:]
    else:
        full = res.x.copy()
    return p_fixed, {"flat": full, "loss": res.fun, "success": res.success}


def run_profile_likelihood(fit_holds, nadir_info, s_base_values, cv_fixed, bs_fixed,
                           tau_0_values=None, p_values=None):
    if tau_0_values is None:
        tau_0_values = [5, 8, 10, 12, 14, 16, 18, 20, 22, 25, 30, 35, 40, 45]
    if p_values is None:
        p_values = [1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

    print(f"\n  Running {len(tau_0_values)} tau_0 profile points in parallel ({N_WORKERS} workers)...")
    args_tau = [(tv, fit_holds, nadir_info, s_base_values, cv_fixed, bs_fixed)
                for tv in tau_0_values]
    with _mp_ctx.Pool(processes=N_WORKERS) as pool:
        raw_tau = pool.map(_profile_tau_worker, args_tau)

    tau_results = {}
    for tau_val, result in raw_tau:
        tau_results[tau_val] = result
        b_s = result["flat"][1]
        p = result["flat"][2]
        print(f"    tau_0={tau_val:5.1f}: loss={result['loss']:.2f}, "
              f"b_s={b_s:.3f}, p={p:.3f}", flush=True)

    print(f"\n  Running {len(p_values)} p profile points in parallel ({N_WORKERS} workers)...")
    args_p = [(pv, fit_holds, nadir_info, s_base_values, cv_fixed, bs_fixed)
              for pv in p_values]
    with _mp_ctx.Pool(processes=N_WORKERS) as pool:
        raw_p = pool.map(_profile_p_worker, args_p)

    p_results = {}
    for p_val, result in raw_p:
        p_results[p_val] = result
        tau_0 = result["flat"][0]
        b_s = result["flat"][1]
        print(f"    p={p_val:5.2f}: loss={result['loss']:.2f}, "
              f"tau_0={tau_0:.3f}, b_s={b_s:.3f}", flush=True)

    return tau_results, p_results


# ── Sponge diagnostics ─────────────────────────────────────────────────────


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


# ── RV#4 Stress Test ───────────────────────────────────────────────────────


def run_rv4_stress_test(rv4_hold, nadir_info, s_base_values, frozen_sensor, cv_fixed):
    """Infer RV#4 latent (S_min, v_up, delta) on frozen globals from Stage A."""
    tau_0 = frozen_sensor["tau_0"]
    b_s = frozen_sensor["b_s"]
    p = frozen_sensor["p"]
    cv = cv_fixed

    t_1hz = np.arange(0, rv4_hold["t"][-1] + 1, 1.0)
    mask = nadir_window_mask(rv4_hold["t"], rv4_hold["t_end"])
    nadir_t = nadir_info[rv4_hold["id"]]["t_nadir"]
    S_start = s_base_values[rv4_hold["id"]]
    B_h = S_start

    bounds_inf = [S_MIN_BOUNDS, V_UP_BOUNDS, DELTA_BOUNDS]

    def objective(x):
        S_min, v_up, delta = x
        latent = build_powerlaw_latent(t_1hz, rv4_hold["t_end"], S_start, S_min, v_up, p)
        eff_lag = max(tau_0 + delta, EFF_LAG_MIN)
        filtered = apply_gamma_kernel(latent, eff_lag, cv)
        pred_1hz = B_h + b_s * (filtered - B_h)
        pred_at_obs = np.interp(rv4_hold["t"], t_1hz, pred_1hz)
        pred_display = np.clip(pred_at_obs, 0.0, 100.0)
        m = mask
        total = student_t_nll(rv4_hold["spo2"][m] - pred_at_obs[m])
        total += nadir_timing_penalty_huber(rv4_hold["t"][m], pred_display[m], nadir_t)
        total += LAMBDA_DELTA * np.log1p(delta**2 / 9.0)
        return total

    res = differential_evolution(
        objective, bounds_inf, maxiter=1000, seed=42, tol=1e-10,
        polish=True, popsize=20, mutation=(0.5, 1.5), recombination=0.9,
    )
    S_min, v_up, delta = res.x
    eff_lag = max(tau_0 + delta, EFF_LAG_MIN)

    latent = build_powerlaw_latent(t_1hz, rv4_hold["t_end"], S_start, S_min, v_up, p)
    filtered = apply_gamma_kernel(latent, eff_lag, cv)
    pred_1hz = B_h + b_s * (filtered - B_h)
    pred = np.interp(rv4_hold["t"], t_1hz, pred_1hz)
    pred = np.clip(pred, 0.0, 100.0)

    r2_nadir = compute_r2(rv4_hold["spo2"][mask], pred[mask]) if mask.sum() > 3 else None
    t_nadir_pred = rv4_hold["t"][np.argmin(pred)]
    nadir_err = t_nadir_pred - nadir_info[rv4_hold["id"]]["t_nadir"]

    return {
        "S_min": S_min,
        "v_up": v_up,
        "delta": delta,
        "eff_lag": eff_lag,
        "r2_nadir": r2_nadir,
        "nadir_err": nadir_err,
        "pred": pred,
    }


# ── Plots ────────────────────────────────────────────────────────────────────


def plot_stage_a_comparison(config_results, all_holds, nadir_info, output_path):
    """Per-hold Stage A plots for all 3 configs side by side."""
    holds_dict = {h["id"]: h for h in all_holds}
    config_keys = sorted(config_results.keys())
    n_configs = len(config_keys)

    # Get holds from first config
    first_eval = config_results[config_keys[0]]["eval_a"]
    hold_ids = [r["hold_id"] for r in first_eval]
    n_holds = len(hold_ids)

    fig, axes = plt.subplots(n_holds, n_configs, figsize=(6 * n_configs, 4.5 * n_holds),
                             squeeze=False)

    for col, ck in enumerate(config_keys):
        eval_a = config_results[ck]["eval_a"]
        latent_curves = config_results[ck]["latent_curves"]
        cfg = CONFIGS[ck]

        for row, (res, lc) in enumerate(zip(eval_a, latent_curves)):
            ax = axes[row, col]
            h = holds_dict[res["hold_id"]]
            ni = nadir_info[res["hold_id"]]

            ax.plot(h["t"], h["spo2"], "k.", markersize=2, alpha=0.5, label="Observed")
            ax.axvline(x=h["t_end"], color="red", linestyle="--", alpha=0.5)
            ax.plot(ni["t_nadir"], ni["spo2_nadir"], "r*", markersize=10, zorder=5)

            ax.plot(lc["t_1hz"], lc["latent"], color="#ff7f0e", linewidth=1.5, alpha=0.6,
                    label=f"Latent (S_min={lc['S_min']:.1f})")
            ax.plot(h["t"], res["pred_full"], color="#1f77b4", linewidth=2, alpha=0.8,
                    label=f"Pred (R2n={res['r2_nadir']:.3f})")

            if row == 0:
                ax.set_title(f"Config {ck}: {cfg['label']}", fontsize=11, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{h['type']}#{h['id']}\nSpO2 (%)")
            ax.set_ylim(30, 105)
            ax.legend(fontsize=6, loc="lower left")
            ax.grid(True, alpha=0.3)
            if row == n_holds - 1:
                ax.set_xlabel("Time (s)")

    fig.suptitle("v7.06 Stage A: Config Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nStage A comparison plot saved to {output_path}")


def plot_loho_comparison(config_results, output_path):
    """LOHO comparison: R2 and timing error bars for all configs."""
    config_keys = sorted(config_results.keys())
    n_configs = len(config_keys)

    fig, axes = plt.subplots(2, n_configs, figsize=(5 * n_configs, 8), squeeze=False)

    for col, ck in enumerate(config_keys):
        loho = config_results[ck]["loho"]
        labels = [f"{r['hold_type']}#{r['hold_id']}" for r in loho]

        # LOHO-Inference R2
        r2s = [r["r2_inf"] if r["r2_inf"] is not None else 0 for r in loho]
        axes[0, col].bar(labels, r2s, color="#1f77b4", alpha=0.8)
        axes[0, col].set_ylim(0, 1)
        axes[0, col].axhline(y=0.9, color="red", linestyle="--", alpha=0.5)
        axes[0, col].set_title(f"Config {ck}: LOHO-Inf R2n", fontweight="bold")
        axes[0, col].grid(True, alpha=0.3)

        # LOHO-Inference timing error
        nerrs = [r["nadir_err_inf"] for r in loho]
        colors = ["#2ca02c" if abs(e) < 5 else "#ff7f0e" if abs(e) < 10 else "#d62728"
                  for e in nerrs]
        axes[1, col].bar(labels, nerrs, color=colors, alpha=0.8)
        axes[1, col].axhline(y=0, color="k", linewidth=0.5)
        axes[1, col].set_title(f"Config {ck}: Timing Err", fontweight="bold")
        axes[1, col].grid(True, alpha=0.3)

    fig.suptitle("v7.06 LOHO-Inference Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"LOHO comparison plot saved to {output_path}")


def plot_profile_comparison(config_results, output_path):
    """Profile likelihood: tau_0 and p for all configs overlaid."""
    config_keys = sorted(config_results.keys())
    colors = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ck in config_keys:
        tau_profile = config_results[ck]["tau_profile"]
        p_profile = config_results[ck]["p_profile"]
        c = colors[ck]
        label = f"Config {ck}"

        # tau_0 loss
        tau_list = sorted(tau_profile.keys())
        losses = [tau_profile[t]["loss"] for t in tau_list]
        axes[0, 0].plot(tau_list, losses, "o-", color=c, linewidth=2, markersize=5, label=label)

        # tau_0 b_s
        bs_vals = [tau_profile[t]["flat"][1] for t in tau_list]
        axes[0, 1].plot(tau_list, bs_vals, "o-", color=c, linewidth=2, markersize=5, label=label)

        # p loss
        p_list = sorted(p_profile.keys())
        losses_p = [p_profile[pv]["loss"] for pv in p_list]
        axes[1, 0].plot(p_list, losses_p, "s-", color=c, linewidth=2, markersize=5, label=label)

        # p tau_0
        tau_vals = [p_profile[pv]["flat"][0] for pv in p_list]
        axes[1, 1].plot(p_list, tau_vals, "s-", color=c, linewidth=2, markersize=5, label=label)

    axes[0, 0].set_xlabel("tau_0 (fixed)"); axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("tau_0 Profile: Loss", fontweight="bold")
    axes[0, 1].set_xlabel("tau_0 (fixed)"); axes[0, 1].set_ylabel("b_s")
    axes[0, 1].set_title("tau_0 Profile: b_s", fontweight="bold")
    axes[1, 0].set_xlabel("p (fixed)"); axes[1, 0].set_ylabel("Loss")
    axes[1, 0].set_title("p Profile: Loss", fontweight="bold")
    axes[1, 1].set_xlabel("p (fixed)"); axes[1, 1].set_ylabel("tau_0")
    axes[1, 1].set_title("p Profile: tau_0", fontweight="bold")

    for ax in axes.flat:
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("v7.06 Profile Likelihood Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Profile comparison plot saved to {output_path}")


def plot_stage_b_comparison(config_results, all_holds, nadir_info, output_path):
    """Stage B true-frozen plots for all configs."""
    holds_dict = {h["id"]: h for h in all_holds}
    config_keys = sorted(config_results.keys())
    n_configs = len(config_keys)

    first_eval = config_results[config_keys[0]]["eval_b"]
    fitted = [r for r in first_eval if not r["is_excluded"]]
    n_holds = len(fitted)

    fig, axes = plt.subplots(n_holds, n_configs, figsize=(6 * n_configs, 4.5 * n_holds),
                             squeeze=False)

    for col, ck in enumerate(config_keys):
        eval_b = config_results[ck]["eval_b"]
        fitted_b = [r for r in eval_b if not r["is_excluded"]]
        cfg = CONFIGS[ck]

        for row, res in enumerate(fitted_b):
            ax = axes[row, col]
            h = holds_dict[res["hold_id"]]
            ni = nadir_info[res["hold_id"]]

            ax.plot(h["t"], h["spo2"], "k.", markersize=2, alpha=0.5, label="Observed")
            ax.axvline(x=h["t_end"], color="red", linestyle="--", alpha=0.5)
            ax.plot(ni["t_nadir"], ni["spo2_nadir"], "r*", markersize=10, zorder=5)

            r2_str = f"R2a={res['r2_apnea']:.3f}" if res["r2_apnea"] is not None else ""
            ax.plot(h["t"], res["pred_full"], color="#2ca02c", linewidth=2, alpha=0.8,
                    label=f"Stage B ({r2_str})")

            if row == 0:
                ax.set_title(f"Config {ck}: {cfg['label']}", fontsize=11, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{h['type']}#{h['id']}\nSpO2 (%)")
            ax.set_ylim(30, 105)
            ax.legend(fontsize=6, loc="lower left")
            ax.grid(True, alpha=0.3)
            if row == n_holds - 1:
                ax.set_xlabel("Time (s)")

    fig.suptitle("v7.06 Stage B (True-Frozen): Config Comparison",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Stage B comparison plot saved to {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────


def run_one_config(config_key, fit_holds, all_holds, nadir_info, s_base_values, qc_flags):
    """Run full pipeline for one ablation configuration."""
    cfg = CONFIGS[config_key]
    cv_fixed = cfg["cv"]
    bs_fixed = cfg["bs_fixed"]
    n_holds = len(fit_holds)

    print(f"\n{'#'*120}")
    print(f"# CONFIG {config_key}: {cfg['label']}")
    print(f"{'#'*120}")

    # ── Stage A ─────────────────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"STAGE A: Config {config_key}")
    print(f"{'='*100}")

    flat_a, conv_a, loss_a = run_stage_a(fit_holds, nadir_info, s_base_values,
                                          cv_fixed, bs_fixed)

    # Print globals (flat_a always has [tau_0, b_s, p, ...])
    tau_0, b_s, p = flat_a[:3]
    global_names = ["tau_0", "b_s", "p"]
    global_bounds = [TAU0_BOUNDS, GAIN_BOUNDS, P_BOUNDS]
    print(f"\n  Stage A global params:")
    for name, val, (lo, hi) in zip(global_names, flat_a[:3], global_bounds):
        fixed_str = " [FIXED]" if (name == "b_s" and bs_fixed is not None) else ""
        flag = " ** AT BOUND **" if is_at_bound(val, lo, hi) and not fixed_str else ""
        print(f"    {name:>12s} = {val:8.4f}  [{lo:>6.2f}, {hi:>6.2f}]{flag}{fixed_str}")

    k_val = 1.0 / (cv_fixed * cv_fixed)
    print(f"\n    Kernel: k={k_val:.2f}, mean={tau_0:.1f}s, std={tau_0*cv_fixed:.1f}s "
          f"(cv={cv_fixed})")

    delta_offset = 3
    latent_offset = delta_offset + n_holds
    deltas_a = flat_a[delta_offset:delta_offset + n_holds]

    print(f"\n  Per-hold breakdown:")
    for i, h in enumerate(fit_holds):
        residual = deltas_a[i]
        eff = max(tau_0 + residual, EFF_LAG_MIN)
        bound_str = " *BOUND*" if is_at_bound(residual, *DELTA_BOUNDS) else ""
        eff_floor_str = " *FLOOR*" if eff <= EFF_LAG_MIN + 0.01 else ""
        qc_str = " [QC-FLAGGED]" if qc_flags[h["id"]] else ""
        lp_start = latent_offset + i * 2
        S_min, v_up = flat_a[lp_start:lp_start + 2]
        print(f"    {h['type']}#{h['id']}: "
              f"delta={residual:+6.2f}, eff_lag={eff:6.2f}, "
              f"B_h={s_base_values[h['id']]:.1f}, "
              f"S_min={S_min:.1f}, v_up={v_up:.2f}{bound_str}{eff_floor_str}{qc_str}")

    eval_a, latent_curves = evaluate_stage_a(flat_a, fit_holds, nadir_info,
                                              s_base_values, cv_fixed)
    for r in eval_a:
        r["is_qc_flagged"] = qc_flags.get(r["hold_id"], False)

    print(f"\n  Stage A fit metrics:")
    for r in eval_a:
        r2n = f"{r['r2_nadir']:.4f}" if r['r2_nadir'] is not None else "N/A"
        r2a = f"{r['r2_apnea']:.4f}" if r['r2_apnea'] is not None else "N/A"
        qc_str = " [QC]" if r.get("is_qc_flagged") else ""
        print(f"    {r['hold_type']}#{r['hold_id']}{qc_str}: "
              f"R2a={r2a}, R2n={r2n}, nadir_err={r['nadir_err']:+.1f}s")

    # ── LOHO ────────────────────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"LOHO: Config {config_key}")
    print(f"{'='*100}")

    loho_results = run_stage_a_loho(fit_holds, nadir_info, s_base_values,
                                     cv_fixed, bs_fixed)

    # LOHO-Inference summary
    loho_inf_r2s = [r["r2_inf"] for r in loho_results if r["r2_inf"] is not None]
    avg_r2_inf = np.mean(loho_inf_r2s) if loho_inf_r2s else float("nan")
    loho_inf_nerrs = [abs(r["nadir_err_inf"]) for r in loho_results]
    avg_nerr_inf = np.mean(loho_inf_nerrs) if loho_inf_nerrs else float("nan")
    print(f"\n  LOHO-Inference avg: R2n={avg_r2_inf:.4f}, |timing|={avg_nerr_inf:.1f}s")

    # ── Profile likelihood ──────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"PROFILE: Config {config_key}")
    print(f"{'='*100}")

    tau_profile, p_profile = run_profile_likelihood(
        fit_holds, nadir_info, s_base_values, cv_fixed, bs_fixed)

    # Check monotonicity
    tau_list = sorted(tau_profile.keys())
    losses_tau = [tau_profile[t]["loss"] for t in tau_list]
    is_mono_tau = (all(losses_tau[i] >= losses_tau[i+1] for i in range(len(losses_tau)-1)) or
                   all(losses_tau[i] <= losses_tau[i+1] for i in range(len(losses_tau)-1)))
    min_tau = tau_list[np.argmin(losses_tau)]
    print(f"\n  tau_0 non-monotone: {'YES' if not is_mono_tau else 'NO'}, min at {min_tau:.1f}")

    p_list = sorted(p_profile.keys())
    losses_p = [p_profile[pv]["loss"] for pv in p_list]
    is_mono_p = (all(losses_p[i] >= losses_p[i+1] for i in range(len(losses_p)-1)) or
                 all(losses_p[i] <= losses_p[i+1] for i in range(len(losses_p)-1)))
    min_p = p_list[np.argmin(losses_p)]
    print(f"  p non-monotone: {'YES' if not is_mono_p else 'NO'}, min at p={min_p:.2f}")

    # ── Stage B: True-Frozen ────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"STAGE B TRUE-FROZEN: Config {config_key}")
    print(f"{'='*100}")

    frozen_sensor = extract_frozen_sensor(flat_a, fit_holds, cv_fixed)
    flat_b, conv_b = run_stage_b_true_frozen(
        fit_holds, nadir_info, frozen_sensor, s_base_values)

    # Print Stage B results
    k_co2_b, gamma_b = flat_b[:2]
    phys_bounds = [(0.02, 0.25), GAMMA_BOUNDS]
    phys_names = ["k_co2", "gamma"]
    print(f"\n  Stage B physiology:")
    print(f"    {'pvo2':>12s} = {PVO2_FIXED:8.4f}  [FIXED]")
    for name, val, (lo, hi) in zip(phys_names, flat_b[:2], phys_bounds):
        flag = " ** AT BOUND **" if is_at_bound(val, lo, hi) else ""
        print(f"    {name:>12s} = {val:8.4f}  [{lo:>6.2f}, {hi:>6.2f}]{flag}")

    # Per-hold ICs
    n_phys_b = 2
    ic_offset_b = n_phys_b + n_holds  # deltas are in the flat_b
    print(f"\n  Per-hold ICs:")
    for i, h in enumerate(fit_holds):
        offset = ic_offset_b + i * N_PH
        tau_washout, paco2_0 = flat_b[offset:offset + N_PH]
        pao2_0 = corrected_pao2_0(paco2_0, 0.0)
        print(f"    {h['type']}#{h['id']}: tau_w={tau_washout:.1f}, "
              f"paco2_0={paco2_0:.1f}, PaO2_0={pao2_0:.1f}")

    eval_b = evaluate_stage_b(flat_b, fit_holds, nadir_info, frozen_sensor,
                               s_base_values, all_holds)
    for r in eval_b:
        r["is_qc_flagged"] = qc_flags.get(r["hold_id"], False)

    print(f"\n  Stage B fit metrics:")
    for r in eval_b:
        if r["is_excluded"]:
            continue
        r2n = f"{r['r2_nadir']:.4f}" if r['r2_nadir'] is not None else "N/A"
        r2a = f"{r['r2_apnea']:.4f}" if r['r2_apnea'] is not None else "N/A"
        print(f"    {r['hold_type']}#{r['hold_id']}: "
              f"R2a={r2a}, R2n={r2n}, nadir_err={r['nadir_err']:+.1f}s")

    # ── Stage B: Weak-lag diagnostic ────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"STAGE B WEAK-LAG: Config {config_key}")
    print(f"{'='*100}")

    flat_b_wl, conv_b_wl = run_stage_b_weak_lag(
        fit_holds, nadir_info, frozen_sensor, s_base_values)

    # Compare frozen vs weak-lag physiology
    print(f"\n  Frozen vs Weak-lag:")
    for pi, pname in enumerate(phys_names):
        v_frozen = flat_b[pi]
        v_weak = flat_b_wl[pi]
        pct_diff = abs(v_weak - v_frozen) / max(abs(v_frozen), 1e-6) * 100
        print(f"    {pname}: frozen={v_frozen:.4f}, weak={v_weak:.4f}, diff={pct_diff:.1f}%")

    max_phys_diff = max(
        abs(flat_b_wl[pi] - flat_b[pi]) / max(abs(flat_b[pi]), 1e-6) * 100
        for pi in range(2)
    )

    # ── Sponge diagnostics ──────────────────────────────────────────────────
    a_param_names = ["tau_0", "b_s", "p"]
    a_bounds = [TAU0_BOUNDS, GAIN_BOUNDS, P_BOUNDS]
    a_prior_sigmas = {"tau_0": 0.4, "b_s": 0.1, "p": 0.35}
    at_bound_a = sponge_diagnostics(flat_a, a_bounds, a_prior_sigmas, a_param_names,
                                     label=f"Stage A (Config {config_key})")

    b_bounds = [(0.02, 0.25), GAMMA_BOUNDS]
    b_prior_sigmas = {"k_co2": 0.02, "gamma": 0.15}
    at_bound_b = sponge_diagnostics(flat_b, b_bounds, b_prior_sigmas, phys_names,
                                     label=f"Stage B (Config {config_key})")

    # Collect results
    return {
        "flat_a": flat_a,
        "loss_a": loss_a,
        "eval_a": eval_a,
        "latent_curves": latent_curves,
        "loho": loho_results,
        "tau_profile": tau_profile,
        "p_profile": p_profile,
        "frozen_sensor": frozen_sensor,
        "flat_b": flat_b,
        "eval_b": eval_b,
        "flat_b_wl": flat_b_wl,
        "max_phys_diff": max_phys_diff,
        "at_bound_a": at_bound_a,
        "at_bound_b": at_bound_b,
        "is_mono_tau": is_mono_tau,
        "is_mono_p": is_mono_p,
        "min_tau": min_tau,
        "min_p": min_p,
    }


def main():
    print("=" * 120)
    print("v7.06: b_s Ablation Study (3 configs, RV#4 excluded)")
    print("=" * 120)

    # ── Load data ────────────────────────────────────────────────────────────
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
    print(f"\nFitting on {len(fit_holds)} holds, "
          f"excluding {sum(1 for h in all_holds if h['id'] in EXCLUDED_IDS)} hold(s)")
    train_labels = [f"{h['type']}#{h['id']}" for h in fit_holds]
    print(f"Training set: {train_labels}")

    # ── Nadir info ──────────────────────────────────────────────────────────
    nadir_info = {}
    for h in all_holds:
        ni = compute_nadir_info(h)
        nadir_info[h["id"]] = ni
        tag = " [EXCLUDED]" if h["id"] in EXCLUDED_IDS else ""
        loc = "recovery" if ni["in_recovery"] else "apnea"
        print(f"  {h['type']}#{h['id']}{tag}: nadir at t={ni['t_nadir']:.0f}s "
              f"({loc}, delay={ni['delay_from_end']:+.0f}s)")

    # ── Baselines ───────────────────────────────────────────────────────────
    s_base_values = {}
    for h in all_holds:
        plateau_mask = h["t"] <= 20
        if plateau_mask.sum() > 0:
            s_base_values[h["id"]] = float(np.median(h["spo2"][plateau_mask]))
        else:
            s_base_values[h["id"]] = float(h["spo2"][0])

    # ── QC flags ────────────────────────────────────────────────────────────
    qc_flags = {}
    for h in all_holds:
        ni = nadir_info[h["id"]]
        qc_flags[h["id"]] = ni["t_nadir_apnea"] < h["t_end"] - 2

    # ── Changes from v7.05 ──────────────────────────────────────────────────
    print(f"\nKey changes from v7.05:")
    print(f"  1. RV#4 excluded from calibration (stress-test only)")
    print(f"  2. b_s ablation: 3 configs (A: free/CV=0.15, B: fixed=1/CV=0.15, C: free/CV=0.10)")
    print(f"  3. p prior recentered: LogNormal(log {P_PRIOR_CENTER}, 0.35)")
    print(f"  4. gamma bounds widened to [{GAMMA_BOUNDS[0]}, {GAMMA_BOUNDS[1]}]")
    print(f"  5. Stage B true-frozen (no delta re-fitting)")
    print(f"  6. Weak-lag sigma=2s (was 3s)")

    # ══════════════════════════════════════════════════════════════════════════
    # RUN ALL 3 CONFIGS
    # ══════════════════════════════════════════════════════════════════════════
    config_results = {}
    for ck in sorted(CONFIGS.keys()):
        config_results[ck] = run_one_config(
            ck, fit_holds, all_holds, nadir_info, s_base_values, qc_flags)

    # ══════════════════════════════════════════════════════════════════════════
    # RV#4 STRESS TEST (all configs)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("RV#4 STRESS TEST (excluded hold, infer on frozen globals)")
    print(f"{'='*120}")

    rv4_hold = next(h for h in all_holds if h["id"] == 4)
    for ck in sorted(CONFIGS.keys()):
        cfg = CONFIGS[ck]
        frozen = config_results[ck]["frozen_sensor"]
        rv4_res = run_rv4_stress_test(rv4_hold, nadir_info, s_base_values,
                                       frozen, cfg["cv"])
        config_results[ck]["rv4_stress"] = rv4_res
        r2_str = f"{rv4_res['r2_nadir']:.4f}" if rv4_res['r2_nadir'] is not None else "N/A"
        print(f"  Config {ck}: R2n={r2_str}, nadir_err={rv4_res['nadir_err']:+.1f}s, "
              f"delta={rv4_res['delta']:+.1f}s, S_min={rv4_res['S_min']:.1f}")

    # ══════════════════════════════════════════════════════════════════════════
    # COMPARISON TABLE
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("ABLATION COMPARISON TABLE")
    print(f"{'='*120}")

    n_holds = len(fit_holds)

    # Header
    print(f"\n  {'Metric':<40s}", end="")
    for ck in sorted(CONFIGS.keys()):
        print(f" | {'Config ' + ck:>15s}", end="")
    print()
    print(f"  {'-'*40}" + ("-+-" + "-"*15) * 3)

    # Global params
    for pi, pname in enumerate(["tau_0", "b_s", "p"]):
        print(f"  {pname:<40s}", end="")
        for ck in sorted(CONFIGS.keys()):
            val = config_results[ck]["flat_a"][pi]
            fixed = " (F)" if pname == "b_s" and CONFIGS[ck]["bs_fixed"] is not None else ""
            print(f" | {val:>10.4f}{fixed:>4s}", end="")
        print()

    # Loss
    print(f"  {'Stage A loss':<40s}", end="")
    for ck in sorted(CONFIGS.keys()):
        print(f" | {config_results[ck]['loss_a']:>15.2f}", end="")
    print()

    # LOHO-Inference avg R2
    print(f"  {'LOHO-Inf avg R2n':<40s}", end="")
    for ck in sorted(CONFIGS.keys()):
        r2s = [r["r2_inf"] for r in config_results[ck]["loho"] if r["r2_inf"] is not None]
        avg = np.mean(r2s) if r2s else float("nan")
        print(f" | {avg:>15.4f}", end="")
    print()

    # LOHO-Inference avg timing error
    print(f"  {'LOHO-Inf avg |timing|':<40s}", end="")
    for ck in sorted(CONFIGS.keys()):
        nerrs = [abs(r["nadir_err_inf"]) for r in config_results[ck]["loho"]]
        avg = np.mean(nerrs) if nerrs else float("nan")
        print(f" | {avg:>14.1f}s", end="")
    print()

    # Per-hold LOHO-Inference R2
    for r_idx in range(len(fit_holds)):
        hid = fit_holds[r_idx]["id"]
        htype = fit_holds[r_idx]["type"]
        print(f"  {'LOHO-Inf R2n ' + htype + '#' + str(hid):<40s}", end="")
        for ck in sorted(CONFIGS.keys()):
            loho = config_results[ck]["loho"]
            r = next((x for x in loho if x["hold_id"] == hid), None)
            if r and r["r2_inf"] is not None:
                print(f" | {r['r2_inf']:>15.4f}", end="")
            else:
                print(f" | {'N/A':>15s}", end="")
        print()

    # Delta range
    print(f"  {'Delta range (s)':<40s}", end="")
    for ck in sorted(CONFIGS.keys()):
        deltas = config_results[ck]["flat_a"][3:3+n_holds]
        dr = float(np.max(deltas) - np.min(deltas))
        print(f" | {dr:>14.1f}s", end="")
    print()

    # Profile non-monotone
    print(f"  {'tau_0 profile non-monotone':<40s}", end="")
    for ck in sorted(CONFIGS.keys()):
        val = "YES" if not config_results[ck]["is_mono_tau"] else "NO"
        print(f" | {val:>15s}", end="")
    print()

    print(f"  {'p profile non-monotone':<40s}", end="")
    for ck in sorted(CONFIGS.keys()):
        val = "YES" if not config_results[ck]["is_mono_p"] else "NO"
        print(f" | {val:>15s}", end="")
    print()

    # gamma at bound
    print(f"  {'gamma value':<40s}", end="")
    for ck in sorted(CONFIGS.keys()):
        g = config_results[ck]["flat_b"][1]
        bound = " *BOUND*" if is_at_bound(g, *GAMMA_BOUNDS) else ""
        print(f" | {g:>10.4f}{bound:>4s}", end="")
    print()

    # Frozen vs weak-lag divergence
    print(f"  {'Frozen vs weak-lag div (%)':<40s}", end="")
    for ck in sorted(CONFIGS.keys()):
        print(f" | {config_results[ck]['max_phys_diff']:>14.1f}%", end="")
    print()

    # RV#4 stress test
    print(f"  {'RV#4 stress R2n':<40s}", end="")
    for ck in sorted(CONFIGS.keys()):
        rv4 = config_results[ck]["rv4_stress"]
        r2_str = f"{rv4['r2_nadir']:.4f}" if rv4['r2_nadir'] is not None else "N/A"
        print(f" | {r2_str:>15s}", end="")
    print()

    # ══════════════════════════════════════════════════════════════════════════
    # DECISION CRITERIA
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("DECISION CRITERIA")
    print(f"{'='*120}")

    for ck in sorted(CONFIGS.keys()):
        cr = config_results[ck]
        cfg = CONFIGS[ck]
        tau_0, b_s, p = cr["flat_a"][:3]
        deltas = cr["flat_a"][3:3+n_holds]
        loho_inf_r2s = [r["r2_inf"] for r in cr["loho"] if r["r2_inf"] is not None]
        avg_loho_r2 = np.mean(loho_inf_r2s) if loho_inf_r2s else float("nan")

        print(f"\n  Config {ck}: {cfg['label']}")
        print(f"    1. LOHO-Inf avg R2n = {avg_loho_r2:.4f} (>= 0.90: "
              f"{'PASS' if avg_loho_r2 >= 0.90 else 'FAIL'})")
        print(f"    2. b_s = {b_s:.4f} (proximity to 1.0: {abs(b_s - 1.0):.4f})")
        print(f"    3. tau_0 non-monotone: {'PASS' if not cr['is_mono_tau'] else 'FAIL'}, "
              f"min at {cr['min_tau']:.1f}")
        print(f"    4. p non-monotone: {'PASS' if not cr['is_mono_p'] else 'FAIL'}, "
              f"min at {cr['min_p']:.2f}")
        n_free = 3 if cfg["bs_fixed"] is None else 2
        print(f"    5. Free global params: {n_free}")
        delta_range = float(np.max(deltas) - np.min(deltas))
        print(f"    6. Delta range: {delta_range:.1f}s")
        print(f"    7. gamma = {cr['flat_b'][1]:.4f} "
              f"(interior: {'YES' if not is_at_bound(cr['flat_b'][1], *GAMMA_BOUNDS) else 'NO'})")

    # ── Pick winner ─────────────────────────────────────────────────────────
    print(f"\n  WINNER SELECTION:")
    best_ck = None
    best_score = -1
    for ck in sorted(CONFIGS.keys()):
        cr = config_results[ck]
        loho_inf_r2s = [r["r2_inf"] for r in cr["loho"] if r["r2_inf"] is not None]
        avg_r2 = np.mean(loho_inf_r2s) if loho_inf_r2s else 0
        b_s = cr["flat_a"][1]
        # Score: R2 is primary, b_s proximity to 1 is tiebreaker
        score = avg_r2 - 0.01 * abs(b_s - 1.0)
        mono_ok = not cr["is_mono_tau"] and not cr["is_mono_p"]
        if not mono_ok:
            score -= 0.1  # penalty for monotone profile
        print(f"    Config {ck}: score={score:.4f} (R2={avg_r2:.4f}, |b_s-1|={abs(b_s-1):.4f}, "
              f"profiles={'ok' if mono_ok else 'MONO'})")
        if score > best_score:
            best_score = score
            best_ck = ck

    print(f"\n  >>> WINNER: Config {best_ck} ({CONFIGS[best_ck]['label']}) <<<")

    # ── Plots ───────────────────────────────────────────────────────────────
    output_dir = Path(__file__).resolve().parent

    plot_stage_a_comparison(config_results, all_holds, nadir_info,
                            output_dir / "exp_v7_06_stage_a.png")

    plot_loho_comparison(config_results, output_dir / "exp_v7_06_loho.png")

    plot_profile_comparison(config_results, output_dir / "exp_v7_06_profile.png")

    plot_stage_b_comparison(config_results, all_holds, nadir_info,
                            output_dir / "exp_v7_06_stage_b.png")

    print(f"\n{'='*120}")
    print("DONE")
    print(f"{'='*120}")


if __name__ == "__main__":
    main()
