"""
v7 Experiment 05: Power-Law Descent + Baseline-Locked Sensor Model.

v7.04 killed the covariate sponge, constrained m_h >= 0, added DE for LOHO held-out,
and added the frozen-lag vs weak-lag diagnostic (passed at 0.4% divergence).

But v7.04 exposed three structural limitations:
  1. Linear latent + normalized kernel = no interior curvature. The gamma kernel
     integrates to 1. Convolving a linear ramp produces the same slope, just shifted
     by the kernel mean. eff_lag spans 1.0 to 37.3 (37x ratio) - model artifact.
  2. r_offset/S_start/b_s baseline ambiguity. r_offset = 2.55, S_start ranges 94.8-100,
     saturation = 17.6%.
  3. PvO2 stuck at lower bound (15.0) everywhere - uninformative.

v7.05 changes:
  1. Power-law descent: s(t) = S_min + (S_start - S_min) * (1 - (t/t_turn)^p).
     Global p adds curvature separating latent shape from sensor delay.
  2. Fix S_start = B_h = median(SpO2[t<=20]) - baseline locked by construction.
  3. Remove r_offset. Measurement: pred = B_h + b_s * (filtered - B_h).
  4. Fix m_h = 0. t_turn = t_end for all holds.
  5. EFF_LAG_MIN = 5s floor.
  6. Fix PvO2 = 25 mmHg in Stage B.
  7. Add p profile likelihood.

Carries forward from v7.04:
  No-clip-in-loss, Student-t NLL, Huber timing, LOHO-Inference, saturation diagnostic,
  QC flagging, Stage B frozen-lag vs weak-lag, DE popsize=40/maxiter=4000, cv=0.15.

Usage:
    cd backend && uv run python -u scripts/experiments/exp_v7_05/exp_v7_05_powerlaw.py
"""

import csv
import io
import os
import sqlite3
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.special import gammainc

N_WORKERS = max(1, os.cpu_count() - 1)  # Leave one core free


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

EXCLUDED_IDS = {1}  # FL#1 excluded (only 2% SpO2 variation)

# ── Stage A: Sensor regularization (MAP-correct: lambda = 1/(2*sigma^2)) ───

LAMBDA_TAU0 = 3.125        # LogNormal(log 18, 0.4): 1/(2*0.4^2) = 3.125
LAMBDA_DELTA = 5.0         # StudentT-like shrinkage, sigma ~3s (structural, not a prior)
LAMBDA_ZEROSUM = 500.0     # Zero-sum on deltas (structural constraint)
LAMBDA_GAIN = 50.0         # N(1, 0.1): 1/(2*0.1^2) = 50
LAMBDA_P = 4.08            # LogNormal(log 2, 0.35): 1/(2*0.35^2) = 4.08
LAMBDA_NADIR = 500.0       # Huber timing penalty (delta=8s) (structural, not a prior)
CV_FIXED = 0.15            # Fixed kernel shape
EFF_LAG_MIN = 5.0          # Minimum effective lag floor (seconds)

# ── Stage B: Physiology regularization (MAP-correct: lambda = 1/(2*sigma^2))

PVO2_FIXED = 25.0           # Fixed PvO2 (was uninformative, always at lower bound)
LAMBDA_K_CO2 = 1250.0      # N(0.06, 0.02): 1/(2*0.02^2) = 1250
LAMBDA_PACO2 = 0.056       # N(40, 3): 1/(2*3^2) = 0.056
LAMBDA_GAMMA = 22.2        # N(1, 0.15): 1/(2*0.15^2) = 22.2
LAMBDA_REG = 10.0          # Per-hold IC -> type-mean (structural, not a prior)

# ── Shared constants ────────────────────────────────────────────────────────

TAU0_PRIOR_CENTER = 18.0   # Prior center for base delay
NADIR_WINDOW_AFTER = 45    # seconds after t_end for loss window

# ── Bounds ──────────────────────────────────────────────────────────────────

# Stage A
TAU0_BOUNDS = (5, 45)          # base delay
GAIN_BOUNDS = (0.5, 2.0)       # b_s gain
P_BOUNDS = (1.0, 5.0)          # power-law curvature exponent
DELTA_BOUNDS = (-20, 20)       # per-hold residual shifts

# Power-law latent
S_MIN_BOUNDS = (30, 100)       # minimum SpO2
V_UP_BOUNDS = (0.0, 3.0)      # recovery slope (%/s)

# Stage B
PERHOLD_BOUNDS = {
    "FL": [(50, 250), (20, 50)],
    "FRC": [(20, 100), (25, 50)],
    "RV": [(10, 80), (30, 55)],
}
PERHOLD_NAMES = ["tau_washout", "paco2_0"]
N_PH = len(PERHOLD_NAMES)
GAMMA_BOUNDS = (0.8, 1.3)     # Tighter than v6.07's (0.8, 3.0)

# Student-t NLL parameters
NU_STUDENT = 4.0   # degrees of freedom
SIGMA_STUDENT = 1.0  # scale


# ── Data loading (from v6.07) ───────────────────────────────────────────────


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


# ── Physiology functions (from v6.07) ───────────────────────────────────────


def corrected_pao2_0(paco2_0, aa):
    """Derive initial PaO2 from PaCO2 and A-a gradient."""
    return max(FIO2_PB_PH2O - paco2_0 / RQ - aa, 1.0)


def pao2_apnea_only(t, pao2_0, pvo2, tau_washout, t_end):
    """PaO2 during apnea only (no recovery branch — tau_reoxy dropped)."""
    return pvo2 + (pao2_0 - pvo2) * np.exp(-t / max(tau_washout, 0.01))


def p50_apnea_only(t, paco2_0, k_co2, t_end):
    """P50 during apnea only (no recovery branch)."""
    paco2 = paco2_0 + k_co2 * t
    return P50_BASE + 0.48 * (paco2 - PACO2_NORMAL)


def odc_severinghaus(pao2, p50_eff, gamma):
    """Severinghaus ODC with Bohr shift and gamma steepness."""
    pao2_virtual = pao2 * (P50_BASE / np.maximum(p50_eff, 0.01))
    pao2_adj = P50_BASE * (np.maximum(pao2_virtual, 0.01) / P50_BASE) ** gamma
    x = np.maximum(pao2_adj, 0.01)
    return 100.0 / (1.0 + 23400.0 / (x**3 + 150.0 * x))


# ── Smooth Discrete Gamma Kernel (from v6.07) ──────────────────────────────


def gamma_kernel_smooth(mean_lag, cv, max_support=120):
    """Smooth discrete gamma kernel via CDF bin integration."""
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
    """Convolve signal with smooth gamma kernel, with preconditioning."""
    h = gamma_kernel_smooth(mean_lag, cv)
    pad_len = len(h)
    padded = np.concatenate([np.full(pad_len, signal[0]), signal])
    convolved = np.convolve(padded, h, mode="full")[:len(padded)]
    return convolved[pad_len:]


# ── NEW: Power-law latent template ──────────────────────────────────────────


def build_powerlaw_latent(t_1hz, t_end, S_start, S_min, v_up, p):
    """Build power-law latent SaO2 curve.

    At t=0: latent = S_start. At t=t_turn: latent = S_min.
    For p > 1, derivative at t=0 is zero (natural plateau), descent accelerates.
    t_turn = t_end (m_h = 0 by design).
    2 free params per hold: S_min, v_up. S_start = B_h (locked). p is global.
    """
    t_turn = max(t_end, 1.0)

    latent = np.empty_like(t_1hz)
    for i, t in enumerate(t_1hz):
        if t <= t_turn:
            frac = t / t_turn
            latent[i] = S_min + (S_start - S_min) * (1.0 - frac ** p)
        else:
            latent[i] = S_min + v_up * (t - t_turn)

    return np.clip(latent, 0.0, 100.0)


# ── NEW: Student-t NLL loss ─────────────────────────────────────────────────


def student_t_nll(residuals, nu=NU_STUDENT, sigma=SIGMA_STUDENT):
    """Student-t negative log-likelihood (robust loss).

    NLL ∝ (nu+1)/2 * sum(log(1 + r^2/(nu*sigma^2)))
    More robust to outliers than Gaussian SSE.
    """
    return (nu + 1.0) / 2.0 * np.sum(np.log1p(residuals**2 / (nu * sigma**2)))


# ── Nadir + loss helpers (from v6.07) ───────────────────────────────────────


def compute_nadir_info(hold):
    """Compute observed nadir timing and SpO2 for a hold.

    Uses local search near t_end with median smoothing to avoid sensor glitches.
    Window: [t_end - 30, t_end + 60] captures all observed nadirs.
    """
    t, spo2, t_end = hold["t"], hold["spo2"], hold["t_end"]
    # Local search: [t_end - 30, t_end + 60]
    local_mask = (t >= t_end - 30) & (t <= t_end + 60)
    if local_mask.sum() >= 5:
        local_spo2 = spo2[local_mask]
        # 5-point median smoothing
        smoothed = np.array([
            np.median(local_spo2[max(0, i - 2):i + 3])
            for i in range(len(local_spo2))
        ])
        idx = np.where(local_mask)[0][np.argmin(smoothed)]
    else:
        idx = np.argmin(spo2)  # fallback
    # Apnea-window nadir: min in [0, t_end+5] for Stage B comparison
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
    """Boolean mask for apnea + first window_after seconds of recovery."""
    return t <= t_end + window_after


def huber_loss(a, delta=8.0):
    """Huber loss: quadratic for |a| <= delta, linear beyond."""
    abs_a = np.abs(a)
    return np.where(abs_a <= delta, 0.5 * a**2, delta * (abs_a - 0.5 * delta))


def nadir_timing_penalty_huber(t, pred, t_nadir_obs, lam=LAMBDA_NADIR, huber_delta=8.0):
    """Huber nadir timing penalty."""
    t_nadir_pred = t[np.argmin(pred)]
    err = t_nadir_pred - t_nadir_obs
    return lam * float(huber_loss(err, delta=huber_delta))


# ── Metrics (from v6.07) ───────────────────────────────────────────────────


def compute_r2(obs, pred):
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def compute_rmse(obs, pred):
    return float(np.sqrt(np.mean((obs - pred) ** 2)))


def is_at_bound(val, lo, hi, tol=1e-3):
    return abs(val - lo) < tol or abs(val - hi) < tol


# ── Covariates (from v6.07) ─────────────────────────────────────────────────


def compute_depth_severity(fit_holds):
    """g_h = max(0, 95 - min(SpO2_obs)) / 10 for each hold."""
    severities = {}
    for h in fit_holds:
        min_spo2 = float(np.min(h["spo2"]))
        severities[h["id"]] = max(0.0, 95.0 - min_spo2) / 10.0
    return severities


def compute_end_slope(fit_holds, window=10):
    """s_h = (SpO2(t_end) - SpO2(t_end - window)) / window.

    Negative when SpO2 is falling at end of apnea.
    """
    slopes = {}
    for h in fit_holds:
        t, spo2 = h["t_apnea"], h["spo2_apnea"]
        mask = t >= (h["t_end"] - window)
        if mask.sum() >= 2:
            t_window = t[mask]
            spo2_window = spo2[mask]
            dt = max(t_window[-1] - t_window[0], 1.0)
            slopes[h["id"]] = (spo2_window[-1] - spo2_window[0]) / dt
        else:
            slopes[h["id"]] = 0.0
    return slopes


# ── Stage A: Sensor-First Calibration ───────────────────────────────────────


def run_stage_a(fit_holds, nadir_info, s_base_values):
    """Stage A: Sensor calibration with power-law latent.

    Parameter layout (18 params for 5 holds):
      Global (3): tau_0, b_s (gain), p (curvature)
      Per-hold deltas (5): delta_h
      Per-hold latent (10): S_min_h, v_up_h (x5)
    """
    n_holds = len(fit_holds)

    # Build bounds
    bounds = [
        TAU0_BOUNDS,      # tau_0
        GAIN_BOUNDS,      # b_s (gain)
        P_BOUNDS,         # p (power-law curvature)
    ]
    n_global = len(bounds)

    # Per-hold deltas
    delta_offset = n_global
    for _ in fit_holds:
        bounds.append(DELTA_BOUNDS)

    # Per-hold latent params: S_min, v_up (S_start locked to B_h)
    latent_offset = delta_offset + n_holds
    n_latent_per_hold = 2
    for _ in fit_holds:
        bounds.append(S_MIN_BOUNDS)
        bounds.append(V_UP_BOUNDS)

    n_total = len(bounds)

    print(f"\n  Stage A: {n_total} params ({n_global} global + {n_holds} delta + "
          f"{n_latent_per_hold}x{n_holds} latent)")
    print(f"  Lag model: eff_lag = max(tau_0 + delta_h, {EFF_LAG_MIN})")
    print(f"  Latent: power-law descent, S_start = B_h (locked), m_h = 0")
    print(f"  Measurement: pred = B_h + b_s * (filtered - B_h)")
    print(f"  Fixed: cv={CV_FIXED}")
    print(f"  Loss: Student-t NLL (nu={NU_STUDENT}, sigma={SIGMA_STUDENT})")
    print(f"  S_base values: {[f'{s_base_values[h['id']]:.1f}' for h in fit_holds]}")

    masks = [nadir_window_mask(h["t"], h["t_end"]) for h in fit_holds]
    nadir_ts = [nadir_info[h["id"]]["t_nadir"] for h in fit_holds]

    # Precompute 1-Hz time grids
    t_grids = []
    for h in fit_holds:
        t_max = h["t"][-1]
        t_1hz = np.arange(0, t_max + 1, 1.0)
        t_grids.append(t_1hz)

    def objective(flat):
        tau_0, b_s, p = flat[:n_global]
        deltas = flat[delta_offset:delta_offset + n_holds]
        total = 0.0

        for i, h in enumerate(fit_holds):
            # Extract latent params
            lp_start = latent_offset + i * n_latent_per_hold
            S_min, v_up = flat[lp_start:lp_start + n_latent_per_hold]

            # S_start locked to baseline
            S_start = s_base_values[h["id"]]

            # Build power-law latent
            t_1hz = t_grids[i]
            latent = build_powerlaw_latent(t_1hz, h["t_end"], S_start, S_min, v_up, p)

            # Apply gamma kernel with effective lag
            eff_lag = max(tau_0 + deltas[i], EFF_LAG_MIN)
            filtered = apply_gamma_kernel(latent, eff_lag, CV_FIXED)

            # Baseline-corrected measurement equation
            B_h = s_base_values[h["id"]]
            pred_1hz = B_h + b_s * (filtered - B_h)
            pred_at_obs = np.interp(h["t"], t_1hz, pred_1hz)
            pred_display = np.clip(pred_at_obs, 0.0, 100.0)

            # Student-t NLL on nadir window (raw predictions)
            m = masks[i]
            residuals = h["spo2"][m] - pred_at_obs[m]
            total += student_t_nll(residuals)

            # Huber timing penalty (clipped for physical argmin)
            total += nadir_timing_penalty_huber(h["t"][m], pred_display[m], nadir_ts[i])

        # ── Priors ──

        # tau_0: LogNormal-like prior centered at 18
        total += LAMBDA_TAU0 * (np.log(max(tau_0, 1.0)) - np.log(TAU0_PRIOR_CENTER)) ** 2

        # Per-hold deltas: StudentT-like shrinkage
        total += LAMBDA_DELTA * np.sum(np.log1p(deltas**2 / 9.0))  # sigma=3
        # Zero-sum constraint
        total += LAMBDA_ZEROSUM * np.sum(deltas) ** 2

        # gain: N(1, 0.1)
        total += LAMBDA_GAIN * (b_s - 1.0) ** 2

        # p: LogNormal(log 2, 0.35)
        total += LAMBDA_P * (np.log(max(p, 0.1)) - np.log(2.0)) ** 2

        return total

    maxiter_a = 4000
    res = differential_evolution(
        objective, bounds, maxiter=maxiter_a, seed=42, tol=1e-10,
        polish=True, popsize=40, mutation=(0.5, 1.5), recombination=0.9,
        callback=make_de_callback("Stage A", maxiter_a),
    )
    print(f"\n  Converged: {res.success}, fun={res.fun:.2f}, nfev={res.nfev}")
    return res.x, res.success


def evaluate_stage_a(flat_a, fit_holds, nadir_info, s_base_values):
    """Evaluate Stage A: sensor model with power-law latent."""
    n_holds = len(fit_holds)
    n_global = 3
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
        filtered = apply_gamma_kernel(latent, eff_lag, CV_FIXED)

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


def extract_frozen_sensor(flat_a, fit_holds):
    """Extract frozen sensor params from Stage A for use in Stage B."""
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


# ── Stage B: Physiology (apnea-only, frozen sensor) ────────────────────────


def predict_v7(t, pvo2, tau_washout, gamma, paco2_0, k_co2, b_s,
               mean_lag, cv, t_end, s_base, shift=0.0):
    """Full sensor pipeline for v7 — apnea-only, baseline-corrected."""
    aa = 0.0
    pao2_0 = corrected_pao2_0(paco2_0, aa)
    pao2 = pao2_apnea_only(t, pao2_0, pvo2, tau_washout, t_end)
    p50 = p50_apnea_only(t, paco2_0, k_co2, t_end)
    sa = odc_severinghaus(pao2, p50, gamma)

    eff_mean_lag = max(mean_lag + shift, EFF_LAG_MIN)
    filtered = apply_gamma_kernel(sa, eff_mean_lag, cv)

    return np.clip(s_base + b_s * (filtered - s_base), 0.0, 100.0)


def run_stage_b(fit_holds, nadir_info, frozen_sensor, s_base_values, label="Stage B"):
    """Stage B: Physiology model with frozen sensor from Stage A.

    Parameter layout (17 params for 5 holds):
      Global physiology (2): k_CO2, gamma (PvO2 fixed at 25)
      Per-hold deltas (5): re-fitted for physiology stage
      Per-hold ICs (10): tau_washout_h, PaCO2_0_h (x5)
    """
    tau_0_frozen = frozen_sensor["tau_0"]
    b_s_frozen = frozen_sensor["b_s"]
    cv_frozen = frozen_sensor["cv"]

    n_holds = len(fit_holds)

    # Physiology shared: k_co2, gamma (pvo2 fixed)
    bounds = [
        (0.02, 0.25),  # k_co2
        GAMMA_BOUNDS,  # gamma [0.8, 1.3]
    ]
    phys_names = ["k_co2", "gamma"]
    n_phys = len(bounds)

    # Per-hold deltas (re-fitted)
    delta_offset = n_phys
    for _ in fit_holds:
        bounds.append(DELTA_BOUNDS)

    # Per-hold ICs
    ic_offset = delta_offset + n_holds
    for h in fit_holds:
        bounds.extend(PERHOLD_BOUNDS[h["type"]])
    n_total = len(bounds)

    type_groups = {}
    for i, h in enumerate(fit_holds):
        type_groups.setdefault(h["type"], []).append(i)

    # Apnea-only loss window: [0, t_end+5]
    apnea_window = 5
    masks = [h["t"] <= h["t_end"] + apnea_window for h in fit_holds]
    weights = [np.where(h["spo2"][m] < 95, 3.0, 1.0) for h, m in zip(fit_holds, masks)]

    print(f"\n  {label}: {n_total} params ({n_phys} physiology + {n_holds} delta + "
          f"{N_PH}x{n_holds} per-hold ICs)")
    print(f"  Frozen sensor: tau_0={tau_0_frozen:.2f}, cv={cv_frozen:.3f}, "
          f"b_s={b_s_frozen:.4f}")
    print(f"  PvO2 fixed at {PVO2_FIXED} mmHg")
    print(f"  gamma bounds: [{GAMMA_BOUNDS[0]}, {GAMMA_BOUNDS[1]}]")
    print(f"  Apnea-only loss: window [0, t_end+{apnea_window}s]")

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

        # Priors
        total += LAMBDA_K_CO2 * (k_co2 - 0.06) ** 2
        total += LAMBDA_GAMMA * (gamma_val - 1.0) ** 2

        # Structural shift penalties
        total += LAMBDA_DELTA * np.sum(np.log1p(deltas**2 / 9.0))
        total += LAMBDA_ZEROSUM * np.sum(deltas) ** 2

        # IC regularization toward type means
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
    return res.x, res.success


def evaluate_stage_b(flat_b, fit_holds, nadir_info,
                     frozen_sensor, s_base_values, all_holds=None):
    """Evaluate Stage B: frozen sensor + apnea-only physiology."""
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

        # Use apnea-window observed nadir for Stage B comparison
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
    """Worker for one LOHO fold. Two-phase: train DE, then DE for held-out.

    Phase 1: Train on train holds (globals + train-only params).
    Phase 2: Optimize 2D held-out latent (S_min, v_up) via DE.
    Phase 2b: 3D held-out (S_min, v_up, delta) for LOHO-Inference.
    """
    leave_idx, fit_holds, nadir_info, s_base_values = args
    left_out = fit_holds[leave_idx]
    train_holds = [h for i, h in enumerate(fit_holds) if i != leave_idx]
    n_train = len(train_holds)

    # ── Phase 1: Train on train holds only ──
    bounds = [TAU0_BOUNDS, GAIN_BOUNDS, P_BOUNDS]
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

    def train_objective(flat):
        tau_0, b_s, p = flat[:n_global]
        deltas = flat[delta_offset:delta_offset + n_train]
        total = 0.0

        for i, h in enumerate(train_holds):
            lp_start = latent_offset + i * 2
            S_min, v_up = flat[lp_start:lp_start + 2]
            S_start = s_base_values[h["id"]]
            t_1hz = t_grids_train[i]
            latent = build_powerlaw_latent(t_1hz, h["t_end"], S_start, S_min, v_up, p)
            eff_lag = max(tau_0 + deltas[i], EFF_LAG_MIN)
            filtered = apply_gamma_kernel(latent, eff_lag, CV_FIXED)
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
        total += LAMBDA_GAIN * (b_s - 1.0) ** 2
        total += LAMBDA_P * (np.log(max(p, 0.1)) - np.log(2.0)) ** 2
        return total

    res = differential_evolution(
        train_objective, bounds, maxiter=2000, seed=42, tol=1e-10,
        polish=True, popsize=25, mutation=(0.5, 1.5), recombination=0.9,
    )

    # ── Phase 2: Optimize held-out latent params via DE (2D) ──
    tau_0_fit, b_s_fit, p_fit = res.x[:n_global]

    # No delta for held-out hold (that's the test)
    eff_lag_ho = max(tau_0_fit, EFF_LAG_MIN)

    t_1hz_ho = np.arange(0, left_out["t"][-1] + 1, 1.0)
    mask_ho = nadir_window_mask(left_out["t"], left_out["t_end"])
    nadir_t_ho = nadir_info[left_out["id"]]["t_nadir"]
    S_start_ho = s_base_values[left_out["id"]]
    B_h_ho = S_start_ho

    # 2D optimization: [S_min, v_up]
    ho_bounds = [S_MIN_BOUNDS, V_UP_BOUNDS]

    def ho_objective(x):
        S_min, v_up = x
        latent = build_powerlaw_latent(t_1hz_ho, left_out["t_end"], S_start_ho, S_min, v_up, p_fit)
        filtered = apply_gamma_kernel(latent, eff_lag_ho, CV_FIXED)
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
    best_loss = ho_res.fun
    ho_S_min, ho_v_up = ho_res.x

    # Predict held-out hold with optimized latent params (LOHO-Global)
    latent_ho = build_powerlaw_latent(
        t_1hz_ho, left_out["t_end"], S_start_ho, ho_S_min, ho_v_up, p_fit)
    filtered_ho = apply_gamma_kernel(latent_ho, eff_lag_ho, CV_FIXED)
    pred_ho_1hz = B_h_ho + b_s_fit * (filtered_ho - B_h_ho)
    pred_ho = np.interp(left_out["t"], t_1hz_ho, pred_ho_1hz)
    pred_ho = np.clip(pred_ho, 0.0, 100.0)

    r2_ho = compute_r2(left_out["spo2"][mask_ho], pred_ho[mask_ho]) if mask_ho.sum() > 3 else None
    rmse_ho = compute_rmse(left_out["spo2"][mask_ho], pred_ho[mask_ho]) if mask_ho.sum() > 3 else None

    t_nadir_obs = nadir_info[left_out["id"]]["t_nadir"]
    t_nadir_pred = left_out["t"][np.argmin(pred_ho)]
    nadir_err = t_nadir_pred - t_nadir_obs

    # ── Phase 2b: LOHO-Inference (3D: S_min, v_up, delta_ho) ──
    ho_bounds_inf = [S_MIN_BOUNDS, V_UP_BOUNDS, DELTA_BOUNDS]

    def ho_inference_objective(x):
        S_min, v_up, delta_ho = x
        latent = build_powerlaw_latent(t_1hz_ho, left_out["t_end"], S_start_ho, S_min, v_up, p_fit)
        eff_lag_inf = max(tau_0_fit + delta_ho, EFF_LAG_MIN)
        filtered = apply_gamma_kernel(latent, eff_lag_inf, CV_FIXED)
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
    best_loss_inf = ho_res_inf.fun
    inf_S_min, inf_v_up, inf_delta = ho_res_inf.x
    eff_lag_inf = max(tau_0_fit + inf_delta, EFF_LAG_MIN)

    latent_inf = build_powerlaw_latent(
        t_1hz_ho, left_out["t_end"], S_start_ho, inf_S_min, inf_v_up, p_fit)
    filtered_inf = apply_gamma_kernel(latent_inf, eff_lag_inf, CV_FIXED)
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
        "ho_loss": best_loss,
        # LOHO-Inference results
        "r2_inf": r2_inf,
        "rmse_inf": rmse_inf,
        "nadir_err_inf": nadir_err_inf,
        "eff_lag_inf": eff_lag_inf,
        "inf_delta": inf_delta,
        "inf_loss": best_loss_inf,
    }


def run_stage_a_loho(fit_holds, nadir_info, s_base_values):
    """Leave-one-hold-out CV for Stage A - parallelized across folds."""
    n_holds = len(fit_holds)
    print(f"\n  Running {n_holds} LOHO folds in parallel ({N_WORKERS} workers)...")

    args_list = [(i, fit_holds, nadir_info, s_base_values)
                 for i in range(n_holds)]

    with ProcessPoolExecutor(max_workers=min(N_WORKERS, n_holds)) as pool:
        loho_results = list(pool.map(_loho_worker, args_list))

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


# ── Time-split for Stage B ──────────────────────────────────────────────────


def run_stage_b_time_split(fit_holds, nadir_info,
                           frozen_sensor, s_base_values, split_frac=0.6):
    """Time-split validation: fit on early apnea, predict late apnea.

    For each hold: fit [0, split_frac*t_end], predict [split_frac*t_end, t_end+5].
    """
    tau_0_frozen = frozen_sensor["tau_0"]
    b_s_frozen = frozen_sensor["b_s"]
    cv_frozen = frozen_sensor["cv"]

    n_holds = len(fit_holds)

    type_groups = {}
    for i, h in enumerate(fit_holds):
        type_groups.setdefault(h["type"], []).append(i)

    # Fit on early portion of each hold (threshold-based: first SpO2 < 95 or fixed fraction)
    split_masks = []
    split_points = []
    for h in fit_holds:
        below_95 = np.where(h["spo2"] < 95)[0]
        t_threshold = h["t"][below_95[0]] if len(below_95) > 0 else float("inf")
        t_split = min(t_threshold, split_frac * h["t_end"])
        split_points.append(t_split)
        split_masks.append(h["t"] <= t_split)

    bounds = [
        (0.02, 0.25),  # k_co2
        GAMMA_BOUNDS,  # gamma
    ]
    n_phys = len(bounds)

    delta_offset = n_phys
    for _ in fit_holds:
        bounds.append(DELTA_BOUNDS)

    ic_offset = delta_offset + n_holds
    for h in fit_holds:
        bounds.extend(PERHOLD_BOUNDS[h["type"]])

    print(f"\n  Time-split: fitting on [0, min(SpO2<95, {split_frac}*t_end)], "
          f"predicting rest to t_end+5")
    for i, h in enumerate(fit_holds):
        print(f"    {h['type']}#{h['id']}: t_split={split_points[i]:.0f}s "
              f"(t_end={h['t_end']:.0f}s)")

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
            m = split_masks[i]
            weights = np.where(h["spo2"][m] < 95, 3.0, 1.0)
            total += np.sum(weights * (h["spo2"][m] - pred[m]) ** 2)

            total += LAMBDA_PACO2 * (paco2_0 - 40.0) ** 2

        total += LAMBDA_K_CO2 * (k_co2 - 0.06) ** 2
        total += LAMBDA_GAMMA * (gamma_val - 1.0) ** 2
        total += LAMBDA_DELTA * np.sum(np.log1p(deltas**2 / 9.0))
        total += LAMBDA_ZEROSUM * np.sum(deltas) ** 2

        for ht, indices in type_groups.items():
            if len(indices) < 2:
                continue
            for p_off in range(N_PH):
                values = [flat[ic_offset + idx * N_PH + p_off] for idx in indices]
                mean_val = np.mean(values)
                total += LAMBDA_REG * sum((v - mean_val) ** 2 for v in values)

        return total

    maxiter_ts = 2000
    res = differential_evolution(
        objective, bounds, maxiter=maxiter_ts, seed=42, tol=1e-10,
        polish=True, popsize=25, mutation=(0.5, 1.5), recombination=0.9,
        callback=make_de_callback("TimeSplit", maxiter_ts),
    )
    print(f"\n  Converged: {res.success}, fun={res.fun:.2f}")

    # Evaluate on prediction window
    k_co2, gamma_val = res.x[:n_phys]
    deltas = res.x[delta_offset:delta_offset + n_holds]
    ts_results = []

    for i, h in enumerate(fit_holds):
        ph_offset = ic_offset + i * N_PH
        tau_washout, paco2_0 = res.x[ph_offset:ph_offset + N_PH]

        pred = predict_v7(
            h["t"], PVO2_FIXED, tau_washout, gamma_val,
            paco2_0, k_co2, b_s_frozen,
            tau_0_frozen, cv_frozen, h["t_end"],
            s_base=s_base_values[h["id"]],
            shift=deltas[i],
        )

        # Predict window: [t_split, t_end+5]
        t_split = split_points[i]
        pred_mask = (h["t"] > t_split) & (h["t"] <= h["t_end"] + 5)
        fit_mask = split_masks[i]

        r2_fit = compute_r2(h["spo2"][fit_mask], pred[fit_mask]) if fit_mask.sum() > 3 else None

        # Skip prediction metrics if fit segment has <3% SpO2 drop (insufficient signal)
        fit_drop = h["spo2"][fit_mask].max() - h["spo2"][fit_mask].min() if fit_mask.sum() > 0 else 0
        if fit_drop < 3.0:
            r2_pred = None  # insufficient signal
            rmse_pred = None
        else:
            r2_pred = compute_r2(h["spo2"][pred_mask], pred[pred_mask]) if pred_mask.sum() > 3 else None
            rmse_pred = compute_rmse(h["spo2"][pred_mask], pred[pred_mask]) if pred_mask.sum() > 3 else None

        ts_results.append({
            "hold_id": h["id"],
            "hold_type": h["type"],
            "r2_fit": r2_fit,
            "r2_pred": r2_pred,
            "rmse_pred": rmse_pred,
            "pred": pred,
            "fit_drop": fit_drop,
        })
        r2_fit_str = f"{r2_fit:.4f}" if r2_fit is not None else "N/A"
        r2_pred_str = f"{r2_pred:.4f}" if r2_pred is not None else f"N/A (drop={fit_drop:.1f}%)"
        rmse_str = f"{rmse_pred:.2f}" if rmse_pred is not None else "N/A"
        print(f"    {h['type']}#{h['id']}: R2_fit={r2_fit_str}, R2_pred={r2_pred_str}, "
              f"RMSE_pred={rmse_str}")

    return ts_results


# ── Profile likelihood ──────────────────────────────────────────────────────


def _profile_tau_worker(args):
    """Worker for one tau_0 profile point. Top-level for pickling."""
    tau_fixed, fit_holds, nadir_info, s_base_values = args
    n_holds = len(fit_holds)
    masks = [nadir_window_mask(h["t"], h["t_end"]) for h in fit_holds]
    nadir_ts = [nadir_info[h["id"]]["t_nadir"] for h in fit_holds]
    t_grids = [np.arange(0, h["t"][-1] + 1, 1.0) for h in fit_holds]

    bounds = [
        (tau_fixed - 0.01, tau_fixed + 0.01),
        GAIN_BOUNDS, P_BOUNDS,
    ]
    n_global = len(bounds)
    delta_offset = n_global
    for _ in fit_holds:
        bounds.append(DELTA_BOUNDS)
    latent_offset = delta_offset + n_holds
    for _ in fit_holds:
        bounds.extend([S_MIN_BOUNDS, V_UP_BOUNDS])

    def objective(flat):
        tau_0, b_s, p = flat[:n_global]
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
        total += LAMBDA_GAIN * (b_s - 1.0) ** 2
        total += LAMBDA_P * (np.log(max(p, 0.1)) - np.log(2.0)) ** 2
        return total

    res = differential_evolution(
        objective, bounds, maxiter=2000, seed=42, tol=1e-10,
        polish=True, popsize=25, mutation=(0.5, 1.5), recombination=0.9,
    )
    return tau_fixed, {"flat": res.x, "loss": res.fun, "success": res.success}


def _profile_p_worker(args):
    """Worker for one p profile point. Top-level for pickling."""
    p_fixed, fit_holds, nadir_info, s_base_values = args
    n_holds = len(fit_holds)
    masks = [nadir_window_mask(h["t"], h["t_end"]) for h in fit_holds]
    nadir_ts = [nadir_info[h["id"]]["t_nadir"] for h in fit_holds]
    t_grids = [np.arange(0, h["t"][-1] + 1, 1.0) for h in fit_holds]

    bounds = [
        TAU0_BOUNDS, GAIN_BOUNDS,
        (p_fixed - 0.01, p_fixed + 0.01),
    ]
    n_global = len(bounds)
    delta_offset = n_global
    for _ in fit_holds:
        bounds.append(DELTA_BOUNDS)
    latent_offset = delta_offset + n_holds
    for _ in fit_holds:
        bounds.extend([S_MIN_BOUNDS, V_UP_BOUNDS])

    def objective(flat):
        tau_0, b_s, p = flat[:n_global]
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
        total += LAMBDA_GAIN * (b_s - 1.0) ** 2
        total += LAMBDA_P * (np.log(max(p, 0.1)) - np.log(2.0)) ** 2
        return total

    res = differential_evolution(
        objective, bounds, maxiter=2000, seed=42, tol=1e-10,
        polish=True, popsize=25, mutation=(0.5, 1.5), recombination=0.9,
    )
    return p_fixed, {"flat": res.x, "loss": res.fun, "success": res.success}


def run_profile_likelihood(fit_holds, nadir_info, s_base_values,
                           tau_0_values=None, p_values=None):
    """Fix tau_0 or p at each value, reoptimize - parallelized."""
    if tau_0_values is None:
        tau_0_values = [5, 8, 10, 12, 14, 16, 18, 20, 22, 25, 30, 35, 40, 45]
    if p_values is None:
        p_values = [1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

    # tau_0 profile
    print(f"\n  Running {len(tau_0_values)} tau_0 profile points in parallel ({N_WORKERS} workers)...")
    args_tau = [(tv, fit_holds, nadir_info, s_base_values) for tv in tau_0_values]
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        raw_tau = list(pool.map(_profile_tau_worker, args_tau))

    tau_results = {}
    for tau_val, result in raw_tau:
        tau_results[tau_val] = result
        b_s = result["flat"][1]
        p = result["flat"][2]
        print(f"    tau_0={tau_val:5.1f}: loss={result['loss']:.2f}, "
              f"b_s={b_s:.3f}, p={p:.3f}", flush=True)

    # p profile
    print(f"\n  Running {len(p_values)} p profile points in parallel ({N_WORKERS} workers)...")
    args_p = [(pv, fit_holds, nadir_info, s_base_values) for pv in p_values]
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        raw_p = list(pool.map(_profile_p_worker, args_p))

    p_results = {}
    for p_val, result in raw_p:
        p_results[p_val] = result
        tau_0 = result["flat"][0]
        b_s = result["flat"][1]
        print(f"    p={p_val:5.2f}: loss={result['loss']:.2f}, "
              f"tau_0={tau_0:.3f}, b_s={b_s:.3f}", flush=True)

    return tau_results, p_results


# ── Sensitivity analysis ───────────────────────────────────────────────────


def _sensitivity_worker(args):
    """Worker for one Stage B at a given tau_0. Top-level for pickling."""
    tau_val, fit_holds, nadir_info, frozen_sensor_template, s_base_values = args
    frozen = dict(frozen_sensor_template)
    frozen["tau_0"] = tau_val
    flat_b, success = run_stage_b(fit_holds, nadir_info, frozen, s_base_values, label="Sensitivity")
    return tau_val, flat_b


def compute_sensitivity(profile_results, fit_holds, nadir_info,
                        frozen_sensor_template, s_base_values):
    """Run Stage B at multiple tau_0 values - parallelized."""
    losses = {k: v["loss"] for k, v in profile_results.items()}
    sorted_tau = sorted(losses, key=losses.get)
    center_idx = list(sorted(profile_results.keys())).index(sorted_tau[0])
    all_taus = sorted(profile_results.keys())
    start = max(0, center_idx - 2)
    end = min(len(all_taus), start + 5)
    tau_values = all_taus[start:end]

    print(f"\n  Sensitivity: running Stage B at tau_0 = {tau_values} "
          f"in parallel ({N_WORKERS} workers)...")

    args_list = [(tv, fit_holds, nadir_info,
                  frozen_sensor_template, s_base_values) for tv in tau_values]

    with ProcessPoolExecutor(max_workers=min(N_WORKERS, len(tau_values))) as pool:
        raw_results = list(pool.map(_sensitivity_worker, args_list))

    stage_b_results = {tv: fb for tv, fb in raw_results}

    phys_names = ["k_co2", "gamma"]
    sensitivities = {}
    tau_list = sorted(stage_b_results.keys())
    if len(tau_list) >= 2:
        for pi, pname in enumerate(phys_names):
            vals = [stage_b_results[t][pi] for t in tau_list]
            derivs = np.gradient(vals, tau_list)
            sensitivities[pname] = {
                "values": dict(zip(tau_list, vals)),
                "derivs": dict(zip(tau_list, derivs)),
                "max_deriv": float(np.max(np.abs(derivs))),
            }
            print(f"    d({pname})/d(tau_0): max |deriv| = {sensitivities[pname]['max_deriv']:.4f}")

    return sensitivities


# ── Sponge diagnostics ─────────────────────────────────────────────────────


def sponge_diagnostics(flat, bounds, prior_sigmas, param_names, label=""):
    """Sponge diagnostic report: params at bounds, contraction ratios.

    prior_sigmas: dict mapping param name -> prior sigma
    Contraction ratio = posterior_range / prior_range
    """
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
            posterior_width = 2 * sigma  # approximate
            contraction = min(posterior_width / prior_range, 1.0) if prior_range > 0 else 1.0
            contr_str = f"{contraction:.3f}"
        else:
            contr_str = "N/A"
        print(f"  {name:<15s} | {val:10.4f} | [{lo:>6.2f}, {hi:>6.2f}] | "
              f"{'YES' if at_b else '   ':>7s} | {contr_str:>11s}")

    print(f"\n  Total at bound: {at_bound_count}/{len(param_names)}")
    return at_bound_count


# ── Output helpers ──────────────────────────────────────────────────────────


def print_comparison_table(all_results, variant_names):
    """Print per-hold R2 + nadir error comparison across variants."""
    print(f"\n{'='*120}")
    print("PER-HOLD COMPARISON")
    print(f"{'='*120}")

    header = f"  {'Hold':<15s}"
    for vn in variant_names:
        header += f" | {'R2a':>6s} {'R2n':>6s} {'NadirErr':>8s}"
    print(header)
    sub = f"  {'':15s}"
    for vn in variant_names:
        sub += f"   {vn:>22s}"
    print(sub)
    print(f"  {'-'*15}" + f"-+-{'-'*22}" * len(variant_names))

    by_hold = {}
    for r in all_results:
        by_hold.setdefault(r["hold_id"], {})[r["variant"]] = r

    for hid in sorted(by_hold.keys()):
        variants = by_hold[hid]
        first = next(iter(variants.values()))
        tag = " (excl)" if first["is_excluded"] else ""
        if first.get("is_qc_flagged"):
            tag += " [QC]"
        label = f"{first['hold_type']}#{hid}{tag}"
        row = f"  {label:<15s}"
        for vn in variant_names:
            r = variants.get(vn)
            if r:
                r2a = f"{r['r2_apnea']:.4f}" if r["r2_apnea"] is not None else "N/A"
                r2n = f"{r['r2_nadir']:.4f}" if r["r2_nadir"] is not None else "N/A"
                nerr = f"{r['nadir_err']:+.1f}s"
                row += f" | {r2a:>6s} {r2n:>6s} {nerr:>8s}"
            else:
                row += f" |    N/A    N/A      N/A"
        print(row)

    print(f"\n  {'Avg (all fit)':<15s}", end="")
    for vn in variant_names:
        r2as, r2ns, nerrs = [], [], []
        for hid, variants in sorted(by_hold.items()):
            r = variants.get(vn)
            if r and not r["is_excluded"]:
                if r["r2_apnea"] is not None:
                    r2as.append(r["r2_apnea"])
                if r["r2_nadir"] is not None:
                    r2ns.append(r["r2_nadir"])
                nerrs.append(abs(r["nadir_err"]))
        avg_a = np.mean(r2as) if r2as else float("nan")
        avg_n = np.mean(r2ns) if r2ns else float("nan")
        avg_ne = np.mean(nerrs) if nerrs else float("nan")
        print(f" | {avg_a:>6.4f} {avg_n:>6.4f} {avg_ne:>7.1f}s", end="")
    print()

    # QC-passing averages
    print(f"  {'Avg (QC-pass)':<15s}", end="")
    for vn in variant_names:
        r2as, r2ns, nerrs = [], [], []
        for hid, variants in sorted(by_hold.items()):
            r = variants.get(vn)
            if r and not r["is_excluded"] and not r.get("is_qc_flagged"):
                if r["r2_apnea"] is not None:
                    r2as.append(r["r2_apnea"])
                if r["r2_nadir"] is not None:
                    r2ns.append(r["r2_nadir"])
                nerrs.append(abs(r["nadir_err"]))
        avg_a = np.mean(r2as) if r2as else float("nan")
        avg_n = np.mean(r2ns) if r2ns else float("nan")
        avg_ne = np.mean(nerrs) if nerrs else float("nan")
        print(f" | {avg_a:>6.4f} {avg_n:>6.4f} {avg_ne:>7.1f}s", end="")
    print()


# ── Plots ────────────────────────────────────────────────────────────────────


def plot_stage_a_detail(eval_a, latent_curves, all_holds, nadir_info, output_path):
    """Per-hold detail plots: latent + kernel + prediction."""
    holds_dict = {h["id"]: h for h in all_holds}
    n = len(eval_a)

    fig, axes = plt.subplots(n, 1, figsize=(16, 4.5 * n), squeeze=False)

    for idx, (res, lc) in enumerate(zip(eval_a, latent_curves)):
        ax = axes[idx, 0]
        h = holds_dict[res["hold_id"]]
        ni = nadir_info[res["hold_id"]]

        # Observed
        ax.plot(h["t"], h["spo2"], "k.", markersize=2, alpha=0.5, label="Observed")
        ax.axvline(x=h["t_end"], color="red", linestyle="--", alpha=0.5, label="Apnea end")
        ax.plot(ni["t_nadir"], ni["spo2_nadir"], "r*", markersize=12, zorder=5,
                label=f"Obs nadir (t={ni['t_nadir']:.0f}s)")

        # Latent (pre-kernel)
        ax.plot(lc["t_1hz"], lc["latent"], color="#ff7f0e", linewidth=2, alpha=0.7,
                label=f"Latent (S_start={lc['S_start']:.1f}, S_min={lc['S_min']:.1f})")

        # Predicted (post-kernel)
        ax.plot(h["t"], res["pred_full"], color="#1f77b4", linewidth=2, alpha=0.8,
                label=f"Predicted (R2n={res['r2_nadir']:.3f}, err={res['nadir_err']:+.1f}s)")

        # Mark turning point (m_h = 0, so t_turn = t_end)
        ax.axvline(x=h["t_end"], color="purple", linestyle=":", alpha=0.5,
                   label="Turning pt (t_end)")

        ax.set_title(f"Stage A: {h['type']}#{h['id']} (eff_lag={res['effective_lag']:.1f}s)",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("SpO2 (%)")
        ax.set_ylim(30, 105)
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(True, alpha=0.3)

    fig.suptitle("v7.05 Stage A: Power-Law Latent + Gamma Kernel",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nStage A detail plot saved to {output_path}")


def plot_stage_b_detail(eval_b, all_holds, nadir_info, frozen_sensor, output_path):
    """Per-hold detail plots for Stage B physiology."""
    holds_dict = {h["id"]: h for h in all_holds}
    fitted = [r for r in eval_b if not r["is_excluded"]]
    n = len(fitted)

    fig, axes = plt.subplots(n, 1, figsize=(16, 4.5 * n), squeeze=False)

    for idx, res in enumerate(fitted):
        ax = axes[idx, 0]
        h = holds_dict[res["hold_id"]]
        ni = nadir_info[res["hold_id"]]

        ax.plot(h["t"], h["spo2"], "k.", markersize=2, alpha=0.5, label="Observed")
        ax.axvline(x=h["t_end"], color="red", linestyle="--", alpha=0.5, label="Apnea end")
        ax.axvline(x=h["t_end"] + 5, color="gray", linestyle=":", alpha=0.3,
                   label="Fit window end (+5s)")
        ax.plot(ni["t_nadir"], ni["spo2_nadir"], "r*", markersize=12, zorder=5)

        r2_str = ""
        if res["r2_apnea"] is not None:
            r2_str = f"R2a={res['r2_apnea']:.3f}"
        r2_str += f", err={res['nadir_err']:+.1f}s"
        ax.plot(h["t"], res["pred_full"], color="#2ca02c", linewidth=2, alpha=0.8,
                label=f"Stage B ({r2_str})")

        ax.set_title(f"Stage B: {h['type']}#{h['id']}",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("SpO2 (%)")
        ax.set_ylim(30, 105)
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(True, alpha=0.3)

    fig.suptitle("v7.05 Stage B: Apnea-Only Physiology (frozen sensor)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Stage B detail plot saved to {output_path}")


def plot_loho_summary(loho_results, output_path):
    """LOHO summary: bar chart of R2 and timing error per held-out hold."""
    n = len(loho_results)
    labels = [f"{r['hold_type']}#{r['hold_id']}" for r in loho_results]
    r2s = [r["r2"] if r["r2"] is not None else 0 for r in loho_results]
    nerrs = [r["nadir_err"] for r in loho_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(labels, r2s, color="#1f77b4", alpha=0.8)
    ax1.set_ylabel("R2 (nadir window)")
    ax1.set_title("LOHO: Held-Out R2", fontweight="bold")
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    colors = ["#2ca02c" if abs(e) < 5 else "#ff7f0e" if abs(e) < 10 else "#d62728"
              for e in nerrs]
    ax2.bar(labels, nerrs, color=colors, alpha=0.8)
    ax2.set_ylabel("Nadir timing error (s)")
    ax2.set_title("LOHO: Held-Out Timing Error", fontweight="bold")
    ax2.axhline(y=0, color="k", linewidth=0.5)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("v7.05 Stage A: Leave-One-Hold-Out Cross-Validation",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"LOHO summary plot saved to {output_path}")


def plot_profile_likelihood(tau_results, p_results, output_path):
    """Profile likelihood: 2x3 grid for tau_0 and p profiles."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 0: tau_0 profile
    tau_list = sorted(tau_results.keys())
    tau_configs = [
        (None, "loss", "tau_0 Profile: Loss"),
        (1, "b_s", "tau_0 Profile: b_s"),
        (2, "p", "tau_0 Profile: p"),
    ]
    for i, (pidx, name, title) in enumerate(tau_configs):
        ax = axes[0, i]
        if pidx is not None:
            vals = [tau_results[t]["flat"][pidx] for t in tau_list]
        else:
            vals = [tau_results[t]["loss"] for t in tau_list]
        ax.plot(tau_list, vals, "o-", color="#1f77b4", linewidth=2, markersize=6)
        ax.set_xlabel("tau_0 (fixed, s)")
        ax.set_ylabel(name)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)

    # Mark minimum on tau_0 loss plot
    losses_tau = [tau_results[t]["loss"] for t in tau_list]
    min_idx = np.argmin(losses_tau)
    axes[0, 0].axvline(x=tau_list[min_idx], color="red", linestyle="--", alpha=0.5)
    axes[0, 0].annotate(f"min at {tau_list[min_idx]:.0f}s",
                        xy=(tau_list[min_idx], losses_tau[min_idx]),
                        xytext=(10, 10), textcoords="offset points",
                        fontsize=9, color="red")

    # Row 1: p profile
    p_list = sorted(p_results.keys())
    p_configs = [
        (None, "loss", "p Profile: Loss"),
        (0, "tau_0", "p Profile: tau_0"),
        (1, "b_s", "p Profile: b_s"),
    ]
    for i, (pidx, name, title) in enumerate(p_configs):
        ax = axes[1, i]
        if pidx is not None:
            vals = [p_results[pv]["flat"][pidx] for pv in p_list]
        else:
            vals = [p_results[pv]["loss"] for pv in p_list]
        ax.plot(p_list, vals, "s-", color="#ff7f0e", linewidth=2, markersize=6)
        ax.set_xlabel("p (fixed)")
        ax.set_ylabel(name)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)

    # Mark minimum on p loss plot
    losses_p = [p_results[pv]["loss"] for pv in p_list]
    min_idx_p = np.argmin(losses_p)
    axes[1, 0].axvline(x=p_list[min_idx_p], color="red", linestyle="--", alpha=0.5)
    axes[1, 0].annotate(f"min at p={p_list[min_idx_p]:.1f}",
                        xy=(p_list[min_idx_p], losses_p[min_idx_p]),
                        xytext=(10, 10), textcoords="offset points",
                        fontsize=9, color="red")

    fig.suptitle("v7.05: Profile Likelihood (tau_0 and p)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Profile likelihood plot saved to {output_path}")


def plot_sensitivity(sensitivities, output_path):
    """Sensitivity bar chart: d(physiology params)/d(tau_0)."""
    if not sensitivities:
        print("  No sensitivity data to plot.")
        return

    phys_names = list(sensitivities.keys())
    max_derivs = [sensitivities[p]["max_deriv"] for p in phys_names]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(phys_names, max_derivs, color="#ff7f0e", alpha=0.8)
    ax.set_ylabel("max |d(param)/d(tau_0)|")
    ax.set_title("Sensitivity of Physiology Params to tau_0", fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Sensitivity plot saved to {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    print("=" * 120)
    print("v7.05: Power-Law Descent + Baseline-Locked Sensor Model")
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

    # ── Compute nadir info ───────────────────────────────────────────────────
    print(f"\n{'='*120}")
    print("OBSERVED NADIR TIMING")
    print(f"{'='*120}")

    nadir_info = {}
    for h in all_holds:
        ni = compute_nadir_info(h)
        nadir_info[h["id"]] = ni
        tag = " [EXCLUDED]" if h["id"] in EXCLUDED_IDS else ""
        loc = "recovery" if ni["in_recovery"] else "apnea"
        print(f"  {h['type']}#{h['id']}{tag}: nadir at t={ni['t_nadir']:.0f}s "
              f"(SpO2={ni['spo2_nadir']:.0f}%, {loc}, "
              f"delay_from_end={ni['delay_from_end']:+.0f}s), "
              f"apnea-window nadir at t={ni['t_nadir_apnea']:.0f}s "
              f"(SpO2={ni['spo2_nadir_apnea']:.0f}%)")

    # ── Compute baselines (S_start locked to B_h) ──────────────────────────
    s_base_values = {}
    for h in all_holds:
        plateau_mask = h["t"] <= 20
        if plateau_mask.sum() > 0:
            s_base_values[h["id"]] = float(np.median(h["spo2"][plateau_mask]))
        else:
            s_base_values[h["id"]] = float(h["spo2"][0])

    print(f"\n  Baselines (S_start = B_h = median SpO2[t<=20], locked):")
    for h in fit_holds:
        print(f"    {h['type']}#{h['id']}: B_h={s_base_values[h['id']]:.1f}")

    # ── QC flags (out-of-regime detection) ────────────────────────────────────
    qc_flags = {}
    for h in all_holds:
        ni = nadir_info[h["id"]]
        qc_flags[h["id"]] = ni["t_nadir_apnea"] < h["t_end"] - 2
    flagged = [h for h in fit_holds if qc_flags[h["id"]]]
    if flagged:
        print(f"\n  QC-flagged holds (apnea-window nadir before t_end-2s):")
        for h in flagged:
            ni = nadir_info[h["id"]]
            print(f"    {h['type']}#{h['id']}: nadir at t={ni['t_nadir_apnea']:.0f}s, "
                  f"t_end={h['t_end']:.0f}s (OUT-OF-REGIME)")
    else:
        print(f"\n  No QC-flagged holds.")

    n_holds = len(fit_holds)

    # ── Summary of changes from v7.04 ────────────────────────────────────────
    print(f"\nKey changes from v7.04:")
    print(f"  1. Power-law descent (replaces linear): p global, curvature param")
    print(f"  2. S_start locked to B_h (removed from parameters)")
    print(f"  3. r_offset removed, measurement: pred = B_h + b_s*(filtered - B_h)")
    print(f"  4. m_h removed (fixed at 0), t_turn = t_end")
    print(f"  5. EFF_LAG_MIN = {EFF_LAG_MIN}s floor")
    print(f"  6. PvO2 fixed at {PVO2_FIXED} mmHg in Stage B")
    print(f"  7. p profile likelihood added")

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE A: Sensor-First Calibration (18 params)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("STAGE A: Sensor-First Calibration (18 params)")
    print(f"{'='*120}")

    flat_a, conv_a = run_stage_a(fit_holds, nadir_info, s_base_values)

    # Print Stage A results
    n_global = 3
    global_names = ["tau_0", "b_s", "p"]
    global_bounds = [TAU0_BOUNDS, GAIN_BOUNDS, P_BOUNDS]
    print(f"\n  Stage A global params:")
    for name, val, (lo, hi) in zip(global_names, flat_a[:n_global], global_bounds):
        flag = " ** AT BOUND **" if is_at_bound(val, lo, hi) else ""
        print(f"    {name:>12s} = {val:8.4f}  [{lo:>6.2f}, {hi:>6.2f}]{flag}")

    tau_0, b_s_a, p_a = flat_a[:n_global]
    k_val = 1.0 / (CV_FIXED * CV_FIXED)
    print(f"\n    Kernel: k={k_val:.2f}, mean={tau_0:.1f}s, std={tau_0*CV_FIXED:.1f}s "
          f"(cv={CV_FIXED} FIXED)")

    delta_offset_a = n_global
    latent_offset_a = delta_offset_a + n_holds
    deltas_a = flat_a[delta_offset_a:delta_offset_a + n_holds]

    print(f"\n  Stage A -- Per-hold breakdown (base tau_0={tau_0:.2f}, p={p_a:.3f}):")
    for i, h in enumerate(fit_holds):
        residual = deltas_a[i]
        eff = max(tau_0 + residual, EFF_LAG_MIN)
        bound_str = " *BOUND*" if is_at_bound(residual, *DELTA_BOUNDS) else ""
        eff_floor_str = " *FLOOR*" if eff <= EFF_LAG_MIN + 0.01 else ""
        qc_str = " [QC-FLAGGED]" if qc_flags[h["id"]] else ""
        lp_start = latent_offset_a + i * 2
        S_min, v_up = flat_a[lp_start:lp_start + 2]
        print(f"    {h['type']}#{h['id']}: "
              f"delta={residual:+6.2f}, eff_lag={eff:6.2f}, "
              f"B_h={s_base_values[h['id']]:.1f}, "
              f"S_min={S_min:.1f}, v_up={v_up:.2f}{bound_str}{eff_floor_str}{qc_str}")

    eval_a, latent_curves_a = evaluate_stage_a(flat_a, fit_holds, nadir_info, s_base_values)

    # Annotate eval results with QC flags
    for r in eval_a:
        r["is_qc_flagged"] = qc_flags.get(r["hold_id"], False)

    # ── Saturation diagnostic ─────────────────────────────────────────────────
    print(f"\n  Saturation Diagnostic (pred_raw > 100):")
    total_sat = 0
    total_pts = 0
    for i, h in enumerate(fit_holds):
        lp_start = latent_offset_a + i * 2
        S_min, v_up = flat_a[lp_start:lp_start + 2]
        S_start = s_base_values[h["id"]]
        t_1hz = np.arange(0, h["t"][-1] + 1, 1.0)
        latent = build_powerlaw_latent(t_1hz, h["t_end"], S_start, S_min, v_up, p_a)
        eff_lag = max(tau_0 + deltas_a[i], EFF_LAG_MIN)
        filtered = apply_gamma_kernel(latent, eff_lag, CV_FIXED)
        B_h = s_base_values[h["id"]]
        pred_raw = B_h + b_s_a * (filtered - B_h)
        n_sat = int(np.sum(pred_raw > 100))
        n_total = len(pred_raw)
        pct = 100.0 * n_sat / n_total if n_total > 0 else 0
        total_sat += n_sat
        total_pts += n_total
        max_raw = float(np.max(pred_raw))
        print(f"    {h['type']}#{h['id']}: {n_sat}/{n_total} ({pct:.1f}%) saturated, "
              f"max pred_raw={max_raw:.1f}")
    pct_total = 100.0 * total_sat / total_pts if total_pts > 0 else 0
    print(f"    Overall: {total_sat}/{total_pts} ({pct_total:.1f}%) saturated")

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE B: Physiology (17 params, frozen sensor)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("STAGE B: Apnea-Only Physiology (17 params, frozen sensor)")
    print(f"{'='*120}")

    frozen_sensor = extract_frozen_sensor(flat_a, fit_holds)

    flat_b, conv_b = run_stage_b(fit_holds, nadir_info, frozen_sensor, s_base_values,
                                  label="Stage B (frozen-lag)")

    def print_stage_b_results(flat_b, label="Stage B"):
        phys_names_b = ["k_co2", "gamma"]
        phys_bounds_b = [(0.02, 0.25), GAMMA_BOUNDS]
        print(f"\n  {label} physiology params (sensor frozen from Stage A):")
        print(f"    {'pvo2':>12s} = {PVO2_FIXED:8.4f}  [FIXED]")
        for name, val, (lo, hi) in zip(phys_names_b, flat_b[:2], phys_bounds_b):
            flag = " ** AT BOUND **" if is_at_bound(val, lo, hi) else ""
            print(f"    {name:>12s} = {val:8.4f}  [{lo:>6.2f}, {hi:>6.2f}]{flag}")

        n_phys_b = 2
        delta_offset_b = n_phys_b
        ic_offset_b = delta_offset_b + n_holds
        deltas_b = flat_b[delta_offset_b:delta_offset_b + n_holds]

        print(f"\n  {label} -- Per-hold shifts:")
        for i, h in enumerate(fit_holds):
            residual = deltas_b[i]
            eff = max(tau_0 + residual, EFF_LAG_MIN)
            print(f"    {h['type']}#{h['id']}: "
                  f"delta={residual:+6.2f}, eff_lag={eff:6.2f}")

        print(f"\n  {label} -- Per-hold ICs:")
        for i, h in enumerate(fit_holds):
            offset = ic_offset_b + i * N_PH
            tau_washout, paco2_0 = flat_b[offset:offset + N_PH]
            pao2_0 = corrected_pao2_0(paco2_0, 0.0)
            ph_bounds = PERHOLD_BOUNDS[h["type"]]
            at = []
            for val, (lo, hi), name in zip([tau_washout, paco2_0], ph_bounds, PERHOLD_NAMES):
                if is_at_bound(val, lo, hi):
                    at.append(f"{name}={'lo' if abs(val - lo) < 1e-3 else 'hi'}")
            bound_str = f"  [{', '.join(at)}]" if at else ""
            print(f"    {h['type']}#{h['id']}: tau_w={tau_washout:.1f}, "
                  f"paco2_0={paco2_0:.1f}, PaO2_0={pao2_0:.1f}{bound_str}")

    print_stage_b_results(flat_b, "Stage B (frozen-lag)")

    eval_b = evaluate_stage_b(
        flat_b, fit_holds, nadir_info, frozen_sensor, s_base_values, all_holds)

    for r in eval_b:
        r["is_qc_flagged"] = qc_flags.get(r["hold_id"], False)

    # ── Comparison table ─────────────────────────────────────────────────────
    variant_names = ["A:sensor", "B:physiology"]
    all_results = eval_a + eval_b
    print_comparison_table(all_results, variant_names)

    # ══════════════════════════════════════════════════════════════════════════
    # LOHO: Leave-One-Hold-Out (Stage A)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("LOHO: Leave-One-Hold-Out Cross-Validation (Stage A)")
    print(f"{'='*120}")

    loho_results = run_stage_a_loho(fit_holds, nadir_info, s_base_values)

    print(f"\n  LOHO-Global Summary (no delta for held-out):")
    print(f"  {'Hold':<15s} | {'R2':>8s} | {'RMSE':>8s} | {'NadirErr':>8s} | {'EffLag':>8s}")
    print(f"  {'-'*15}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for r in loho_results:
        r2_str = f"{r['r2']:.4f}" if r["r2"] is not None else "N/A"
        rmse_str = f"{r['rmse']:.2f}" if r["rmse"] is not None else "N/A"
        print(f"  {r['hold_type']}#{r['hold_id']:<10d} | {r2_str:>8s} | {rmse_str:>8s} | "
              f"{r['nadir_err']:+8.1f} | {r['eff_lag']:8.1f}")

    loho_nerrs = [abs(r["nadir_err"]) for r in loho_results]
    avg_loho_nerr = np.mean(loho_nerrs)
    print(f"\n  Avg LOHO-Global |timing error|: {avg_loho_nerr:.1f}s (target < 5s)")

    print(f"\n  LOHO-Inference Summary (delta inferred for held-out):")
    print(f"  {'Hold':<15s} | {'R2':>8s} | {'RMSE':>8s} | {'NadirErr':>8s} | {'EffLag':>8s} | {'Delta':>8s}")
    print(f"  {'-'*15}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for r in loho_results:
        r2_str = f"{r['r2_inf']:.4f}" if r["r2_inf"] is not None else "N/A"
        rmse_str = f"{r['rmse_inf']:.2f}" if r["rmse_inf"] is not None else "N/A"
        print(f"  {r['hold_type']}#{r['hold_id']:<10d} | {r2_str:>8s} | {rmse_str:>8s} | "
              f"{r['nadir_err_inf']:+8.1f} | {r['eff_lag_inf']:8.1f} | {r['inf_delta']:+8.1f}")

    loho_inf_nerrs = [abs(r["nadir_err_inf"]) for r in loho_results]
    avg_loho_inf_nerr = np.mean(loho_inf_nerrs)
    print(f"\n  Avg LOHO-Inference |timing error|: {avg_loho_inf_nerr:.1f}s")

    # ══════════════════════════════════════════════════════════════════════════
    # TIME-SPLIT (Stage B)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("TIME-SPLIT: Fit Early Apnea, Predict Late (Stage B)")
    print(f"{'='*120}")

    ts_results = run_stage_b_time_split(fit_holds, nadir_info, frozen_sensor, s_base_values)

    r2_preds = [r["r2_pred"] for r in ts_results if r["r2_pred"] is not None]
    avg_r2_pred = np.mean(r2_preds) if r2_preds else float("nan")
    print(f"\n  Avg time-split R2_pred: {avg_r2_pred:.4f} (target > 0.90)")

    # ══════════════════════════════════════════════════════════════════════════
    # PROFILE LIKELIHOOD (tau_0 and p)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("PROFILE LIKELIHOOD: tau_0 and p sweeps (Stage A)")
    print(f"{'='*120}")

    tau_0_values = [5, 8, 10, 12, 14, 16, 18, 20, 22, 25, 30, 35, 40, 45]
    p_values = [1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    tau_profile, p_profile = run_profile_likelihood(
        fit_holds, nadir_info, s_base_values, tau_0_values, p_values)

    print(f"\n  {'tau_0':>8s} | {'loss':>10s} | {'b_s':>8s} | {'p':>8s}")
    print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}")
    for tau_val in sorted(tau_profile.keys()):
        r = tau_profile[tau_val]
        s = r["flat"]
        _, b_s_v, p_v = s[:3]
        print(f"  {tau_val:8.1f} | {r['loss']:10.2f} | {b_s_v:8.4f} | {p_v:8.4f}")

    losses = [tau_profile[t]["loss"] for t in sorted(tau_profile.keys())]
    is_monotone_dec = all(losses[i] >= losses[i + 1] for i in range(len(losses) - 1))
    is_monotone_inc = all(losses[i] <= losses[i + 1] for i in range(len(losses) - 1))
    is_monotone_tau = is_monotone_dec or is_monotone_inc
    sorted_taus = sorted(tau_profile.keys())
    min_tau = sorted_taus[np.argmin(losses)]
    print(f"\n  tau_0 profile non-monotone: {'YES (good!)' if not is_monotone_tau else 'NO (degenerate)'}")
    print(f"  Minimum loss at tau_0={min_tau:.1f}")

    print(f"\n  {'p':>8s} | {'loss':>10s} | {'tau_0':>8s} | {'b_s':>8s}")
    print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}")
    for p_val in sorted(p_profile.keys()):
        r = p_profile[p_val]
        s = r["flat"]
        tau_v, b_s_v, _ = s[:3]
        print(f"  {p_val:8.2f} | {r['loss']:10.2f} | {tau_v:8.4f} | {b_s_v:8.4f}")

    losses_p = [p_profile[pv]["loss"] for pv in sorted(p_profile.keys())]
    is_monotone_p_dec = all(losses_p[i] >= losses_p[i + 1] for i in range(len(losses_p) - 1))
    is_monotone_p_inc = all(losses_p[i] <= losses_p[i + 1] for i in range(len(losses_p) - 1))
    is_monotone_p = is_monotone_p_dec or is_monotone_p_inc
    sorted_ps = sorted(p_profile.keys())
    min_p = sorted_ps[np.argmin(losses_p)]
    print(f"\n  p profile non-monotone: {'YES (good!)' if not is_monotone_p else 'NO (degenerate)'}")
    print(f"  Minimum loss at p={min_p:.2f}")

    # ══════════════════════════════════════════════════════════════════════════
    # SENSITIVITY: d(physiology)/d(tau_0)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("SENSITIVITY: d(physiology params)/d(tau_0)")
    print(f"{'='*120}")

    sensitivities = compute_sensitivity(
        tau_profile, fit_holds, nadir_info, frozen_sensor, s_base_values)

    # ══════════════════════════════════════════════════════════════════════════
    # SPONGE DIAGNOSTICS
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("SPONGE DIAGNOSTICS")
    print(f"{'='*120}")

    # Stage A sponge
    a_param_names = ["tau_0", "b_s", "p"]
    a_bounds = [TAU0_BOUNDS, GAIN_BOUNDS, P_BOUNDS]
    a_prior_sigmas = {
        "tau_0": 0.4,  # LogNormal sigma in log-space
        "b_s": 0.1,
        "p": 0.35,  # LogNormal sigma in log-space
    }
    at_bound_a = sponge_diagnostics(
        flat_a, a_bounds, a_prior_sigmas, a_param_names, label="Stage A")

    # Stage B sponge
    b_param_names = ["k_co2", "gamma"]
    b_bounds = [(0.02, 0.25), GAMMA_BOUNDS]
    b_prior_sigmas = {
        "k_co2": 0.02,
        "gamma": 0.15,
    }
    at_bound_b = sponge_diagnostics(
        flat_b, b_bounds, b_prior_sigmas, b_param_names, label="Stage B")

    # Hold influence from LOHO
    print(f"\n  Hold Influence (from LOHO):")
    if loho_results:
        loho_losses = [r["loss"] for r in loho_results]
        mean_loss = np.mean(loho_losses)
        for r in loho_results:
            influence = (r["loss"] - mean_loss) / mean_loss * 100
            print(f"    {r['hold_type']}#{r['hold_id']}: loss={r['loss']:.2f} "
                  f"(influence={influence:+.1f}%)")

    # ── Plots ────────────────────────────────────────────────────────────────
    output_dir = Path(__file__).resolve().parent

    plot_stage_a_detail(eval_a, latent_curves_a, all_holds, nadir_info,
                        output_dir / "exp_v7_05_stage_a.png")

    plot_stage_b_detail(eval_b, all_holds, nadir_info, frozen_sensor,
                        output_dir / "exp_v7_05_stage_b.png")

    plot_loho_summary(loho_results, output_dir / "exp_v7_05_loho.png")

    plot_profile_likelihood(tau_profile, p_profile, output_dir / "exp_v7_05_profile.png")

    plot_sensitivity(sensitivities, output_dir / "exp_v7_05_sensitivity.png")

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE B WEAK-LAG DIAGNOSTIC
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("STAGE B WEAK-LAG DIAGNOSTIC (delta free with prior)")
    print(f"{'='*120}")

    delta_a_values = frozen_sensor["deltas"]
    print(f"  Stage A delta values: {[f'{d:+.2f}' for d in delta_a_values]}")
    print(f"  Weak-lag: delta free with prior N(delta_stageA, 3)")

    def run_stage_b_weak_lag(fit_holds, nadir_info, frozen_sensor, delta_a_values, s_base_values):
        """Stage B with weak lag prior: delta free with N(delta_stageA, 3)."""
        tau_0_frozen = frozen_sensor["tau_0"]
        b_s_frozen = frozen_sensor["b_s"]
        cv_frozen = frozen_sensor["cv"]

        n_holds = len(fit_holds)
        bounds = [
            (0.02, 0.25),  # k_co2
            GAMMA_BOUNDS,  # gamma
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

        lambda_wl = 1.0 / (2.0 * 3.0**2)  # sigma=3

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
            callback=make_de_callback("WeakLag", maxiter_wl),
        )
        print(f"\n  Converged: {res.success}, fun={res.fun:.2f}, nfev={res.nfev}")
        return res.x, res.success

    flat_b_wl, conv_b_wl = run_stage_b_weak_lag(
        fit_holds, nadir_info, frozen_sensor, delta_a_values, s_base_values)

    print_stage_b_results(flat_b_wl, "Stage B (weak-lag)")

    eval_b_wl = evaluate_stage_b(
        flat_b_wl, fit_holds, nadir_info, frozen_sensor, s_base_values, all_holds)

    # Compare frozen-lag vs weak-lag physiology
    print(f"\n  Frozen-lag vs Weak-lag Comparison:")
    print(f"  {'Param':<12s} | {'Frozen':>10s} | {'Weak':>10s} | {'Diff%':>8s}")
    print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")
    phys_names_cmp = ["k_co2", "gamma"]
    for pi, pname in enumerate(phys_names_cmp):
        v_frozen = flat_b[pi]
        v_weak = flat_b_wl[pi]
        pct_diff = abs(v_weak - v_frozen) / max(abs(v_frozen), 1e-6) * 100
        print(f"  {pname:<12s} | {v_frozen:10.4f} | {v_weak:10.4f} | {pct_diff:7.1f}%")

    max_phys_diff = max(
        abs(flat_b_wl[pi] - flat_b[pi]) / max(abs(flat_b[pi]), 1e-6) * 100
        for pi in range(2)
    )
    wl_converged = max_phys_diff < 20
    print(f"\n  Max physiology divergence: {max_phys_diff:.1f}% "
          f"(target < 20%: {'PASS' if wl_converged else 'FAIL'})")

    # ── Success criteria ─────────────────────────────────────────────────────
    print(f"\n{'='*120}")
    print("SUCCESS CRITERIA CHECK")
    print(f"{'='*120}")

    # 1. b_s near 1.0
    print(f"\n  1. b_s={b_s_a:.4f} in (0.8, 1.2): "
          f"{'PASS' if 0.8 < b_s_a < 1.2 else 'FAIL'}")

    # 2. tau_0 in [10, 30]
    print(f"  2. tau_0={tau_0:.1f} in [10, 30]: "
          f"{'PASS' if 10 <= tau_0 <= 30 else 'FAIL'}")

    # 3. p in [1.5, 3.5]
    print(f"  3. p={p_a:.3f} in [1.5, 3.5]: "
          f"{'PASS' if 1.5 <= p_a <= 3.5 else 'FAIL (but ok if identifiable)'}")

    # 4. Saturation near 0%
    print(f"  4. Saturation: {pct_total:.1f}% (target ~0%): "
          f"{'PASS' if pct_total < 1 else 'FAIL'}")

    # 5. No deltas at bounds
    deltas_at_bound = sum(1 for d in deltas_a if is_at_bound(d, *DELTA_BOUNDS))
    print(f"  5. Deltas at bounds: {deltas_at_bound}/5: "
          f"{'PASS' if deltas_at_bound == 0 else 'FAIL'}")

    # 6. Delta range < 15s
    delta_range = float(np.max(deltas_a) - np.min(deltas_a))
    print(f"  6. Delta range: {delta_range:.1f}s (target < 15s): "
          f"{'PASS' if delta_range < 15 else 'FAIL'}")

    # 7. eff_lag > EFF_LAG_MIN for all holds
    eff_lags = [max(tau_0 + d, EFF_LAG_MIN) for d in deltas_a]
    all_above_floor = all(el > EFF_LAG_MIN + 0.01 for el in eff_lags)
    print(f"  7. eff_lag > {EFF_LAG_MIN}: {['%.1f' % e for e in eff_lags]}: "
          f"{'PASS' if all_above_floor else 'FLOOR BINDING'}")

    # 8. LOHO timing error
    print(f"  8. LOHO-Global |timing error| avg={avg_loho_nerr:.1f}s (target < 5s): "
          f"{'PASS' if avg_loho_nerr < 5 else 'FAIL'}")

    # 9. LOHO-Inference R2
    loho_inf_r2s = [r["r2_inf"] for r in loho_results if r["r2_inf"] is not None]
    all_r2_above_05 = all(r2 >= 0.5 for r2 in loho_inf_r2s)
    print(f"  9. LOHO-Inference R2 >= 0.5: {['%.3f' % r for r in loho_inf_r2s]}: "
          f"{'PASS' if all_r2_above_05 else 'FAIL'}")

    # 10. tau_0 profile non-monotone
    print(f"  10. tau_0 profile non-monotone: "
          f"{'PASS' if not is_monotone_tau else 'FAIL (monotone)'}")
    print(f"      Minimum at tau_0={min_tau:.1f}")

    # 11. p profile non-monotone
    print(f"  11. p profile non-monotone: "
          f"{'PASS' if not is_monotone_p else 'FAIL (monotone)'}")
    print(f"      Minimum at p={min_p:.2f}")

    # 12. Frozen-lag vs weak-lag convergence
    print(f"  12. Frozen vs weak-lag divergence={max_phys_diff:.1f}% (target < 20%): "
          f"{'PASS' if wl_converged else 'FAIL'}")

    # 13. Gamma interior
    gamma_b = flat_b[1]  # index 1 in [k_co2, gamma]
    gamma_interior = not is_at_bound(gamma_b, *GAMMA_BOUNDS)
    print(f"  13. gamma={gamma_b:.4f} interior: "
          f"{'PASS' if gamma_interior else 'FAIL (at bound)'}")

    # 14. Params at bounds
    total_at_bound = at_bound_a + at_bound_b
    print(f"  14. Global params at bounds: {total_at_bound}")

    # 15. QC-passing averages
    qc_passing = [r for r in eval_a if not r["is_excluded"] and not qc_flags[r["hold_id"]]]
    if qc_passing:
        qc_r2n = np.mean([r["r2_nadir"] for r in qc_passing if r["r2_nadir"] is not None])
        qc_nerr = np.mean([abs(r["nadir_err"]) for r in qc_passing])
        print(f"  15. QC-passing avg R2n={qc_r2n:.4f}, |timing|={qc_nerr:.1f}s")

    print(f"\n{'='*120}")
    print("DONE")
    print(f"{'='*120}")


if __name__ == "__main__":
    main()
