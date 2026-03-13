"""
v7 Experiment 01: Minimal Identifiable Sensor-First SpO2 Model.

v6.07 established two-stage fitting (sensor-first, then physiology) but revealed
identifiability issues: cv always collapses to 0.10 (bound), tau_reoxy always hits
30s (bound), and RV#4 is a structural outlier.

V7 fixes:
  1. Fix cv=0.15 (was unidentifiable, always at bound)
  2. Drop tau_reoxy entirely (always at bound, recovery model inadequate)
  3. Piecewise-linear latent (3 params/hold, not 8-knot PCHIP)
  4. Student-t NLL loss (robust to outliers)
  5. Tight m_h prior (HalfNormal(2)) — critical identifiability lever
  6. Gain parameter b_s (scale invariance)
  7. Tighter gamma bounds [0.8, 1.3] with strong prior

Two stages + diagnostics:
  Stage A: Sensor calibration (30 params) — nadir window only
  Stage B: Physiology (18 params) — apnea-only, frozen sensor
  Diagnostics: LOHO, time-split, profile likelihood, sensitivity, sponge

Usage:
    cd backend && uv run python -u scripts/experiments/exp_v7_01/exp_v7_01_identifiable_sensor.py
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
from scipy.optimize import differential_evolution
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

# ── Stage A: Sensor regularization ──────────────────────────────────────────

LAMBDA_TAU0 = 50.0         # LogNormal(log 18, 0.4) — prior on base delay
LAMBDA_BETA = 50.0         # N(0, 2^2) per covariate
LAMBDA_DELTA = 5.0         # StudentT-like shrinkage, sigma ~3s
LAMBDA_ZEROSUM = 500.0     # Zero-sum on deltas
LAMBDA_M = 200.0           # HalfNormal(2) — TIGHT (critical identifiability lever)
LAMBDA_OFFSET = 50.0       # N(0, 2^2) for r_offset
LAMBDA_GAIN = 5000.0       # N(1, 0.02^2) for gain
LAMBDA_NADIR = 500.0       # Huber timing penalty (delta=8s)
CV_FIXED = 0.15            # Fixed kernel shape

# ── Stage B: Physiology regularization ──────────────────────────────────────

LAMBDA_PVO2 = 20.0         # N(25, 5^2)
LAMBDA_K_CO2 = 2000.0      # N(0.06, 0.02^2)
LAMBDA_PACO2 = 1000.0      # Per-hold PaCO2_0 → 40
LAMBDA_GAMMA = 5000.0      # N(1.0, 0.1^2) — TIGHT
LAMBDA_REG = 10.0          # Per-hold IC → type-mean

# ── Shared constants ────────────────────────────────────────────────────────

TAU0_PRIOR_CENTER = 18.0   # Prior center for base delay
NADIR_WINDOW_AFTER = 45    # seconds after t_end for loss window

# ── Bounds ──────────────────────────────────────────────────────────────────

# Stage A
TAU0_BOUNDS = (5, 45)          # base delay
OFFSET_BOUNDS = (-8, 8)        # r_offset (a_s)
GAIN_BOUNDS = (0.8, 1.2)       # b_s gain
BETA_G_BOUNDS = (-2.0, 2.0)    # depth severity coefficient
BETA_S_BOUNDS = (-2.0, 2.0)    # slope coefficient
DELTA_BOUNDS = (-15, 15)       # per-hold residual shifts
M_H_BOUNDS = (-10, 15)         # per-hold nadir shift

# Piecewise-linear latent
S_START_BOUNDS = (90, 100)     # initial SpO2
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


# ── NEW: Piecewise-linear latent template ───────────────────────────────────


def build_piecewise_linear(t_1hz, t_end, S_start, S_min, v_up, m_h):
    """Build piecewise-linear latent SaO2 curve.

    3 free params per hold: S_start, S_min, v_up.
    Turning point at t_turn = t_end + m_h.
    v_down is derived: (S_start - S_min) / t_turn.
    Forces kernel to "do something" because cusp must be rounded by convolution.
    """
    t_turn = max(t_end + m_h, 1.0)

    # Down slope derived from geometry
    if t_turn > 0:
        v_down = (S_start - S_min) / t_turn
    else:
        v_down = 0.0

    latent = np.empty_like(t_1hz)
    for i, t in enumerate(t_1hz):
        if t <= t_turn:
            # Linear descent
            latent[i] = S_start - v_down * t
        else:
            # Linear ascent from nadir
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
    """Compute observed nadir timing and SpO2 for a hold."""
    idx = np.argmin(hold["spo2"])
    return {
        "t_nadir": hold["t"][idx],
        "spo2_nadir": hold["spo2"][idx],
        "in_recovery": hold["t"][idx] > hold["t_end"],
        "delay_from_end": hold["t"][idx] - hold["t_end"],
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


def run_stage_a(fit_holds, nadir_info, severities, end_slopes):
    """Stage A: Sensor calibration with piecewise-linear latent.

    Parameter layout (30 params for 5 holds):
      Global (3): tau_0, r_offset (a_s), gain (b_s)
      Covariates (2): beta_g, beta_s
      Per-hold deltas (5): delta_h
      Per-hold nadir shifts (5): m_h
      Per-hold latent (15): S_start_h, S_min_h, v_up_h (x5)
    """
    n_holds = len(fit_holds)

    # Build bounds
    bounds = [
        TAU0_BOUNDS,      # tau_0
        OFFSET_BOUNDS,    # r_offset (a_s)
        GAIN_BOUNDS,      # b_s (gain)
        BETA_G_BOUNDS,    # beta_g
        BETA_S_BOUNDS,    # beta_s
    ]
    n_global = len(bounds)

    # Per-hold deltas
    delta_offset = n_global
    for _ in fit_holds:
        bounds.append(DELTA_BOUNDS)

    # Per-hold nadir shifts
    m_h_offset = delta_offset + n_holds
    for _ in fit_holds:
        bounds.append(M_H_BOUNDS)

    # Per-hold latent params: S_start, S_min, v_up
    latent_offset = m_h_offset + n_holds
    n_latent_per_hold = 3
    for _ in fit_holds:
        bounds.append(S_START_BOUNDS)
        bounds.append(S_MIN_BOUNDS)
        bounds.append(V_UP_BOUNDS)

    n_total = len(bounds)

    g_h = [severities[h["id"]] for h in fit_holds]
    neg_s_h = [-end_slopes[h["id"]] for h in fit_holds]

    print(f"\n  Stage A: {n_total} params ({n_global} global + {n_holds} delta + "
          f"{n_holds} m_h + {n_latent_per_hold}x{n_holds} latent)")
    print(f"  Lag model: eff_lag = tau_0 + beta_g*g_h + beta_s*(-s_h) + delta_h")
    print(f"  Fixed: cv={CV_FIXED}")
    print(f"  Loss: Student-t NLL (nu={NU_STUDENT}, sigma={SIGMA_STUDENT})")
    print(f"  Severity g_h: {[f'{g:.2f}' for g in g_h]}")
    print(f"  End-slope -s_h: {[f'{s:.3f}' for s in neg_s_h]}")

    masks = [nadir_window_mask(h["t"], h["t_end"]) for h in fit_holds]
    nadir_ts = [nadir_info[h["id"]]["t_nadir"] for h in fit_holds]

    # Precompute 1-Hz time grids
    t_grids = []
    for h in fit_holds:
        t_max = h["t"][-1]
        t_1hz = np.arange(0, t_max + 1, 1.0)
        t_grids.append(t_1hz)

    def objective(flat):
        tau_0, r_offset, b_s, beta_g, beta_s = flat[:n_global]
        deltas = flat[delta_offset:delta_offset + n_holds]
        m_shifts = flat[m_h_offset:m_h_offset + n_holds]
        total = 0.0

        for i, h in enumerate(fit_holds):
            # Extract latent params
            lp_start = latent_offset + i * n_latent_per_hold
            S_start, S_min, v_up = flat[lp_start:lp_start + n_latent_per_hold]

            # Build piecewise-linear latent
            t_1hz = t_grids[i]
            latent = build_piecewise_linear(t_1hz, h["t_end"], S_start, S_min, v_up, m_shifts[i])

            # Apply gamma kernel with effective lag
            shift = beta_g * g_h[i] + beta_s * neg_s_h[i] + deltas[i]
            eff_lag = max(tau_0 + shift, 1.0)
            filtered = apply_gamma_kernel(latent, eff_lag, CV_FIXED)

            # Measurement equation: y = a_s + b_s * filtered
            pred_1hz = r_offset + b_s * filtered
            pred_at_obs = np.interp(h["t"], t_1hz, pred_1hz)
            pred_at_obs = np.clip(pred_at_obs, 0.0, 100.0)

            # Student-t NLL on nadir window
            m = masks[i]
            residuals = h["spo2"][m] - pred_at_obs[m]
            total += student_t_nll(residuals)

            # Huber timing penalty
            total += nadir_timing_penalty_huber(h["t"][m], pred_at_obs[m], nadir_ts[i])

        # ── Priors ──

        # tau_0: LogNormal-like prior centered at 18
        total += LAMBDA_TAU0 * (np.log(max(tau_0, 1.0)) - np.log(TAU0_PRIOR_CENTER)) ** 2

        # Covariates: N(0, 2^2)
        total += LAMBDA_BETA * beta_g ** 2
        total += LAMBDA_BETA * beta_s ** 2

        # Per-hold deltas: StudentT-like shrinkage
        total += LAMBDA_DELTA * np.sum(np.log1p(deltas**2 / 9.0))  # sigma=3
        # Zero-sum constraint
        total += LAMBDA_ZEROSUM * np.sum(deltas) ** 2

        # m_h: HalfNormal(2) prior — TIGHT
        total += LAMBDA_M * np.sum(m_shifts**2 / 4.0)  # sigma=2

        # r_offset: N(0, 2^2)
        total += LAMBDA_OFFSET * r_offset ** 2

        # gain: N(1, 0.02^2)
        total += LAMBDA_GAIN * (b_s - 1.0) ** 2

        return total

    maxiter_a = 3000
    res = differential_evolution(
        objective, bounds, maxiter=maxiter_a, seed=42, tol=1e-10,
        polish=True, popsize=30, mutation=(0.5, 1.5), recombination=0.9,
        callback=make_de_callback("Stage A", maxiter_a),
    )
    print(f"\n  Converged: {res.success}, fun={res.fun:.2f}, nfev={res.nfev}")
    return res.x, res.success


def evaluate_stage_a(flat_a, fit_holds, nadir_info, severities, end_slopes):
    """Evaluate Stage A: sensor model with piecewise-linear latent."""
    n_holds = len(fit_holds)
    n_global = 5
    delta_offset = n_global
    m_h_offset = delta_offset + n_holds
    latent_offset = m_h_offset + n_holds
    n_latent_per_hold = 3

    tau_0, r_offset, b_s, beta_g, beta_s = flat_a[:n_global]
    deltas = flat_a[delta_offset:delta_offset + n_holds]
    m_shifts = flat_a[m_h_offset:m_h_offset + n_holds]
    g_h = [severities[h["id"]] for h in fit_holds]
    neg_s_h = [-end_slopes[h["id"]] for h in fit_holds]

    results = []
    latent_curves = []

    for i, h in enumerate(fit_holds):
        lp_start = latent_offset + i * n_latent_per_hold
        S_start, S_min, v_up = flat_a[lp_start:lp_start + n_latent_per_hold]

        t_1hz = np.arange(0, h["t"][-1] + 1, 1.0)
        latent = build_piecewise_linear(t_1hz, h["t_end"], S_start, S_min, v_up, m_shifts[i])

        shift = beta_g * g_h[i] + beta_s * neg_s_h[i] + deltas[i]
        eff_lag = max(tau_0 + shift, 1.0)
        filtered = apply_gamma_kernel(latent, eff_lag, CV_FIXED)

        pred_1hz = r_offset + b_s * filtered
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
            "m_h": m_shifts[i],
        })

    return results, latent_curves


def extract_frozen_sensor(flat_a, fit_holds, severities, end_slopes):
    """Extract frozen sensor params from Stage A for use in Stage B."""
    n_holds = len(fit_holds)
    n_global = 5
    delta_offset = n_global

    tau_0, r_offset, b_s, beta_g, beta_s = flat_a[:n_global]
    deltas = flat_a[delta_offset:delta_offset + n_holds]
    m_h_offset = delta_offset + n_holds
    m_shifts = flat_a[m_h_offset:m_h_offset + n_holds]
    latent_offset = m_h_offset + n_holds

    return {
        "tau_0": tau_0,
        "r_offset": r_offset,
        "b_s": b_s,
        "beta_g": beta_g,
        "beta_s": beta_s,
        "deltas": deltas,
        "m_shifts": m_shifts,
        "cv": CV_FIXED,
    }


# ── Stage B: Physiology (apnea-only, frozen sensor) ────────────────────────


def predict_v7(t, pvo2, tau_washout, gamma, paco2_0, k_co2, r_offset, b_s,
               mean_lag, cv, t_end, shift=0.0):
    """Full sensor pipeline for v7 — apnea-only (no tau_reoxy)."""
    aa = 0.0
    pao2_0 = corrected_pao2_0(paco2_0, aa)
    pao2 = pao2_apnea_only(t, pao2_0, pvo2, tau_washout, t_end)
    p50 = p50_apnea_only(t, paco2_0, k_co2, t_end)
    sa = odc_severinghaus(pao2, p50, gamma)

    eff_mean_lag = max(mean_lag + shift, 1.0)
    filtered = apply_gamma_kernel(sa, eff_mean_lag, cv)

    return np.clip(r_offset + b_s * filtered, 0.0, 100.0)


def run_stage_b(fit_holds, nadir_info, severities, end_slopes, frozen_sensor):
    """Stage B: Physiology model with frozen sensor from Stage A.

    Parameter layout (18 params for 5 holds):
      Global physiology (3): PvO2, k_CO2, gamma
      Per-hold deltas (5): re-fitted for physiology stage
      Per-hold ICs (10): tau_washout_h, PaCO2_0_h (x5)
    """
    tau_0_frozen = frozen_sensor["tau_0"]
    r_offset_frozen = frozen_sensor["r_offset"]
    b_s_frozen = frozen_sensor["b_s"]
    beta_g_frozen = frozen_sensor["beta_g"]
    beta_s_frozen = frozen_sensor["beta_s"]
    cv_frozen = frozen_sensor["cv"]

    n_holds = len(fit_holds)

    # Physiology shared: pvo2, k_co2, gamma
    bounds = [
        (15, 50),      # pvo2
        (0.02, 0.25),  # k_co2
        GAMMA_BOUNDS,  # gamma [0.8, 1.3]
    ]
    phys_names = ["pvo2", "k_co2", "gamma"]
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

    g_h = [severities[h["id"]] for h in fit_holds]
    neg_s_h = [-end_slopes[h["id"]] for h in fit_holds]

    # Apnea-only loss window: [0, t_end+5]
    apnea_window = 5
    masks = [h["t"] <= h["t_end"] + apnea_window for h in fit_holds]
    weights = [np.where(h["spo2"][m] < 95, 3.0, 1.0) for h, m in zip(fit_holds, masks)]

    print(f"\n  Stage B: {n_total} params ({n_phys} physiology + {n_holds} delta + "
          f"{N_PH}x{n_holds} per-hold ICs)")
    print(f"  Frozen sensor: tau_0={tau_0_frozen:.2f}, cv={cv_frozen:.3f}, "
          f"r_offset={r_offset_frozen:.2f}, b_s={b_s_frozen:.4f}, "
          f"beta_g={beta_g_frozen:.3f}, beta_s={beta_s_frozen:.3f}")
    print(f"  gamma bounds: [{GAMMA_BOUNDS[0]}, {GAMMA_BOUNDS[1]}] (tighter than v6.07)")
    print(f"  Apnea-only loss: window [0, t_end+{apnea_window}s]")

    def objective(flat):
        pvo2, k_co2, gamma_val = flat[:n_phys]
        deltas = flat[delta_offset:delta_offset + n_holds]
        total = 0.0

        for i, h in enumerate(fit_holds):
            ph_offset = ic_offset + i * N_PH
            tau_washout, paco2_0 = flat[ph_offset:ph_offset + N_PH]

            shift = (beta_g_frozen * g_h[i] + beta_s_frozen * neg_s_h[i]
                     + deltas[i])
            pred = predict_v7(
                h["t"], pvo2, tau_washout, gamma_val,
                paco2_0, k_co2, r_offset_frozen, b_s_frozen,
                tau_0_frozen, cv_frozen, h["t_end"],
                shift=shift,
            )
            m = masks[i]
            total += np.sum(weights[i] * (h["spo2"][m] - pred[m]) ** 2)

            total += LAMBDA_PACO2 * (paco2_0 - 40.0) ** 2

        # Priors
        total += LAMBDA_PVO2 * (pvo2 - 25.0) ** 2
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
        callback=make_de_callback("Stage B", maxiter_b),
    )
    print(f"\n  Converged: {res.success}, fun={res.fun:.2f}, nfev={res.nfev}")
    return res.x, res.success


def evaluate_stage_b(flat_b, fit_holds, nadir_info, severities, end_slopes,
                     frozen_sensor, all_holds=None):
    """Evaluate Stage B: frozen sensor + apnea-only physiology."""
    results = []
    target_holds = all_holds if all_holds is not None else fit_holds
    fit_ids = {h["id"] for h in fit_holds}
    n_holds = len(fit_holds)

    n_phys = 3
    delta_offset = n_phys
    ic_offset = delta_offset + n_holds

    pvo2, k_co2, gamma_val = flat_b[:n_phys]
    deltas = flat_b[delta_offset:delta_offset + n_holds]

    tau_0 = frozen_sensor["tau_0"]
    cv = frozen_sensor["cv"]
    r_offset = frozen_sensor["r_offset"]
    b_s = frozen_sensor["b_s"]
    beta_g = frozen_sensor["beta_g"]
    beta_s = frozen_sensor["beta_s"]

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
            neg_s = -end_slopes.get(h["id"], 0.0)
            shift = beta_g * severities[h["id"]] + beta_s * neg_s + delta_val

        pred_full = predict_v7(
            h["t"], pvo2, tau_washout, gamma_val,
            paco2_0, k_co2, r_offset, b_s,
            tau_0, cv, h["t_end"],
            shift=shift,
        )
        pred_apnea = predict_v7(
            h["t_apnea"], pvo2, tau_washout, gamma_val,
            paco2_0, k_co2, r_offset, b_s,
            tau_0, cv, h["t_end"],
            shift=shift,
        )

        r2_full = compute_r2(h["spo2"], pred_full)
        r2_apnea = compute_r2(h["spo2_apnea"], pred_apnea)

        mask = nadir_window_mask(h["t"], h["t_end"])
        r2_nadir = compute_r2(h["spo2"][mask], pred_full[mask]) if mask.sum() > 3 else None

        r2_recovery = None
        if len(h["t_recovery"]) > 3:
            pred_rec = predict_v7(
                h["t_recovery"], pvo2, tau_washout, gamma_val,
                paco2_0, k_co2, r_offset, b_s,
                tau_0, cv, h["t_end"],
                shift=shift,
            )
            r2_recovery = compute_r2(h["spo2_recovery"], pred_rec)

        t_nadir_obs = nadir_info[h["id"]]["t_nadir"]
        t_nadir_pred = h["t"][np.argmin(pred_full)]
        nadir_err = t_nadir_pred - t_nadir_obs

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
            "is_excluded": is_excl or h["id"] in EXCLUDED_IDS,
            "effective_lag": max(tau_0 + shift, 1.0),
            "delta": delta_val,
        }
        results.append(rec)
    return results


# ── LOHO: Leave-One-Hold-Out for Stage A ────────────────────────────────────


def _loho_worker(args):
    """Worker for one LOHO fold. Top-level for pickling."""
    leave_idx, fit_holds, nadir_info, severities, end_slopes = args
    left_out = fit_holds[leave_idx]
    train_holds = [h for i, h in enumerate(fit_holds) if i != leave_idx]
    n_train = len(train_holds)

    bounds = [TAU0_BOUNDS, OFFSET_BOUNDS, GAIN_BOUNDS, BETA_G_BOUNDS, BETA_S_BOUNDS]
    n_global = len(bounds)

    delta_offset = n_global
    for _ in train_holds:
        bounds.append(DELTA_BOUNDS)

    m_h_offset = delta_offset + n_train
    for _ in train_holds:
        bounds.append(M_H_BOUNDS)

    latent_offset = m_h_offset + n_train
    for _ in train_holds:
        bounds.extend([S_START_BOUNDS, S_MIN_BOUNDS, V_UP_BOUNDS])

    heldout_m_h_idx = len(bounds)
    bounds.append(M_H_BOUNDS)
    heldout_latent_offset = heldout_m_h_idx + 1
    bounds.extend([S_START_BOUNDS, S_MIN_BOUNDS, V_UP_BOUNDS])

    g_h_train = [severities[h["id"]] for h in train_holds]
    neg_s_h_train = [-end_slopes[h["id"]] for h in train_holds]
    masks_train = [nadir_window_mask(h["t"], h["t_end"]) for h in train_holds]
    nadir_ts_train = [nadir_info[h["id"]]["t_nadir"] for h in train_holds]
    t_grids_train = [np.arange(0, h["t"][-1] + 1, 1.0) for h in train_holds]

    def objective(flat):
        tau_0, r_off, b_s, beta_g, beta_s = flat[:n_global]
        deltas = flat[delta_offset:delta_offset + n_train]
        m_shifts = flat[m_h_offset:m_h_offset + n_train]
        total = 0.0

        for i, h in enumerate(train_holds):
            lp_start = latent_offset + i * 3
            S_start, S_min, v_up = flat[lp_start:lp_start + 3]
            t_1hz = t_grids_train[i]
            latent = build_piecewise_linear(t_1hz, h["t_end"], S_start, S_min, v_up, m_shifts[i])
            shift = beta_g * g_h_train[i] + beta_s * neg_s_h_train[i] + deltas[i]
            eff_lag = max(tau_0 + shift, 1.0)
            filtered = apply_gamma_kernel(latent, eff_lag, CV_FIXED)
            pred_1hz = r_off + b_s * filtered
            pred_at_obs = np.interp(h["t"], t_1hz, pred_1hz)
            pred_at_obs = np.clip(pred_at_obs, 0.0, 100.0)
            m = masks_train[i]
            total += student_t_nll(h["spo2"][m] - pred_at_obs[m])
            total += nadir_timing_penalty_huber(h["t"][m], pred_at_obs[m], nadir_ts_train[i])

        total += LAMBDA_TAU0 * (np.log(max(tau_0, 1.0)) - np.log(TAU0_PRIOR_CENTER)) ** 2
        total += LAMBDA_BETA * (beta_g ** 2 + beta_s ** 2)
        total += LAMBDA_DELTA * np.sum(np.log1p(deltas**2 / 9.0))
        total += LAMBDA_ZEROSUM * np.sum(deltas) ** 2
        total += LAMBDA_M * np.sum(m_shifts**2 / 4.0)
        total += LAMBDA_OFFSET * r_off ** 2
        total += LAMBDA_GAIN * (b_s - 1.0) ** 2
        return total

    res = differential_evolution(
        objective, bounds, maxiter=2000, seed=42, tol=1e-10,
        polish=True, popsize=25, mutation=(0.5, 1.5), recombination=0.9,
    )

    # Predict held-out hold
    tau_0_fit, r_off_fit, b_s_fit, beta_g_fit, beta_s_fit = res.x[:n_global]
    ho_m_h = res.x[heldout_m_h_idx]
    ho_S_start, ho_S_min, ho_v_up = res.x[heldout_latent_offset:heldout_latent_offset + 3]

    g_ho = severities[left_out["id"]]
    neg_s_ho = -end_slopes[left_out["id"]]
    shift_ho = beta_g_fit * g_ho + beta_s_fit * neg_s_ho

    t_1hz_ho = np.arange(0, left_out["t"][-1] + 1, 1.0)
    latent_ho = build_piecewise_linear(
        t_1hz_ho, left_out["t_end"], ho_S_start, ho_S_min, ho_v_up, ho_m_h)
    eff_lag_ho = max(tau_0_fit + shift_ho, 1.0)
    filtered_ho = apply_gamma_kernel(latent_ho, eff_lag_ho, CV_FIXED)
    pred_ho_1hz = r_off_fit + b_s_fit * filtered_ho
    pred_ho = np.interp(left_out["t"], t_1hz_ho, pred_ho_1hz)
    pred_ho = np.clip(pred_ho, 0.0, 100.0)

    mask_ho = nadir_window_mask(left_out["t"], left_out["t_end"])
    r2_ho = compute_r2(left_out["spo2"][mask_ho], pred_ho[mask_ho]) if mask_ho.sum() > 3 else None
    rmse_ho = compute_rmse(left_out["spo2"][mask_ho], pred_ho[mask_ho]) if mask_ho.sum() > 3 else None

    t_nadir_obs = nadir_info[left_out["id"]]["t_nadir"]
    t_nadir_pred = left_out["t"][np.argmin(pred_ho)]
    nadir_err = t_nadir_pred - t_nadir_obs

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
    }


def run_stage_a_loho(fit_holds, nadir_info, severities, end_slopes):
    """Leave-one-hold-out CV for Stage A — parallelized across folds."""
    n_holds = len(fit_holds)
    print(f"\n  Running {n_holds} LOHO folds in parallel ({N_WORKERS} workers)...")

    args_list = [(i, fit_holds, nadir_info, severities, end_slopes)
                 for i in range(n_holds)]

    with ProcessPoolExecutor(max_workers=min(N_WORKERS, n_holds)) as pool:
        loho_results = list(pool.map(_loho_worker, args_list))

    for r in loho_results:
        print(f"    Held-out {r['hold_type']}#{r['hold_id']}: R2={r['r2']:.4f}, "
              f"RMSE={r['rmse']:.2f}, timing_err={r['nadir_err']:+.1f}s, "
              f"eff_lag={r['eff_lag']:.1f}s")

    return loho_results


# ── Time-split for Stage B ──────────────────────────────────────────────────


def run_stage_b_time_split(fit_holds, nadir_info, severities, end_slopes,
                           frozen_sensor, split_frac=0.6):
    """Time-split validation: fit on early apnea, predict late apnea.

    For each hold: fit [0, split_frac*t_end], predict [split_frac*t_end, t_end+5].
    """
    tau_0_frozen = frozen_sensor["tau_0"]
    r_offset_frozen = frozen_sensor["r_offset"]
    b_s_frozen = frozen_sensor["b_s"]
    beta_g_frozen = frozen_sensor["beta_g"]
    beta_s_frozen = frozen_sensor["beta_s"]
    cv_frozen = frozen_sensor["cv"]

    n_holds = len(fit_holds)
    g_h = [severities[h["id"]] for h in fit_holds]
    neg_s_h = [-end_slopes[h["id"]] for h in fit_holds]

    type_groups = {}
    for i, h in enumerate(fit_holds):
        type_groups.setdefault(h["type"], []).append(i)

    # Fit on early portion of each hold
    split_masks = []
    for h in fit_holds:
        t_split = split_frac * h["t_end"]
        split_masks.append(h["t"] <= t_split)

    bounds = [
        (15, 50),      # pvo2
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

    print(f"\n  Time-split: fitting on [0, {split_frac}*t_end], "
          f"predicting [{split_frac}*t_end, t_end+5]")

    def objective(flat):
        pvo2, k_co2, gamma_val = flat[:n_phys]
        deltas = flat[delta_offset:delta_offset + n_holds]
        total = 0.0

        for i, h in enumerate(fit_holds):
            ph_offset = ic_offset + i * N_PH
            tau_washout, paco2_0 = flat[ph_offset:ph_offset + N_PH]

            shift = (beta_g_frozen * g_h[i] + beta_s_frozen * neg_s_h[i]
                     + deltas[i])
            pred = predict_v7(
                h["t"], pvo2, tau_washout, gamma_val,
                paco2_0, k_co2, r_offset_frozen, b_s_frozen,
                tau_0_frozen, cv_frozen, h["t_end"],
                shift=shift,
            )
            m = split_masks[i]
            weights = np.where(h["spo2"][m] < 95, 3.0, 1.0)
            total += np.sum(weights * (h["spo2"][m] - pred[m]) ** 2)

            total += LAMBDA_PACO2 * (paco2_0 - 40.0) ** 2

        total += LAMBDA_PVO2 * (pvo2 - 25.0) ** 2
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
    pvo2, k_co2, gamma_val = res.x[:n_phys]
    deltas = res.x[delta_offset:delta_offset + n_holds]
    ts_results = []

    for i, h in enumerate(fit_holds):
        ph_offset = ic_offset + i * N_PH
        tau_washout, paco2_0 = res.x[ph_offset:ph_offset + N_PH]

        shift = (beta_g_frozen * g_h[i] + beta_s_frozen * neg_s_h[i] + deltas[i])
        pred = predict_v7(
            h["t"], pvo2, tau_washout, gamma_val,
            paco2_0, k_co2, r_offset_frozen, b_s_frozen,
            tau_0_frozen, cv_frozen, h["t_end"],
            shift=shift,
        )

        # Predict window: [split_frac*t_end, t_end+5]
        t_split = split_frac * h["t_end"]
        pred_mask = (h["t"] > t_split) & (h["t"] <= h["t_end"] + 5)
        fit_mask = split_masks[i]

        r2_fit = compute_r2(h["spo2"][fit_mask], pred[fit_mask]) if fit_mask.sum() > 3 else None
        r2_pred = compute_r2(h["spo2"][pred_mask], pred[pred_mask]) if pred_mask.sum() > 3 else None
        rmse_pred = compute_rmse(h["spo2"][pred_mask], pred[pred_mask]) if pred_mask.sum() > 3 else None

        ts_results.append({
            "hold_id": h["id"],
            "hold_type": h["type"],
            "r2_fit": r2_fit,
            "r2_pred": r2_pred,
            "rmse_pred": rmse_pred,
            "pred": pred,
        })
        print(f"    {h['type']}#{h['id']}: R2_fit={r2_fit:.4f}, R2_pred={r2_pred:.4f}, "
              f"RMSE_pred={rmse_pred:.2f}")

    return ts_results


# ── Profile likelihood ──────────────────────────────────────────────────────


def _profile_worker(args):
    """Worker for one profile likelihood point. Top-level for pickling."""
    tau_fixed, fit_holds, severities, end_slopes, nadir_info = args
    n_holds = len(fit_holds)
    g_h = [severities[h["id"]] for h in fit_holds]
    neg_s_h = [-end_slopes[h["id"]] for h in fit_holds]
    masks = [nadir_window_mask(h["t"], h["t_end"]) for h in fit_holds]
    nadir_ts = [nadir_info[h["id"]]["t_nadir"] for h in fit_holds]
    t_grids = [np.arange(0, h["t"][-1] + 1, 1.0) for h in fit_holds]

    bounds = [
        (tau_fixed - 0.01, tau_fixed + 0.01),
        OFFSET_BOUNDS, GAIN_BOUNDS, BETA_G_BOUNDS, BETA_S_BOUNDS,
    ]
    n_global = len(bounds)
    delta_offset = n_global
    for _ in fit_holds:
        bounds.append(DELTA_BOUNDS)
    m_h_offset = delta_offset + n_holds
    for _ in fit_holds:
        bounds.append(M_H_BOUNDS)
    latent_offset = m_h_offset + n_holds
    for _ in fit_holds:
        bounds.extend([S_START_BOUNDS, S_MIN_BOUNDS, V_UP_BOUNDS])

    def objective(flat):
        tau_0, r_off, b_s, beta_g, beta_s = flat[:n_global]
        deltas = flat[delta_offset:delta_offset + n_holds]
        m_shifts = flat[m_h_offset:m_h_offset + n_holds]
        total = 0.0
        for i, h in enumerate(fit_holds):
            lp_start = latent_offset + i * 3
            S_start, S_min, v_up = flat[lp_start:lp_start + 3]
            t_1hz = t_grids[i]
            latent = build_piecewise_linear(t_1hz, h["t_end"], S_start, S_min, v_up, m_shifts[i])
            shift = beta_g * g_h[i] + beta_s * neg_s_h[i] + deltas[i]
            eff_lag = max(tau_0 + shift, 1.0)
            filtered = apply_gamma_kernel(latent, eff_lag, CV_FIXED)
            pred_1hz = r_off + b_s * filtered
            pred_at_obs = np.interp(h["t"], t_1hz, pred_1hz)
            pred_at_obs = np.clip(pred_at_obs, 0.0, 100.0)
            m = masks[i]
            total += student_t_nll(h["spo2"][m] - pred_at_obs[m])
            total += nadir_timing_penalty_huber(h["t"][m], pred_at_obs[m], nadir_ts[i])
        total += LAMBDA_TAU0 * (np.log(max(tau_0, 1.0)) - np.log(TAU0_PRIOR_CENTER)) ** 2
        total += LAMBDA_BETA * (beta_g ** 2 + beta_s ** 2)
        total += LAMBDA_DELTA * np.sum(np.log1p(deltas**2 / 9.0))
        total += LAMBDA_ZEROSUM * np.sum(deltas) ** 2
        total += LAMBDA_M * np.sum(m_shifts**2 / 4.0)
        total += LAMBDA_OFFSET * r_off ** 2
        total += LAMBDA_GAIN * (b_s - 1.0) ** 2
        return total

    res = differential_evolution(
        objective, bounds, maxiter=2000, seed=42, tol=1e-10,
        polish=True, popsize=25, mutation=(0.5, 1.5), recombination=0.9,
    )
    return tau_fixed, {"flat": res.x, "loss": res.fun, "success": res.success}


def run_profile_likelihood(fit_holds, nadir_info, severities, end_slopes,
                           tau_0_values=None):
    """Fix tau_0 at each value, reoptimize — parallelized."""
    if tau_0_values is None:
        tau_0_values = [5, 8, 10, 12, 14, 16, 18, 20, 22, 25, 30, 35, 40, 45]

    print(f"\n  Running {len(tau_0_values)} profile points in parallel ({N_WORKERS} workers)...")

    args_list = [(tau_val, fit_holds, severities, end_slopes, nadir_info)
                 for tau_val in tau_0_values]

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        raw_results = list(pool.map(_profile_worker, args_list))

    results = {}
    for tau_val, result in raw_results:
        results[tau_val] = result
        bg = result["flat"][3]
        bs = result["flat"][4]
        print(f"    tau_0={tau_val:5.1f}: loss={result['loss']:.2f}, "
              f"beta_g={bg:.3f}, beta_s={bs:.3f}", flush=True)

    return results


# ── Sensitivity analysis ───────────────────────────────────────────────────


def _sensitivity_worker(args):
    """Worker for one Stage B at a given tau_0. Top-level for pickling."""
    tau_val, fit_holds, nadir_info, severities, end_slopes, frozen_sensor_template = args
    frozen = dict(frozen_sensor_template)
    frozen["tau_0"] = tau_val
    flat_b, success = run_stage_b(fit_holds, nadir_info, severities, end_slopes, frozen)
    return tau_val, flat_b


def compute_sensitivity(profile_results, fit_holds, nadir_info, severities,
                        end_slopes, frozen_sensor_template):
    """Run Stage B at multiple tau_0 values — parallelized."""
    losses = {k: v["loss"] for k, v in profile_results.items()}
    sorted_tau = sorted(losses, key=losses.get)
    center_idx = list(sorted(profile_results.keys())).index(sorted_tau[0])
    all_taus = sorted(profile_results.keys())
    start = max(0, center_idx - 2)
    end = min(len(all_taus), start + 5)
    tau_values = all_taus[start:end]

    print(f"\n  Sensitivity: running Stage B at tau_0 = {tau_values} "
          f"in parallel ({N_WORKERS} workers)...")

    args_list = [(tv, fit_holds, nadir_info, severities, end_slopes,
                  frozen_sensor_template) for tv in tau_values]

    with ProcessPoolExecutor(max_workers=min(N_WORKERS, len(tau_values))) as pool:
        raw_results = list(pool.map(_sensitivity_worker, args_list))

    stage_b_results = {tv: fb for tv, fb in raw_results}

    phys_names = ["pvo2", "k_co2", "gamma"]
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

    print(f"\n  {'Avg (fitted)':<15s}", end="")
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

        # Mark turning point
        t_turn = h["t_end"] + lc["m_h"]
        ax.axvline(x=t_turn, color="purple", linestyle=":", alpha=0.5,
                   label=f"Turning pt (m_h={lc['m_h']:+.1f}s)")

        ax.set_title(f"Stage A: {h['type']}#{h['id']} (eff_lag={res['effective_lag']:.1f}s)",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("SpO2 (%)")
        ax.set_ylim(30, 105)
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(True, alpha=0.3)

    fig.suptitle("v7.01 Stage A: Piecewise-Linear Latent + Gamma Kernel",
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

    fig.suptitle("v7.01 Stage B: Apnea-Only Physiology (frozen sensor)",
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

    fig.suptitle("v7.01 Stage A: Leave-One-Hold-Out Cross-Validation",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"LOHO summary plot saved to {output_path}")


def plot_profile_likelihood(profile_results, output_path):
    """Profile likelihood plot: loss and key params vs fixed tau_0."""
    tau_list = sorted(profile_results.keys())

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    configs = [
        (None, "loss", "Profile Loss"),
        (1, "r_offset", "r_offset (a_s)"),
        (2, "b_s", "b_s (gain)"),
        (3, "beta_g", "beta_g (depth)"),
        (4, "beta_s", "beta_s (slope)"),
    ]

    for i, (pidx, name, title) in enumerate(configs):
        ax = axes[i]
        if pidx is not None:
            vals = [profile_results[t]["flat"][pidx] for t in tau_list]
        else:
            vals = [profile_results[t]["loss"] for t in tau_list]
        ax.plot(tau_list, vals, "o-", color="#1f77b4", linewidth=2, markersize=6)
        ax.set_xlabel("tau_0 (fixed, s)")
        ax.set_ylabel(name)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)

    # Last panel: empty or use for annotation
    axes[5].axis("off")

    # Mark minimum on loss plot
    losses = [profile_results[t]["loss"] for t in tau_list]
    min_idx = np.argmin(losses)
    axes[0].axvline(x=tau_list[min_idx], color="red", linestyle="--", alpha=0.5)
    axes[0].annotate(f"min at {tau_list[min_idx]:.0f}s",
                     xy=(tau_list[min_idx], losses[min_idx]),
                     xytext=(10, 10), textcoords="offset points",
                     fontsize=9, color="red")

    fig.suptitle("v7.01: tau_0 Profile Likelihood (Stage A)",
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
    print("v7.01: Minimal Identifiable Sensor-First SpO2 Model")
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
              f"delay_from_end={ni['delay_from_end']:+.0f}s)")

    # ── Compute covariates ───────────────────────────────────────────────────
    severities = compute_depth_severity(fit_holds)
    end_slopes = compute_end_slope(fit_holds)

    print(f"\n  Covariates:")
    print(f"  {'Hold':<15s} | {'g_h (depth)':>12s} | {'s_h (slope)':>12s} | {'-s_h':>8s}")
    print(f"  {'-'*15}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}")
    for h in fit_holds:
        g = severities[h["id"]]
        s = end_slopes[h["id"]]
        print(f"  {h['type']}#{h['id']:<10d} | {g:12.3f} | {s:12.3f} | {-s:8.3f}")

    n_holds = len(fit_holds)

    # ── Summary of changes from v6.07 ────────────────────────────────────────
    print(f"\nKey changes from v6.07:")
    print(f"  1. Fixed cv={CV_FIXED} (was free, always collapsed to 0.10)")
    print(f"  2. Dropped tau_reoxy (was free, always hit 30s bound)")
    print(f"  3. Piecewise-linear latent (3 params/hold vs 8-knot PCHIP)")
    print(f"  4. Student-t NLL loss (nu={NU_STUDENT}, robust to outliers)")
    print(f"  5. Tight m_h prior HalfNormal(2) (LAMBDA_M={LAMBDA_M})")
    print(f"  6. Gain param b_s (scale invariance)")
    print(f"  7. Tighter gamma bounds [{GAMMA_BOUNDS[0]}, {GAMMA_BOUNDS[1]}]")

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE A: Sensor-First Calibration (30 params)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("STAGE A: Sensor-First Calibration (30 params)")
    print(f"{'='*120}")

    flat_a, conv_a = run_stage_a(fit_holds, nadir_info, severities, end_slopes)

    # Print Stage A results
    n_global = 5
    global_names = ["tau_0", "r_offset", "b_s", "beta_g", "beta_s"]
    global_bounds = [TAU0_BOUNDS, OFFSET_BOUNDS, GAIN_BOUNDS, BETA_G_BOUNDS, BETA_S_BOUNDS]
    print(f"\n  Stage A global params:")
    for name, val, (lo, hi) in zip(global_names, flat_a[:n_global], global_bounds):
        flag = " ** AT BOUND **" if is_at_bound(val, lo, hi) else ""
        print(f"    {name:>12s} = {val:8.4f}  [{lo:>6.2f}, {hi:>6.2f}]{flag}")

    tau_0, r_offset_a, b_s_a, beta_g_a, beta_s_a = flat_a[:n_global]
    k_val = 1.0 / (CV_FIXED * CV_FIXED)
    print(f"\n    Kernel: k={k_val:.2f}, mean={tau_0:.1f}s, std={tau_0*CV_FIXED:.1f}s "
          f"(cv={CV_FIXED} FIXED)")

    delta_offset_a = n_global
    m_h_offset_a = delta_offset_a + n_holds
    latent_offset_a = m_h_offset_a + n_holds
    deltas_a = flat_a[delta_offset_a:delta_offset_a + n_holds]
    m_shifts_a = flat_a[m_h_offset_a:m_h_offset_a + n_holds]

    g_h_list = [severities[h["id"]] for h in fit_holds]
    neg_s_h_list = [-end_slopes[h["id"]] for h in fit_holds]

    print(f"\n  Stage A -- Shift breakdown (base tau_0={tau_0:.2f}, "
          f"beta_g={beta_g_a:.3f}, beta_s={beta_s_a:.3f}):")
    for i, h in enumerate(fit_holds):
        sys_g = beta_g_a * g_h_list[i]
        sys_s = beta_s_a * neg_s_h_list[i]
        residual = deltas_a[i]
        total_shift = sys_g + sys_s + residual
        eff = max(tau_0 + total_shift, 1.0)
        bound_str = " *BOUND*" if is_at_bound(residual, *DELTA_BOUNDS) else ""
        lp_start = latent_offset_a + i * 3
        S_start, S_min, v_up = flat_a[lp_start:lp_start + 3]
        print(f"    {h['type']}#{h['id']}: bg*g={sys_g:+6.2f}, bs*(-s)={sys_s:+6.2f}, "
              f"delta={residual:+6.2f}, eff_lag={eff:6.2f}, m_h={m_shifts_a[i]:+.1f}s, "
              f"S_start={S_start:.1f}, S_min={S_min:.1f}, v_up={v_up:.2f}{bound_str}")

    eval_a, latent_curves_a = evaluate_stage_a(
        flat_a, fit_holds, nadir_info, severities, end_slopes)

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE B: Physiology (18 params, frozen sensor)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("STAGE B: Apnea-Only Physiology (18 params, frozen sensor)")
    print(f"{'='*120}")

    frozen_sensor = extract_frozen_sensor(flat_a, fit_holds, severities, end_slopes)

    flat_b, conv_b = run_stage_b(fit_holds, nadir_info, severities, end_slopes, frozen_sensor)

    phys_names_b = ["pvo2", "k_co2", "gamma"]
    phys_bounds_b = [(15, 50), (0.02, 0.25), GAMMA_BOUNDS]
    print(f"\n  Stage B physiology params (sensor frozen from Stage A):")
    for name, val, (lo, hi) in zip(phys_names_b, flat_b[:3], phys_bounds_b):
        flag = " ** AT BOUND **" if is_at_bound(val, lo, hi) else ""
        print(f"    {name:>12s} = {val:8.4f}  [{lo:>6.2f}, {hi:>6.2f}]{flag}")

    n_phys_b = 3
    delta_offset_b = n_phys_b
    ic_offset_b = delta_offset_b + n_holds
    deltas_b = flat_b[delta_offset_b:delta_offset_b + n_holds]

    print(f"\n  Stage B -- Per-hold shifts:")
    for i, h in enumerate(fit_holds):
        sys_g = beta_g_a * severities[h["id"]]
        sys_s = beta_s_a * (-end_slopes[h["id"]])
        residual = deltas_b[i]
        total_shift = sys_g + sys_s + residual
        eff = max(tau_0 + total_shift, 1.0)
        print(f"    {h['type']}#{h['id']}: bg*g={sys_g:+6.2f}, bs*(-s)={sys_s:+6.2f}, "
              f"delta={residual:+6.2f}, eff_lag={eff:6.2f}")

    # Print per-hold ICs
    print(f"\n  Stage B -- Per-hold ICs:")
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

    eval_b = evaluate_stage_b(
        flat_b, fit_holds, nadir_info, severities, end_slopes, frozen_sensor, all_holds)

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

    loho_results = run_stage_a_loho(fit_holds, nadir_info, severities, end_slopes)

    print(f"\n  LOHO Summary:")
    print(f"  {'Hold':<15s} | {'R2':>8s} | {'RMSE':>8s} | {'NadirErr':>8s} | {'EffLag':>8s}")
    print(f"  {'-'*15}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for r in loho_results:
        r2_str = f"{r['r2']:.4f}" if r["r2"] is not None else "N/A"
        rmse_str = f"{r['rmse']:.2f}" if r["rmse"] is not None else "N/A"
        print(f"  {r['hold_type']}#{r['hold_id']:<10d} | {r2_str:>8s} | {rmse_str:>8s} | "
              f"{r['nadir_err']:+8.1f} | {r['eff_lag']:8.1f}")

    loho_nerrs = [abs(r["nadir_err"]) for r in loho_results]
    avg_loho_nerr = np.mean(loho_nerrs)
    print(f"\n  Avg LOHO |timing error|: {avg_loho_nerr:.1f}s (target < 5s)")

    # ══════════════════════════════════════════════════════════════════════════
    # TIME-SPLIT (Stage B)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("TIME-SPLIT: Fit Early Apnea, Predict Late (Stage B)")
    print(f"{'='*120}")

    ts_results = run_stage_b_time_split(
        fit_holds, nadir_info, severities, end_slopes, frozen_sensor)

    r2_preds = [r["r2_pred"] for r in ts_results if r["r2_pred"] is not None]
    avg_r2_pred = np.mean(r2_preds) if r2_preds else float("nan")
    print(f"\n  Avg time-split R2_pred: {avg_r2_pred:.4f} (target > 0.90)")

    # ══════════════════════════════════════════════════════════════════════════
    # PROFILE LIKELIHOOD (tau_0)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("PROFILE LIKELIHOOD: tau_0 sweep (Stage A)")
    print(f"{'='*120}")

    tau_0_values = [5, 8, 10, 12, 14, 16, 18, 20, 22, 25, 30, 35, 40, 45]
    profile_results = run_profile_likelihood(
        fit_holds, nadir_info, severities, end_slopes, tau_0_values)

    print(f"\n  {'tau_0':>8s} | {'loss':>10s} | {'r_offset':>8s} | {'b_s':>8s} | "
          f"{'beta_g':>8s} | {'beta_s':>8s}")
    print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for tau_val in sorted(profile_results.keys()):
        r = profile_results[tau_val]
        s = r["flat"]
        tau_0_v, r_off, b_s, bg, bs = s[:5]
        print(f"  {tau_val:8.1f} | {r['loss']:10.2f} | {r_off:8.4f} | {b_s:8.4f} | "
              f"{bg:8.4f} | {bs:8.4f}")

    losses = [profile_results[t]["loss"] for t in sorted(profile_results.keys())]
    is_monotone_dec = all(losses[i] >= losses[i + 1] for i in range(len(losses) - 1))
    is_monotone_inc = all(losses[i] <= losses[i + 1] for i in range(len(losses) - 1))
    is_monotone = is_monotone_dec or is_monotone_inc
    sorted_taus = sorted(profile_results.keys())
    min_tau = sorted_taus[np.argmin(losses)]
    print(f"\n  Non-monotone: {'YES (good!)' if not is_monotone else 'NO (degenerate)'}")
    print(f"  Minimum loss at tau_0={min_tau:.1f}")

    # ══════════════════════════════════════════════════════════════════════════
    # SENSITIVITY: d(physiology)/d(tau_0)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("SENSITIVITY: d(physiology params)/d(tau_0)")
    print(f"{'='*120}")

    sensitivities = compute_sensitivity(
        profile_results, fit_holds, nadir_info, severities, end_slopes, frozen_sensor)

    # ══════════════════════════════════════════════════════════════════════════
    # SPONGE DIAGNOSTICS
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("SPONGE DIAGNOSTICS")
    print(f"{'='*120}")

    # Stage A sponge
    a_param_names = ["tau_0", "r_offset", "b_s", "beta_g", "beta_s"]
    a_bounds = [TAU0_BOUNDS, OFFSET_BOUNDS, GAIN_BOUNDS, BETA_G_BOUNDS, BETA_S_BOUNDS]
    a_prior_sigmas = {
        "tau_0": TAU0_PRIOR_CENTER * 0.4,  # LogNormal sigma
        "r_offset": 2.0,
        "b_s": 0.02,
        "beta_g": 2.0,
        "beta_s": 2.0,
    }
    at_bound_a = sponge_diagnostics(
        flat_a, a_bounds, a_prior_sigmas, a_param_names, label="Stage A")

    # Stage B sponge
    b_param_names = ["pvo2", "k_co2", "gamma"]
    b_bounds = [(15, 50), (0.02, 0.25), GAMMA_BOUNDS]
    b_prior_sigmas = {
        "pvo2": 5.0,
        "k_co2": 0.02,
        "gamma": 0.1,
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
                        output_dir / "exp_v7_01_stage_a.png")

    plot_stage_b_detail(eval_b, all_holds, nadir_info, frozen_sensor,
                        output_dir / "exp_v7_01_stage_b.png")

    plot_loho_summary(loho_results, output_dir / "exp_v7_01_loho.png")

    plot_profile_likelihood(profile_results, output_dir / "exp_v7_01_profile.png")

    plot_sensitivity(sensitivities, output_dir / "exp_v7_01_sensitivity.png")

    # ── Success criteria ─────────────────────────────────────────────────────
    print(f"\n{'='*120}")
    print("SUCCESS CRITERIA CHECK")
    print(f"{'='*120}")

    # 1. Stage A timing error
    nerrs_a = [abs(r["nadir_err"]) for r in eval_a if not r["is_excluded"]]
    avg_nerr_a = np.mean(nerrs_a) if nerrs_a else float("nan")
    print(f"\n  1. Stage A timing error avg={avg_nerr_a:.1f}s (target < 1s, v6.07 Exp B: 0.4s): "
          f"{'PASS' if avg_nerr_a < 1 else 'FAIL'}")

    # 2. LOHO timing error
    print(f"  2. LOHO timing error avg={avg_loho_nerr:.1f}s (target < 5s): "
          f"{'PASS' if avg_loho_nerr < 5 else 'FAIL'}")

    # 3. Stage B R2 on apnea
    r2_apnea_b = [r["r2_apnea"] for r in eval_b
                  if not r["is_excluded"] and r["r2_apnea"] is not None]
    avg_r2a_b = np.mean(r2_apnea_b) if r2_apnea_b else float("nan")
    print(f"  3. Stage B R2(apnea) avg={avg_r2a_b:.4f} (target >= 0.95, v6.07 Exp C: 0.961): "
          f"{'PASS' if avg_r2a_b >= 0.95 else 'FAIL'}")

    # 4. Time-split R2
    print(f"  4. Time-split R2_pred avg={avg_r2_pred:.4f} (target > 0.90): "
          f"{'PASS' if avg_r2_pred > 0.90 else 'FAIL'}")

    # 5. Gamma interior
    gamma_b = flat_b[2]
    gamma_interior = not is_at_bound(gamma_b, *GAMMA_BOUNDS)
    print(f"  5. gamma={gamma_b:.4f} interior of [{GAMMA_BOUNDS[0]}, {GAMMA_BOUNDS[1]}]: "
          f"{'PASS' if gamma_interior else 'FAIL (at bound)'}")

    # 6. Profile: clear interior minimum
    print(f"  6. Profile non-monotone: {'PASS' if not is_monotone else 'FAIL (monotone)'}")
    print(f"     Minimum at tau_0={min_tau:.1f}")

    # 7. Params at bounds (fewer than v6.07)
    total_at_bound = at_bound_a + at_bound_b
    print(f"  7. Params at bounds: {total_at_bound} "
          f"(v6.07 had cv=0.10, tau_reoxy=30 — both now removed)")

    # 8. Gain b_s near 1
    print(f"  8. Gain b_s={b_s_a:.4f} (should be near 1.0): "
          f"{'PASS' if abs(b_s_a - 1.0) < 0.1 else 'MARGINAL'}")

    print(f"\n{'='*120}")
    print("DONE")
    print(f"{'='*120}")


if __name__ == "__main__":
    main()
