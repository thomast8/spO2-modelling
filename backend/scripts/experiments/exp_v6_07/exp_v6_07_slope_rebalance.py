"""
v6 Experiment 7: Slope Covariate + Rebalanced Sensor-Only + Apnea-Only Physiology.

v6.06 established: (1) smooth gamma kernel works (cv continuous, no cap artifact),
(2) beta > 0 is consistent (deeper desat -> longer sensor lag), (3) sensor-only
identification is promising but under-constrained.

v6.06 failures:
  - Depth severity alone can't distinguish RV#4 from FRC#5 (both g_h=5.4, eff lags 8s vs 37s)
  - Exp C penalty balance broken: lambda_shrink=300 makes delta 30x more expensive than beta,
    forcing beta to saturation (2.0, at bound), cv->0.10 (at bound), no timing penalty

Three targeted fixes:
  1. End-slope covariate: s_h = (SpO2(t_end) - SpO2(t_end-10))/10, separates RV#4 from FRC#5
  2. Rebalanced sensor-only: Huber timing penalty, 10x less delta shrinkage, stronger beta priors
  3. Relaxed latent min + apnea-only physiology: per-hold nadir shift m_h, fit on [0, t_end+5]

Sub-experiments:
  A: Co-trained + two covariates (24 params) — v6.06 Exp B + slope covariate
  B: Sensor-only, rebalanced (55 params) — all fixes applied
  C: Frozen sensor + apnea-only physiology (19 params) — loss on [0, t_end+5] only

Usage:
    cd backend && uv run python -u scripts/exp_v6_07_slope_rebalance.py
"""

import csv
import io
import sqlite3
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import differential_evolution
from scipy.special import gammainc


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

DB_PATH = Path(__file__).resolve().parents[4] / "data" / "spo2.db"

P50_BASE = 26.6
P_EQ = 100.0
PACO2_NORMAL = 40.0
TAU_CLEAR_FIXED = 30.0
FIO2_PB_PH2O = 149.2  # FiO2 * (PB - PH2O) = 0.2093 * (760 - 47)
RQ = 0.8

EXCLUDED_IDS = {1}  # FL#1 excluded (only 2% SpO2 variation)

# ── Regularization strengths ─────────────────────────────────────────────────

LAMBDA_REG = 10.0       # per-hold IC -> type-mean
LAMBDA_NADIR = 1000.0   # nadir timing penalty per hold (Exp A: squared)
LAMBDA_K_CO2 = 2000.0   # k_co2 prior toward 0.06
LAMBDA_PACO2 = 1000.0   # paco2_0 prior toward 40
LAMBDA_GAMMA = 2000.0   # gamma prior toward 1.0
LAMBDA_MEAN_LAG = 2000.0  # mean_lag prior toward center
LAMBDA_R_OFFSET = 500.0 # r_offset prior toward 0
LAMBDA_SPLINE_SMOOTH = 50.0  # spline 2nd-difference penalty

# Exp A penalties (co-trained, same as v6.06 Exp B)
LAMBDA_SHRINK_A = 300.0  # per-hold delta shrinkage (co-trained)
LAMBDA_BETA_G_A = 200.0  # beta_g prior (co-trained)
LAMBDA_BETA_S_A = 200.0  # beta_s prior (co-trained)
LAMBDA_ZEROSUM_A = 500.0 # zero-sum on deltas (co-trained)

# Exp B penalties (sensor-only, REBALANCED from v6.06)
LAMBDA_SHRINK_B = 30.0   # delta shrinkage: 10x less than v6.06's 300
LAMBDA_BETA_G_B = 500.0  # beta_g prior: stronger to prevent saturation
LAMBDA_BETA_S_B = 500.0  # beta_s prior: same treatment
LAMBDA_NADIR_B = 500.0   # Huber timing penalty (was ABSENT in v6.06 Exp C)
LAMBDA_ZEROSUM_B = 500.0 # zero-sum on deltas
LAMBDA_M = 20.0          # soft prior on min_shift (~sigma_m = 5s)

MEAN_LAG_PRIOR_CENTER = 15.5  # v6.03 d finding
NADIR_WINDOW_AFTER = 45  # seconds after t_end for loss window

# ── Bounds ───────────────────────────────────────────────────────────────────

# Per-hold ICs: tau_washout, paco2_0
PERHOLD_BOUNDS = {
    "FL": [(50, 250), (20, 50)],
    "FRC": [(20, 100), (25, 50)],
    "RV": [(10, 80), (30, 55)],
}
PERHOLD_NAMES = ["tau_washout", "paco2_0"]
N_PH = len(PERHOLD_NAMES)

# Shared params: pvo2, k_co2, r_offset, tau_reoxy, mean_lag, cv, gamma
SHARED_BOUNDS = [
    (15, 50),      # pvo2
    (0.02, 0.25),  # k_co2
    (-8, 8),       # r_offset
    (5, 30),       # tau_reoxy
    (5, 45),       # mean_lag
    (0.10, 1.2),   # cv
    (0.8, 3.0),    # gamma
]
SHARED_NAMES = ["pvo2", "k_co2", "r_offset", "tau_reoxy", "mean_lag", "cv", "gamma"]
N_SHARED = len(SHARED_BOUNDS)

DELTA_BOUNDS_A = (-15, 15)   # Exp A residual shifts
DELTA_BOUNDS_B = (-15, 15)   # Exp B residual shifts
BETA_G_BOUNDS = (-2.0, 2.0)  # severity coefficient
BETA_S_BOUNDS = (-2.0, 2.0)  # slope coefficient
MIN_SHIFT_BOUNDS = (-8, 8)   # per-hold nadir shift (Exp B)


# ── Data loading (reused from v6.06) ─────────────────────────────────────────


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


# ── Physiology functions (reused from v6.06) ─────────────────────────────────


def corrected_pao2_0(paco2_0, aa):
    """Derive initial PaO2 from PaCO2 and A-a gradient (corrected sign)."""
    return max(FIO2_PB_PH2O - paco2_0 / RQ - aa, 1.0)


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


# ── Smooth Discrete Gamma Kernel (from v6.06) ────────────────────────────────


def gamma_kernel_smooth(mean_lag, cv, max_support=120):
    """Smooth discrete gamma kernel via CDF bin integration.

    h[i] = P(i <= X < i+1) where X ~ Gamma(k, theta).
    Uses scipy.special.gammainc (regularized lower incomplete gamma) for speed.
    Smooth in both mean_lag and cv (no rounding, no cap).
    """
    k = 1.0 / (cv * cv)           # continuous shape
    theta = mean_lag * cv * cv     # scale
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


# ── Core predict function (smooth gamma kernel) ──────────────────────────────


def predict_v6(t, pvo2, tau_washout, gamma, paco2_0, k_co2, r_offset,
               tau_reoxy, mean_lag, cv, t_end, shift=0.0):
    """Full sensor pipeline with smooth gamma kernel."""
    aa = 0.0
    pao2_0 = corrected_pao2_0(paco2_0, aa)
    pao2 = pao2_with_exp_recovery(t, pao2_0, pvo2, tau_washout, tau_reoxy, t_end)
    p50 = p50_with_exp_recovery(t, paco2_0, k_co2, TAU_CLEAR_FIXED, t_end)
    sa = odc_severinghaus(pao2, p50, gamma)

    eff_mean_lag = max(mean_lag + shift, 1.0)
    filtered = apply_gamma_kernel(sa, eff_mean_lag, cv)

    return np.clip(filtered + r_offset, 0.0, 100.0)


# ── Nadir + loss helpers ─────────────────────────────────────────────────────


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


def nadir_timing_penalty_squared(t, pred, t_nadir_obs, lam=LAMBDA_NADIR):
    """Squared nadir timing penalty (used in Exp A, co-trained)."""
    t_nadir_pred = t[np.argmin(pred)]
    return lam * (t_nadir_pred - t_nadir_obs) ** 2


def huber_loss(a, delta=8.0):
    """Huber loss: quadratic for |a| <= delta, linear beyond.

    delta=8s: errors <= 8s penalized quadratically, beyond that linearly.
    Prevents outliers (e.g. RV#4 38s error in v6.06) from dominating.
    """
    abs_a = np.abs(a)
    return np.where(abs_a <= delta, 0.5 * a**2, delta * (abs_a - 0.5 * delta))


def nadir_timing_penalty_huber(t, pred, t_nadir_obs, lam=LAMBDA_NADIR_B, huber_delta=8.0):
    """Huber nadir timing penalty (used in Exp B, sensor-only)."""
    t_nadir_pred = t[np.argmin(pred)]
    err = t_nadir_pred - t_nadir_obs
    return lam * float(huber_loss(err, delta=huber_delta))


# ── Metrics ──────────────────────────────────────────────────────────────────


def compute_r2(obs, pred):
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def compute_rmse(obs, pred):
    return float(np.sqrt(np.mean((obs - pred) ** 2)))


def is_at_bound(val, lo, hi, tol=1e-3):
    return abs(val - lo) < tol or abs(val - hi) < tol


# ── Common penalty terms ─────────────────────────────────────────────────────


def shared_priors(k_co2, gamma, mean_lag, r_offset):
    """CO2 prior + gamma prior + mean_lag prior + r_offset prior."""
    return (
        LAMBDA_K_CO2 * (k_co2 - 0.06) ** 2
        + LAMBDA_GAMMA * (gamma - 1.0) ** 2
        + LAMBDA_MEAN_LAG * (mean_lag - MEAN_LAG_PRIOR_CENTER) ** 2
        + LAMBDA_R_OFFSET * r_offset ** 2
    )


def ic_regularization(flat, n_shared, fit_holds, type_groups, ic_offset=None):
    """Per-hold IC regularization toward type means."""
    if ic_offset is None:
        ic_offset = n_shared
    total = 0.0
    for ht, indices in type_groups.items():
        if len(indices) < 2:
            continue
        for p_off in range(N_PH):
            values = [flat[ic_offset + idx * N_PH + p_off] for idx in indices]
            mean_val = np.mean(values)
            total += LAMBDA_REG * sum((v - mean_val) ** 2 for v in values)
    return total


# ── Depth severity + End-slope covariate ─────────────────────────────────────


def compute_depth_severity(fit_holds):
    """g_h = max(0, 95 - min(SpO2_obs)) / 10 for each hold."""
    severities = {}
    for h in fit_holds:
        min_spo2 = float(np.min(h["spo2"]))
        severities[h["id"]] = max(0.0, 95.0 - min_spo2) / 10.0
    return severities


def compute_end_slope(fit_holds, window=10):
    """s_h = (SpO2(t_end) - SpO2(t_end - window)) / window from observed data.

    Negative when SpO2 is falling at end of apnea.
    Separates RV#4 (still falling fast) from FRC#5 (plateau-ish).
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


# ── Latent spline infrastructure ─────────────────────────────────────────────


def build_latent_spline(t_1hz, t_end, spline_params, min_shift=0.0):
    """Build monotone latent SaO2 curve from spline parameters.

    Parameterization via decrements/increments ensures monotonicity by construction:
      - x_0: initial value (near 98)
      - 4 decrements d_i >= 0: monotone decrease during apnea
      - 3 increments r_i >= 0: monotone increase during recovery
      - Minimum is at t_end + min_shift (min_shift allows nadir shift)

    Knots: [0, t_end/4, t_end/2, 3*t_end/4, t_end+m_h, t_end+m_h+10, t_end+m_h+30, t_end+m_h+45]
    """
    x_0 = spline_params[0]
    decrements = spline_params[1:5]  # 4 decrements for apnea
    increments = spline_params[5:8]  # 3 increments for recovery

    # Nadir point shifts with min_shift
    t_nadir = t_end + min_shift

    # Build knot times
    knot_times = np.array([
        0.0,
        t_end / 4.0,
        t_end / 2.0,
        3.0 * t_end / 4.0,
        t_nadir,
        t_nadir + 10.0,
        t_nadir + 30.0,
        t_nadir + 45.0,
    ])

    # Ensure monotonically increasing knot times (min_shift could cause issues)
    for j in range(1, len(knot_times)):
        if knot_times[j] <= knot_times[j - 1]:
            knot_times[j] = knot_times[j - 1] + 0.5

    # Build knot values: cumulative decrements then increments
    values = np.zeros(8)
    values[0] = x_0
    for j in range(4):
        values[j + 1] = values[j] - decrements[j]
    for j in range(3):
        values[5 + j] = values[4 + j] + increments[j]

    # Clip to physiological range
    values = np.clip(values, 0.0, 100.0)

    # PCHIP interpolation to 1-Hz grid
    interp = PchipInterpolator(knot_times, values)
    latent = interp(t_1hz)
    return np.clip(latent, 0.0, 100.0)


# ── Exp A: Co-trained + two covariates (24 params) ──────────────────────────
# v6.06 Exp B + slope covariate. Tests whether second covariate helps timing.


def run_exp_a(fit_holds, nadir_info, severities, end_slopes):
    """Exp A: Co-trained with two covariates (beta_g, beta_s).

    effective_lag_h = mean_lag + beta_g * g_h + beta_s * (-s_h) + delta_h
    -s_h positive when SpO2 falling (steeper fall -> expect earlier nadir -> beta_s < 0 expected).

    Shared (7): pvo2, k_co2, r_offset, tau_reoxy, mean_lag, cv, gamma
    Beta_g (1): depth severity coefficient
    Beta_s (1): end-slope coefficient
    Deltas (5): residual per-hold shifts with Sigma delta=0
    Per-hold (2x5=10): tau_washout_i, paco2_0_i
    Total: 24 free params.
    """
    n_holds = len(fit_holds)
    bounds = list(SHARED_BOUNDS)
    bounds.append(BETA_G_BOUNDS)         # beta_g
    bounds.append(BETA_S_BOUNDS)         # beta_s
    for _ in fit_holds:
        bounds.append(DELTA_BOUNDS_A)    # delta_h
    for h in fit_holds:
        bounds.extend(PERHOLD_BOUNDS[h["type"]])
    n_total = len(bounds)

    type_groups = {}
    for i, h in enumerate(fit_holds):
        type_groups.setdefault(h["type"], []).append(i)

    beta_g_idx = N_SHARED
    beta_s_idx = N_SHARED + 1
    delta_offset = N_SHARED + 2
    ic_offset = delta_offset + n_holds

    masks = [nadir_window_mask(h["t"], h["t_end"]) for h in fit_holds]
    weights = [np.where(h["spo2"][m] < 95, 3.0, 1.0) for h, m in zip(fit_holds, masks)]
    nadir_ts = [nadir_info[h["id"]]["t_nadir"] for h in fit_holds]
    g_h = [severities[h["id"]] for h in fit_holds]
    neg_s_h = [-end_slopes[h["id"]] for h in fit_holds]  # negate: positive when falling

    print(f"\n  Exp A: {n_total} params ({N_SHARED} shared + 2 beta + {n_holds} delta + "
          f"{N_PH}x{n_holds} per-hold ICs)")
    print(f"  Lag model: eff_lag = mean_lag + beta_g*g_h + beta_s*(-s_h) + delta_h")
    print(f"  Severity g_h: {[f'{g:.2f}' for g in g_h]}")
    print(f"  End-slope -s_h: {[f'{s:.3f}' for s in neg_s_h]}")
    print(f"  lambda_shrink={LAMBDA_SHRINK_A}, lambda_beta_g={LAMBDA_BETA_G_A}, "
          f"lambda_beta_s={LAMBDA_BETA_S_A}, lambda_zerosum={LAMBDA_ZEROSUM_A}")

    def objective(flat):
        pvo2, k_co2, r_offset, tau_reoxy, mean_lag, cv, gamma_val = flat[:N_SHARED]
        beta_g = flat[beta_g_idx]
        beta_s = flat[beta_s_idx]
        deltas = flat[delta_offset : delta_offset + n_holds]
        total = 0.0

        for i, h in enumerate(fit_holds):
            ph_offset = ic_offset + i * N_PH
            tau_washout, paco2_0 = flat[ph_offset : ph_offset + N_PH]

            shift = beta_g * g_h[i] + beta_s * neg_s_h[i] + deltas[i]
            pred = predict_v6(
                h["t"], pvo2, tau_washout, gamma_val,
                paco2_0, k_co2, r_offset, tau_reoxy, mean_lag, cv, h["t_end"],
                shift=shift,
            )
            m = masks[i]
            total += np.sum(weights[i] * (h["spo2"][m] - pred[m]) ** 2)

            total += nadir_timing_penalty_squared(h["t"], pred, nadir_ts[i])
            total += LAMBDA_PACO2 * (paco2_0 - 40.0) ** 2

        # Structural shift penalties
        total += LAMBDA_SHRINK_A * np.sum(deltas**2)
        total += LAMBDA_ZEROSUM_A * np.sum(deltas) ** 2
        total += LAMBDA_BETA_G_A * beta_g ** 2
        total += LAMBDA_BETA_S_A * beta_s ** 2

        total += shared_priors(k_co2, gamma_val, mean_lag, r_offset)
        total += ic_regularization(flat, N_SHARED, fit_holds, type_groups, ic_offset=ic_offset)

        return total

    maxiter_a = 3000
    res = differential_evolution(
        objective, bounds, maxiter=maxiter_a, seed=42, tol=1e-10,
        polish=True, popsize=30, mutation=(0.5, 1.5), recombination=0.9,
        callback=make_de_callback("Exp A", maxiter_a),
    )
    print(f"\n  Converged: {res.success}, fun={res.fun:.2f}, nfev={res.nfev}")
    return res.x, res.success


# ── Exp B: Sensor-only, rebalanced (55 params) ──────────────────────────────
# All three fixes: end-slope, Huber timing, relaxed latent min, rebalanced penalties.


def run_exp_b(fit_holds, nadir_info, severities, end_slopes):
    """Exp B: Sensor-only identification with all v6.07 fixes.

    Kernel (5): mean_lag, cv, r_offset, beta_g, beta_s
    Deltas (5): residual per-hold shifts with Sigma delta=0
    Min shifts (5): m_h per-hold nadir shifts
    Splines (8x5=40): 8 params per hold
    Total: 55 free params.
    """
    n_holds = len(fit_holds)

    # Build bounds
    bounds = [
        (5, 45),       # mean_lag
        (0.10, 1.2),   # cv
        (-8, 8),       # r_offset
        (-2.0, 2.0),   # beta_g (severity coefficient)
        (-2.0, 2.0),   # beta_s (slope coefficient)
    ]
    n_kernel = len(bounds)

    # Per-hold deltas
    for _ in fit_holds:
        bounds.append(DELTA_BOUNDS_B)
    delta_offset = n_kernel

    # Per-hold min_shift (nadir shift)
    for _ in fit_holds:
        bounds.append(MIN_SHIFT_BOUNDS)
    min_shift_offset = delta_offset + n_holds

    # Per-hold spline params: x_0, 4 decrements, 3 increments
    spline_offset = min_shift_offset + n_holds
    for _ in fit_holds:
        bounds.append((85, 100))  # x_0
        for _ in range(4):
            bounds.append((0, 20))   # decrements
        for _ in range(3):
            bounds.append((0, 30))   # increments
    n_spline_per_hold = 8

    n_total = len(bounds)
    g_h = [severities[h["id"]] for h in fit_holds]
    neg_s_h = [-end_slopes[h["id"]] for h in fit_holds]

    print(f"\n  Exp B: {n_total} params ({n_kernel} kernel + {n_holds} delta + "
          f"{n_holds} min_shift + {n_spline_per_hold}x{n_holds} spline)")
    print(f"  Sensor-only, rebalanced: Huber timing, reduced delta shrinkage, min_shift")
    print(f"  lambda_shrink={LAMBDA_SHRINK_B}, lambda_beta_g={LAMBDA_BETA_G_B}, "
          f"lambda_nadir={LAMBDA_NADIR_B}, lambda_m={LAMBDA_M}")

    masks = [nadir_window_mask(h["t"], h["t_end"]) for h in fit_holds]
    weights = [np.where(h["spo2"][m] < 95, 3.0, 1.0) for h, m in zip(fit_holds, masks)]
    nadir_ts = [nadir_info[h["id"]]["t_nadir"] for h in fit_holds]

    # Precompute 1-Hz time grids
    t_grids = []
    for h in fit_holds:
        t_max = h["t"][-1]
        t_1hz = np.arange(0, t_max + 1, 1.0)
        t_grids.append(t_1hz)

    def objective(flat):
        mean_lag, cv, r_offset, beta_g, beta_s = flat[:n_kernel]
        deltas = flat[delta_offset : delta_offset + n_holds]
        min_shifts = flat[min_shift_offset : min_shift_offset + n_holds]
        total = 0.0

        for i, h in enumerate(fit_holds):
            # Extract spline params
            sp_start = spline_offset + i * n_spline_per_hold
            spline_params = flat[sp_start : sp_start + n_spline_per_hold]

            # Build latent curve with nadir shift
            t_1hz = t_grids[i]
            latent = build_latent_spline(t_1hz, h["t_end"], spline_params,
                                         min_shift=min_shifts[i])

            # Apply gamma kernel with shift
            shift = beta_g * g_h[i] + beta_s * neg_s_h[i] + deltas[i]
            eff_mean_lag = max(mean_lag + shift, 1.0)
            filtered = apply_gamma_kernel(latent, eff_mean_lag, cv)

            # Interpolate to observation times
            pred_at_obs = np.interp(h["t"], t_1hz, filtered + r_offset)
            pred_at_obs = np.clip(pred_at_obs, 0.0, 100.0)

            # Loss on nadir window
            m = masks[i]
            total += np.sum(weights[i] * (h["spo2"][m] - pred_at_obs[m]) ** 2)

            # Huber timing penalty (robust to outliers)
            total += nadir_timing_penalty_huber(h["t"][m], pred_at_obs[m], nadir_ts[i])

            # Spline smoothness: 2nd differences of knot values
            values = np.zeros(8)
            values[0] = spline_params[0]
            for j in range(4):
                values[j + 1] = values[j] - spline_params[1 + j]
            for j in range(3):
                values[5 + j] = values[4 + j] + spline_params[5 + j]
            second_diff = np.diff(values, n=2)
            total += LAMBDA_SPLINE_SMOOTH * np.sum(second_diff ** 2)

        # Kernel priors
        total += LAMBDA_MEAN_LAG * (mean_lag - MEAN_LAG_PRIOR_CENTER) ** 2
        total += LAMBDA_R_OFFSET * r_offset ** 2

        # Structural shift penalties (REBALANCED)
        total += LAMBDA_SHRINK_B * np.sum(deltas**2)
        total += LAMBDA_ZEROSUM_B * np.sum(deltas) ** 2
        total += LAMBDA_BETA_G_B * beta_g ** 2
        total += LAMBDA_BETA_S_B * beta_s ** 2

        # Min-shift prior
        total += LAMBDA_M * np.sum(min_shifts ** 2)

        return total

    maxiter_b = 5000
    res = differential_evolution(
        objective, bounds, maxiter=maxiter_b, seed=42, tol=1e-10,
        polish=True, popsize=40, mutation=(0.5, 1.5), recombination=0.9,
        callback=make_de_callback("Exp B", maxiter_b),
    )
    print(f"\n  Converged: {res.success}, fun={res.fun:.2f}, nfev={res.nfev}")
    return res.x, res.success


# ── Exp C: Frozen sensor + apnea-only physiology (19 params) ─────────────────


def run_exp_c(fit_holds, nadir_info, severities, end_slopes, frozen_kernel):
    """Exp C: Physiology model with frozen sensor kernel from Exp B.

    Freeze: mean_lag, cv, r_offset, beta_g, beta_s from Exp B.
    Fit: physiology on apnea window only [0, t_end+5] (no recovery).

    Physiology (4): pvo2, k_co2, tau_reoxy, gamma
    Deltas (5): residual per-hold shifts with Sigma delta=0
    Per-hold (2x5=10): tau_washout_i, paco2_0_i
    Total: 19 free params.
    """
    mean_lag_frozen = frozen_kernel["mean_lag"]
    cv_frozen = frozen_kernel["cv"]
    r_offset_frozen = frozen_kernel["r_offset"]
    beta_g_frozen = frozen_kernel["beta_g"]
    beta_s_frozen = frozen_kernel["beta_s"]

    n_holds = len(fit_holds)

    # Physiology shared: pvo2, k_co2, tau_reoxy, gamma
    bounds = [
        (15, 50),      # pvo2
        (0.02, 0.25),  # k_co2
        (5, 30),       # tau_reoxy
        (0.8, 3.0),    # gamma
    ]
    phys_names = ["pvo2", "k_co2", "tau_reoxy", "gamma"]
    n_phys = len(bounds)

    # Per-hold deltas
    for _ in fit_holds:
        bounds.append(DELTA_BOUNDS_B)
    delta_offset = n_phys
    ic_offset = delta_offset + n_holds

    # Per-hold ICs
    for h in fit_holds:
        bounds.extend(PERHOLD_BOUNDS[h["type"]])
    n_total = len(bounds)

    type_groups = {}
    for i, h in enumerate(fit_holds):
        type_groups.setdefault(h["type"], []).append(i)

    g_h = [severities[h["id"]] for h in fit_holds]
    neg_s_h = [-end_slopes[h["id"]] for h in fit_holds]
    nadir_ts = [nadir_info[h["id"]]["t_nadir"] for h in fit_holds]

    # Apnea-only loss window: [0, t_end+5] (exclude recovery)
    apnea_window = 5  # seconds past t_end to include
    masks = [h["t"] <= h["t_end"] + apnea_window for h in fit_holds]
    weights = [np.where(h["spo2"][m] < 95, 3.0, 1.0) for h, m in zip(fit_holds, masks)]

    print(f"\n  Exp C: {n_total} params ({n_phys} physiology + {n_holds} delta + "
          f"{N_PH}x{n_holds} per-hold ICs)")
    print(f"  Frozen sensor: mean_lag={mean_lag_frozen:.2f}, cv={cv_frozen:.3f}, "
          f"r_offset={r_offset_frozen:.2f}, beta_g={beta_g_frozen:.3f}, "
          f"beta_s={beta_s_frozen:.3f}")
    print(f"  Apnea-only loss: window [0, t_end+{apnea_window}s] (no recovery)")

    def objective(flat):
        pvo2, k_co2, tau_reoxy, gamma_val = flat[:n_phys]
        deltas = flat[delta_offset : delta_offset + n_holds]
        total = 0.0

        for i, h in enumerate(fit_holds):
            ph_offset = ic_offset + i * N_PH
            tau_washout, paco2_0 = flat[ph_offset : ph_offset + N_PH]

            shift = (beta_g_frozen * g_h[i] + beta_s_frozen * neg_s_h[i]
                     + deltas[i])
            pred = predict_v6(
                h["t"], pvo2, tau_washout, gamma_val,
                paco2_0, k_co2, r_offset_frozen, tau_reoxy,
                mean_lag_frozen, cv_frozen, h["t_end"],
                shift=shift,
            )
            m = masks[i]
            total += np.sum(weights[i] * (h["spo2"][m] - pred[m]) ** 2)

            total += nadir_timing_penalty_squared(h["t"], pred, nadir_ts[i])
            total += LAMBDA_PACO2 * (paco2_0 - 40.0) ** 2

        # Priors on physiology params
        total += LAMBDA_K_CO2 * (k_co2 - 0.06) ** 2
        total += LAMBDA_GAMMA * (gamma_val - 1.0) ** 2

        # Structural shift penalties
        total += LAMBDA_SHRINK_B * np.sum(deltas**2)
        total += LAMBDA_ZEROSUM_B * np.sum(deltas) ** 2

        total += ic_regularization(flat, n_phys, fit_holds, type_groups, ic_offset=ic_offset)

        return total

    maxiter_c = 3000
    res = differential_evolution(
        objective, bounds, maxiter=maxiter_c, seed=42, tol=1e-10,
        polish=True, popsize=30, mutation=(0.5, 1.5), recombination=0.9,
        callback=make_de_callback("Exp C", maxiter_c),
    )
    print(f"\n  Converged: {res.success}, fun={res.fun:.2f}, nfev={res.nfev}")
    return res.x, res.success


# ── Mean-lag profile sweep ───────────────────────────────────────────────────


def run_mean_lag_profile(fit_holds, nadir_info, severities, end_slopes, mean_lag_values):
    """Fix mean_lag at each value, re-optimize using Exp A structure (two covariates)."""
    mean_lag_idx = 4  # mean_lag is 5th shared param (0-indexed)

    type_groups = {}
    for i, h in enumerate(fit_holds):
        type_groups.setdefault(h["type"], []).append(i)

    n_holds = len(fit_holds)
    masks = [nadir_window_mask(h["t"], h["t_end"]) for h in fit_holds]
    weights = [np.where(h["spo2"][m] < 95, 3.0, 1.0) for h, m in zip(fit_holds, masks)]
    nadir_ts = [nadir_info[h["id"]]["t_nadir"] for h in fit_holds]
    g_h = [severities[h["id"]] for h in fit_holds]
    neg_s_h = [-end_slopes[h["id"]] for h in fit_holds]

    results = {}
    for ml_fixed in mean_lag_values:
        fixed_bounds = list(SHARED_BOUNDS)
        fixed_bounds[mean_lag_idx] = (ml_fixed - 0.01, ml_fixed + 0.01)
        bounds = list(fixed_bounds)
        bounds.append(BETA_G_BOUNDS)      # beta_g
        bounds.append(BETA_S_BOUNDS)      # beta_s
        for _ in fit_holds:
            bounds.append(DELTA_BOUNDS_A)
        for h in fit_holds:
            bounds.extend(PERHOLD_BOUNDS[h["type"]])

        beta_g_idx = N_SHARED
        beta_s_idx = N_SHARED + 1
        delta_offset = N_SHARED + 2
        ic_offset = delta_offset + n_holds

        def objective(flat, _tg=type_groups, _masks=masks, _weights=weights,
                      _nadir=nadir_ts, _g_h=g_h, _neg_s_h=neg_s_h,
                      _bg_idx=beta_g_idx, _bs_idx=beta_s_idx,
                      _d_off=delta_offset, _ic_off=ic_offset):
            pvo2, k_co2, r_offset, tau_reoxy, mean_lag, cv, gamma_val = flat[:N_SHARED]
            beta_g = flat[_bg_idx]
            beta_s = flat[_bs_idx]
            deltas = flat[_d_off : _d_off + n_holds]
            total = 0.0

            for i, h in enumerate(fit_holds):
                offset = _ic_off + i * N_PH
                tau_washout, paco2_0 = flat[offset : offset + N_PH]

                shift = beta_g * _g_h[i] + beta_s * _neg_s_h[i] + deltas[i]
                pred = predict_v6(
                    h["t"], pvo2, tau_washout, gamma_val,
                    paco2_0, k_co2, r_offset, tau_reoxy, mean_lag, cv, h["t_end"],
                    shift=shift,
                )
                m = _masks[i]
                total += np.sum(_weights[i] * (h["spo2"][m] - pred[m]) ** 2)

                total += nadir_timing_penalty_squared(h["t"], pred, _nadir[i])
                total += LAMBDA_PACO2 * (paco2_0 - 40.0) ** 2

            total += LAMBDA_SHRINK_A * np.sum(deltas**2)
            total += LAMBDA_ZEROSUM_A * np.sum(deltas) ** 2
            total += LAMBDA_BETA_G_A * beta_g ** 2
            total += LAMBDA_BETA_S_A * beta_s ** 2

            total += LAMBDA_K_CO2 * (k_co2 - 0.06) ** 2
            total += LAMBDA_GAMMA * (gamma_val - 1.0) ** 2
            total += LAMBDA_R_OFFSET * r_offset ** 2

            total += ic_regularization(flat, N_SHARED, fit_holds, _tg, ic_offset=_ic_off)
            return total

        maxiter_p = 2000
        sweep_idx = mean_lag_values.index(ml_fixed) + 1
        sweep_total = len(mean_lag_values)
        res = differential_evolution(
            objective, bounds, maxiter=maxiter_p, seed=42, tol=1e-10,
            polish=True, popsize=30, mutation=(0.5, 1.5), recombination=0.9,
            callback=make_de_callback(f"Profile {sweep_idx}/{sweep_total}", maxiter_p),
        )
        results[ml_fixed] = {"flat": res.x, "loss": res.fun, "success": res.success}
        beta_g_val = res.x[N_SHARED]
        beta_s_val = res.x[N_SHARED + 1]
        print(f"\n    mean_lag={ml_fixed:5.1f}: loss={res.fun:.2f}, "
              f"cv={res.x[5]:.3f}, beta_g={beta_g_val:.3f}, beta_s={beta_s_val:.3f}, "
              f"nfev={res.nfev}", flush=True)

    return results


# ── Evaluation helpers ───────────────────────────────────────────────────────


def evaluate_exp_a(flat, fit_holds, nadir_info, severities, end_slopes, all_holds=None):
    """Evaluator for Exp A (co-trained with two covariates)."""
    results = []
    target_holds = all_holds if all_holds is not None else fit_holds
    fit_ids = {h["id"] for h in fit_holds}
    n_holds = len(fit_holds)

    beta_g_idx = N_SHARED
    beta_s_idx = N_SHARED + 1
    delta_offset = N_SHARED + 2
    ic_offset = delta_offset + n_holds

    pvo2, k_co2, r_offset, tau_reoxy, mean_lag, cv, gamma_val = flat[:N_SHARED]
    beta_g = flat[beta_g_idx]
    beta_s = flat[beta_s_idx]
    deltas = flat[delta_offset : delta_offset + n_holds]

    for h in target_holds:
        if h["id"] not in fit_ids:
            type_indices = [i for i, fh in enumerate(fit_holds) if fh["type"] == h["type"]]
            if not type_indices:
                continue
            avg_ph = np.mean(
                [flat[ic_offset + idx * N_PH : ic_offset + (idx + 1) * N_PH]
                 for idx in type_indices],
                axis=0,
            )
            tau_washout, paco2_0 = avg_ph
            hold_idx = None
            is_excl = True
        else:
            hold_idx = next(i for i, fh in enumerate(fit_holds) if fh["id"] == h["id"])
            ph_start = ic_offset + hold_idx * N_PH
            tau_washout, paco2_0 = flat[ph_start : ph_start + N_PH]
            is_excl = False

        # Per-hold shift
        shift = 0.0
        delta_val = 0.0
        if hold_idx is not None:
            delta_val = deltas[hold_idx]
            neg_s = -end_slopes.get(h["id"], 0.0)
            shift = beta_g * severities[h["id"]] + beta_s * neg_s + delta_val

        pred_full = predict_v6(
            h["t"], pvo2, tau_washout, gamma_val,
            paco2_0, k_co2, r_offset, tau_reoxy, mean_lag, cv, h["t_end"],
            shift=shift,
        )
        pred_apnea = predict_v6(
            h["t_apnea"], pvo2, tau_washout, gamma_val,
            paco2_0, k_co2, r_offset, tau_reoxy, mean_lag, cv, h["t_end"],
            shift=shift,
        )

        r2_full = compute_r2(h["spo2"], pred_full)
        r2_apnea = compute_r2(h["spo2_apnea"], pred_apnea)

        mask = nadir_window_mask(h["t"], h["t_end"])
        r2_nadir = compute_r2(h["spo2"][mask], pred_full[mask]) if mask.sum() > 3 else None

        r2_recovery = None
        if len(h["t_recovery"]) > 3:
            pred_rec = predict_v6(
                h["t_recovery"], pvo2, tau_washout, gamma_val,
                paco2_0, k_co2, r_offset, tau_reoxy, mean_lag, cv, h["t_end"],
                shift=shift,
            )
            r2_recovery = compute_r2(h["spo2_recovery"], pred_rec)

        t_nadir_obs = nadir_info[h["id"]]["t_nadir"]
        t_nadir_pred = h["t"][np.argmin(pred_full)]
        nadir_err = t_nadir_pred - t_nadir_obs

        effective_mean_lag = max(mean_lag + shift, 1.0)

        rec = {
            "variant": "A:2-covariate",
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
            "effective_mean_lag": effective_mean_lag,
            "delta": delta_val,
        }
        results.append(rec)
    return results


def evaluate_exp_b(flat_b, fit_holds, nadir_info, severities, end_slopes):
    """Evaluate Exp B: sensor-only with latent splines + all fixes."""
    n_kernel = 5
    n_holds = len(fit_holds)
    delta_offset = n_kernel
    min_shift_offset = delta_offset + n_holds
    spline_offset = min_shift_offset + n_holds
    n_spline_per_hold = 8

    mean_lag, cv, r_offset, beta_g, beta_s = flat_b[:n_kernel]
    deltas = flat_b[delta_offset : delta_offset + n_holds]
    min_shifts = flat_b[min_shift_offset : min_shift_offset + n_holds]
    g_h = [severities[h["id"]] for h in fit_holds]
    neg_s_h = [-end_slopes[h["id"]] for h in fit_holds]

    results = []
    latent_curves = []

    for i, h in enumerate(fit_holds):
        sp_start = spline_offset + i * n_spline_per_hold
        spline_params = flat_b[sp_start : sp_start + n_spline_per_hold]

        # Build latent on 1-Hz grid with min_shift
        t_1hz = np.arange(0, h["t"][-1] + 1, 1.0)
        latent = build_latent_spline(t_1hz, h["t_end"], spline_params,
                                     min_shift=min_shifts[i])

        # Apply kernel
        shift = beta_g * g_h[i] + beta_s * neg_s_h[i] + deltas[i]
        eff_mean_lag = max(mean_lag + shift, 1.0)
        filtered = apply_gamma_kernel(latent, eff_mean_lag, cv)

        # Interpolate to obs times
        pred_at_obs = np.interp(h["t"], t_1hz, filtered + r_offset)
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
            "variant": "B:sensor-rebal",
            "hold_id": h["id"],
            "hold_type": h["type"],
            "r2_full": r2_full,
            "r2_apnea": r2_apnea,
            "r2_nadir": r2_nadir,
            "r2_recovery": None,
            "pred_full": pred_at_obs,
            "nadir_err": nadir_err,
            "t_nadir_pred": t_nadir_pred,
            "is_excluded": h["id"] in EXCLUDED_IDS,
            "effective_mean_lag": max(mean_lag + shift, 1.0),
        })
        latent_curves.append({
            "hold_id": h["id"],
            "t_1hz": t_1hz,
            "latent": latent,
            "filtered": filtered + r_offset,
            "spline_params": spline_params,
            "min_shift": min_shifts[i],
        })

    return results, latent_curves


def evaluate_exp_c(flat_c, fit_holds, nadir_info, severities, end_slopes, frozen_kernel,
                   all_holds=None):
    """Evaluate Exp C: frozen sensor + apnea-only physiology."""
    results = []
    target_holds = all_holds if all_holds is not None else fit_holds
    fit_ids = {h["id"] for h in fit_holds}
    n_holds = len(fit_holds)

    n_phys = 4
    delta_offset = n_phys
    ic_offset = delta_offset + n_holds

    pvo2, k_co2, tau_reoxy, gamma_val = flat_c[:n_phys]
    deltas = flat_c[delta_offset : delta_offset + n_holds]

    mean_lag = frozen_kernel["mean_lag"]
    cv = frozen_kernel["cv"]
    r_offset = frozen_kernel["r_offset"]
    beta_g = frozen_kernel["beta_g"]
    beta_s = frozen_kernel["beta_s"]

    for h in target_holds:
        if h["id"] not in fit_ids:
            type_indices = [i for i, fh in enumerate(fit_holds) if fh["type"] == h["type"]]
            if not type_indices:
                continue
            avg_ph = np.mean(
                [flat_c[ic_offset + idx * N_PH : ic_offset + (idx + 1) * N_PH]
                 for idx in type_indices],
                axis=0,
            )
            tau_washout, paco2_0 = avg_ph
            hold_idx = None
            is_excl = True
        else:
            hold_idx = next(i for i, fh in enumerate(fit_holds) if fh["id"] == h["id"])
            ph_start = ic_offset + hold_idx * N_PH
            tau_washout, paco2_0 = flat_c[ph_start : ph_start + N_PH]
            is_excl = False

        shift = 0.0
        delta_val = 0.0
        if hold_idx is not None:
            delta_val = deltas[hold_idx]
            neg_s = -end_slopes.get(h["id"], 0.0)
            shift = beta_g * severities[h["id"]] + beta_s * neg_s + delta_val

        pred_full = predict_v6(
            h["t"], pvo2, tau_washout, gamma_val,
            paco2_0, k_co2, r_offset, tau_reoxy, mean_lag, cv, h["t_end"],
            shift=shift,
        )
        pred_apnea = predict_v6(
            h["t_apnea"], pvo2, tau_washout, gamma_val,
            paco2_0, k_co2, r_offset, tau_reoxy, mean_lag, cv, h["t_end"],
            shift=shift,
        )

        r2_full = compute_r2(h["spo2"], pred_full)
        r2_apnea = compute_r2(h["spo2_apnea"], pred_apnea)

        mask = nadir_window_mask(h["t"], h["t_end"])
        r2_nadir = compute_r2(h["spo2"][mask], pred_full[mask]) if mask.sum() > 3 else None

        r2_recovery = None
        if len(h["t_recovery"]) > 3:
            pred_rec = predict_v6(
                h["t_recovery"], pvo2, tau_washout, gamma_val,
                paco2_0, k_co2, r_offset, tau_reoxy, mean_lag, cv, h["t_end"],
                shift=shift,
            )
            r2_recovery = compute_r2(h["spo2_recovery"], pred_rec)

        t_nadir_obs = nadir_info[h["id"]]["t_nadir"]
        t_nadir_pred = h["t"][np.argmin(pred_full)]
        nadir_err = t_nadir_pred - t_nadir_obs

        rec = {
            "variant": "C:frozen+apnea",
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
            "effective_mean_lag": max(mean_lag + shift, 1.0),
            "delta": delta_val,
        }
        results.append(rec)
    return results


# ── Output functions ─────────────────────────────────────────────────────────


def print_shared_params(flat, label, names=None, bounds=None):
    """Print shared parameter values with bounds."""
    if names is None:
        names = SHARED_NAMES
    if bounds is None:
        bounds = SHARED_BOUNDS
    print(f"\n  {label}:")
    for name, val, (lo, hi) in zip(names, flat[:len(names)], bounds):
        flag = " ** AT BOUND **" if is_at_bound(val, lo, hi) else ""
        print(f"    {name:>12s} = {val:8.4f}  [{lo:>6.2f}, {hi:>6.2f}]{flag}")

    # Kernel stats if mean_lag and cv are present
    if "mean_lag" in names and "cv" in names:
        ml_idx = names.index("mean_lag")
        cv_idx = names.index("cv")
        mean_lag, cv = flat[ml_idx], flat[cv_idx]
        k = 1.0 / (cv * cv)
        std = mean_lag * cv
        print(f"\n    Kernel (smooth): k={k:.2f}, mean={mean_lag:.1f}s, std={std:.1f}s (no cap)")


def print_perhold_ics(flat, ic_offset, fit_holds, label):
    """Print per-hold IC parameters."""
    print(f"\n  {label} -- Per-hold ICs (Aa=0 fixed):")
    for i, h in enumerate(fit_holds):
        offset = ic_offset + i * N_PH
        tau_washout, paco2_0 = flat[offset : offset + N_PH]
        pao2_0 = corrected_pao2_0(paco2_0, 0.0)
        ph_bounds = PERHOLD_BOUNDS[h["type"]]
        at = []
        for val, (lo, hi), name in zip([tau_washout, paco2_0], ph_bounds, PERHOLD_NAMES):
            if is_at_bound(val, lo, hi):
                at.append(f"{name}={'lo' if abs(val - lo) < 1e-3 else 'hi'}")
        bound_str = f"  [{', '.join(at)}]" if at else ""
        print(f"    {h['type']}#{h['id']}: tau_w={tau_washout:.1f}, "
              f"paco2_0={paco2_0:.1f}, PaO2_0={pao2_0:.1f}{bound_str}")


def print_comparison_table(all_results, variant_names):
    """Print per-hold R2 + nadir error comparison across variants."""
    print(f"\n{'='*140}")
    print("PER-HOLD COMPARISON")
    print(f"{'='*140}")

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


def plot_per_hold_detail(all_results, all_holds, variant_names, nadir_info, output_path):
    """Per-hold detail plots: panels for A, B, C overlaid."""
    by_hold = {}
    for r in all_results:
        by_hold.setdefault(r["hold_id"], {})[r["variant"]] = r

    holds_dict = {h["id"]: h for h in all_holds}
    hold_ids = sorted(by_hold.keys())
    n = len(hold_ids)

    fig, axes = plt.subplots(n, 1, figsize=(16, 4.5 * n), squeeze=False)
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

    for idx, hid in enumerate(hold_ids):
        ax = axes[idx, 0]
        h = holds_dict[hid]
        variants = by_hold[hid]
        ni = nadir_info[hid]

        ax.plot(h["t"], h["spo2"], "k.", markersize=2, alpha=0.5, label="Observed")
        ax.axvline(x=h["t_end"], color="red", linestyle="--", alpha=0.5, label="Apnea end")
        ax.axvline(x=h["t_end"] + NADIR_WINDOW_AFTER, color="gray", linestyle=":",
                   alpha=0.3, label=f"Nadir window (+{NADIR_WINDOW_AFTER}s)")
        ax.plot(ni["t_nadir"], ni["spo2_nadir"], "r*", markersize=12, zorder=5,
                label=f"Obs nadir (t={ni['t_nadir']:.0f}s)")

        for i, vn in enumerate(variant_names):
            r = variants.get(vn)
            if not r:
                continue
            if len(r["pred_full"]) == len(h["t"]):
                t_plot = h["t"]
            else:
                t_plot = np.arange(len(r["pred_full"]))

            r2_str = ""
            if r["r2_apnea"] is not None:
                r2_str = f"R2a={r['r2_apnea']:.3f}"
            if r["r2_nadir"] is not None:
                r2_str += f",nw={r['r2_nadir']:.3f}"
            r2_str += f",ne={r['nadir_err']:+.1f}s"
            ax.plot(t_plot, r["pred_full"], color=colors[i % len(colors)],
                    linewidth=1.5, alpha=0.8, label=f"{vn} ({r2_str})")

        tag = " [EXCLUDED]" if hid in EXCLUDED_IDS else ""
        ax.set_title(f"{h['type']}#{hid}{tag}", fontsize=13, fontweight="bold",
                     color="red" if hid in EXCLUDED_IDS else "black")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("SpO2 (%)")
        ax.set_ylim(30, 105)
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(True, alpha=0.3)

    fig.suptitle("v6.07: Slope Covariate + Rebalanced Sensor-Only — Per-Hold Detail",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {output_path}")


def plot_kernel_shape(flat_a, flat_b, output_path):
    """Plot smooth gamma kernel shapes for Exp A and B."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Exp A kernel
    mean_lag_a, cv_a = flat_a[4], flat_a[5]
    h_a = gamma_kernel_smooth(mean_lag_a, cv_a)
    k_a = 1.0 / (cv_a * cv_a)
    t_a = np.arange(len(h_a))
    axes[0].fill_between(t_a, h_a, alpha=0.3, color="#1f77b4")
    axes[0].plot(t_a, h_a, color="#1f77b4", linewidth=2, label="v6.07 smooth")
    axes[0].axvline(x=mean_lag_a, color="red", linestyle="--", alpha=0.7,
                    label=f"mean={mean_lag_a:.1f}s")
    axes[0].set_title(f"Exp A: 2-Covariate Kernel (k={k_a:.1f}, cv={cv_a:.3f})",
                      fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Time lag (s)")
    axes[0].set_ylabel("h(t)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Exp B kernel (sensor-only rebalanced)
    mean_lag_b, cv_b = flat_b[0], flat_b[1]
    beta_g_b = flat_b[3]
    beta_s_b = flat_b[4]
    h_b = gamma_kernel_smooth(mean_lag_b, cv_b)
    k_b = 1.0 / (cv_b * cv_b)
    t_b = np.arange(len(h_b))
    axes[1].fill_between(t_b, h_b, alpha=0.3, color="#2ca02c")
    axes[1].plot(t_b, h_b, color="#2ca02c", linewidth=2,
                 label=f"Base (beta_g={beta_g_b:.2f}, beta_s={beta_s_b:.2f})")
    axes[1].axvline(x=mean_lag_b, color="red", linestyle="--", alpha=0.7,
                    label=f"mean={mean_lag_b:.1f}s")

    axes[1].set_title(f"Exp B: Sensor-Only Rebalanced (k={k_b:.1f}, cv={cv_b:.3f})",
                      fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Time lag (s)")
    axes[1].set_ylabel("h(t)")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("v6.07: Smooth Gamma Kernel Impulse Responses",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Kernel shape plot saved to {output_path}")


def plot_latent_splines(latent_curves, fit_holds, nadir_info, output_path):
    """Plot latent splines from Exp B: pre-kernel SaO2, post-kernel SpO2, and observed."""
    n = len(latent_curves)
    fig, axes = plt.subplots(n, 1, figsize=(16, 4.5 * n), squeeze=False)

    holds_dict = {h["id"]: h for h in fit_holds}

    for idx, lc in enumerate(latent_curves):
        ax = axes[idx, 0]
        h = holds_dict[lc["hold_id"]]
        ni = nadir_info[lc["hold_id"]]
        m_h = lc.get("min_shift", 0.0)

        # Observed
        ax.plot(h["t"], h["spo2"], "k.", markersize=3, alpha=0.5, label="Observed SpO2")

        # Latent (pre-kernel)
        ax.plot(lc["t_1hz"], lc["latent"], color="#ff7f0e", linewidth=2, alpha=0.8,
                label="Latent SaO2 (pre-kernel)")

        # Filtered (post-kernel)
        ax.plot(lc["t_1hz"], np.clip(lc["filtered"], 0, 100), color="#1f77b4",
                linewidth=2, alpha=0.8, label="Predicted SpO2 (post-kernel)")

        ax.axvline(x=h["t_end"], color="red", linestyle="--", alpha=0.5, label="Apnea end")
        ax.plot(ni["t_nadir"], ni["spo2_nadir"], "r*", markersize=12, zorder=5)

        # Mark spline knots (with min_shift)
        sp = lc["spline_params"]
        t_end = h["t_end"]
        t_nadir_knot = t_end + m_h
        knot_times = [0, t_end / 4, t_end / 2, 3 * t_end / 4, t_nadir_knot,
                      t_nadir_knot + 10, t_nadir_knot + 30, t_nadir_knot + 45]
        values = np.zeros(8)
        values[0] = sp[0]
        for j in range(4):
            values[j + 1] = values[j] - sp[1 + j]
        for j in range(3):
            values[5 + j] = values[4 + j] + sp[5 + j]
        values = np.clip(values, 0, 100)
        ax.plot(knot_times, values, "s", color="#ff7f0e", markersize=8, zorder=5,
                label="Spline knots")

        # Show min_shift
        if abs(m_h) > 0.1:
            ax.axvline(x=t_nadir_knot, color="purple", linestyle=":", alpha=0.5,
                       label=f"Shifted nadir (m_h={m_h:+.1f}s)")

        ax.set_title(f"Exp B Latent: {h['type']}#{h['id']} (m_h={m_h:+.1f}s)",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("SaO2 / SpO2 (%)")
        ax.set_ylim(30, 105)
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(True, alpha=0.3)

    fig.suptitle("v6.07 Exp B: Latent Spline SaO2 with Nadir Shift (min_shift)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Latent spline plot saved to {output_path}")


def plot_mean_lag_profile(results_per_ml, output_path):
    """Plot loss and key params vs fixed mean_lag."""
    ml_list = sorted(results_per_ml.keys())

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    configs = [
        (None, "loss", "Total Loss"),
        (0, "pvo2", "PvO2 (mmHg)"),
        (1, "k_co2", "k_CO2 (mmHg/s)"),
        (2, "r_offset", "r_offset"),
        (5, "cv", "cv (kernel shape)"),
        (6, "gamma", "gamma"),
        (N_SHARED, "beta_g", "beta_g (severity)"),
        (N_SHARED + 1, "beta_s", "beta_s (slope)"),
    ]

    for i, (pidx, name, title) in enumerate(configs):
        ax = axes[i]
        if pidx is not None:
            vals = [results_per_ml[ml]["flat"][pidx] for ml in ml_list]
        else:
            vals = [results_per_ml[ml]["loss"] for ml in ml_list]
        ax.plot(ml_list, vals, "o-", color="#1f77b4", linewidth=2, markersize=6)
        ax.set_xlabel("mean_lag (fixed, s)")
        ax.set_ylabel(name)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)

    fig.suptitle("v6.07: mean_lag Profile (two covariates, 9-point sweep)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"mean_lag profile plot saved to {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    print("=" * 140)
    print("v6.07: Slope Covariate + Rebalanced Sensor-Only + Apnea-Only Physiology")
    print("=" * 140)

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
    print(f"\n{'='*140}")
    print("OBSERVED NADIR TIMING")
    print(f"{'='*140}")

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
    print(f"  {'Hold':<15s} | {'g_h (depth)':>12s} | {'s_h (slope)':>12s} | {'-s_h':>8s} | {'min SpO2':>8s}")
    print(f"  {'-'*15}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}-+-{'-'*8}")
    for h in fit_holds:
        g = severities[h["id"]]
        s = end_slopes[h["id"]]
        print(f"  {h['type']}#{h['id']:<10d} | {g:12.3f} | {s:12.3f} | {-s:8.3f} | {np.min(h['spo2']):8.0f}")

    n_holds = len(fit_holds)

    # ── Summary of changes from v6.06 ────────────────────────────────────────
    print(f"\nChanges from v6.06:")
    print(f"  1. End-slope covariate: s_h = (SpO2(t_end) - SpO2(t_end-10)) / 10")
    print(f"  2. Two-covariate lag model: eff_lag = mean_lag + beta_g*g_h + beta_s*(-s_h) + delta_h")
    print(f"  3. Rebalanced sensor-only: LAMBDA_SHRINK 300->30, Huber timing, min_shift")
    print(f"  4. Apnea-only physiology in Exp C: loss window [0, t_end+5]")

    # ══════════════════════════════════════════════════════════════════════════
    # EXP A: Co-trained + two covariates (24 params)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*140}")
    print("EXP A: Co-trained + Two Covariates (24 params)")
    print(f"{'='*140}")

    flat_a, conv_a = run_exp_a(fit_holds, nadir_info, severities, end_slopes)
    print_shared_params(flat_a, "Exp A")

    beta_g_a = flat_a[N_SHARED]
    beta_s_a = flat_a[N_SHARED + 1]
    print(f"\n    beta_g = {beta_g_a:.4f}  [{BETA_G_BOUNDS[0]:.2f}, {BETA_G_BOUNDS[1]:.2f}]"
          f"{' ** AT BOUND **' if is_at_bound(beta_g_a, *BETA_G_BOUNDS) else ''}")
    print(f"    beta_s = {beta_s_a:.4f}  [{BETA_S_BOUNDS[0]:.2f}, {BETA_S_BOUNDS[1]:.2f}]"
          f"{' ** AT BOUND **' if is_at_bound(beta_s_a, *BETA_S_BOUNDS) else ''}")

    delta_offset_a = N_SHARED + 2
    ic_offset_a = delta_offset_a + n_holds
    print_perhold_ics(flat_a, ic_offset_a, fit_holds, "Exp A")

    # Structural shift breakdown
    mean_lag_a = flat_a[4]
    g_h_list = [severities[h["id"]] for h in fit_holds]
    neg_s_h_list = [-end_slopes[h["id"]] for h in fit_holds]
    deltas_a = flat_a[delta_offset_a : delta_offset_a + n_holds]
    print(f"\n  Exp A -- Shift breakdown (base mean_lag={mean_lag_a:.2f}, "
          f"beta_g={beta_g_a:.3f}, beta_s={beta_s_a:.3f}):")
    for i, h in enumerate(fit_holds):
        sys_g = beta_g_a * g_h_list[i]
        sys_s = beta_s_a * neg_s_h_list[i]
        residual = deltas_a[i]
        total_shift = sys_g + sys_s + residual
        eff = mean_lag_a + total_shift
        bound_str = " *BOUND*" if is_at_bound(residual, *DELTA_BOUNDS_A) else ""
        print(f"    {h['type']}#{h['id']}: bg*g={sys_g:+6.2f}, bs*(-s)={sys_s:+6.2f}, "
              f"delta={residual:+6.2f}, total={total_shift:+6.2f}, eff_lag={eff:6.2f}{bound_str}")

    eval_a = evaluate_exp_a(flat_a, fit_holds, nadir_info, severities, end_slopes, all_holds)

    # ══════════════════════════════════════════════════════════════════════════
    # EXP B: Sensor-only, rebalanced (55 params)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*140}")
    print("EXP B: Sensor-Only, Rebalanced (55 params)")
    print(f"{'='*140}")

    flat_b, conv_b = run_exp_b(fit_holds, nadir_info, severities, end_slopes)
    n_kernel_b = 5
    mean_lag_b, cv_b, r_offset_b, beta_g_b, beta_s_b = flat_b[:n_kernel_b]
    delta_offset_b = n_kernel_b
    min_shift_offset_b = delta_offset_b + n_holds
    spline_offset_b = min_shift_offset_b + n_holds
    deltas_b = flat_b[delta_offset_b : delta_offset_b + n_holds]
    min_shifts_b = flat_b[min_shift_offset_b : min_shift_offset_b + n_holds]

    print(f"\n  Exp B kernel params:")
    kernel_names_b = ["mean_lag", "cv", "r_offset", "beta_g", "beta_s"]
    kernel_bounds_b = [(5, 45), (0.10, 1.2), (-8, 8), (-2, 2), (-2, 2)]
    for name, val, (lo, hi) in zip(kernel_names_b, flat_b[:n_kernel_b], kernel_bounds_b):
        flag = " ** AT BOUND **" if is_at_bound(val, lo, hi) else ""
        print(f"    {name:>12s} = {val:8.4f}  [{lo:>6.2f}, {hi:>6.2f}]{flag}")
    k_b = 1.0 / (cv_b * cv_b)
    print(f"\n    Kernel (smooth): k={k_b:.2f}, mean={mean_lag_b:.1f}s, std={mean_lag_b * cv_b:.1f}s")

    print(f"\n  Exp B -- Per-hold shifts + min_shift:")
    for i, h in enumerate(fit_holds):
        sys_g = beta_g_b * severities[h["id"]]
        sys_s = beta_s_b * (-end_slopes[h["id"]])
        residual = deltas_b[i]
        total_shift = sys_g + sys_s + residual
        eff = mean_lag_b + total_shift
        m_h = min_shifts_b[i]
        print(f"    {h['type']}#{h['id']}: bg*g={sys_g:+6.2f}, bs*(-s)={sys_s:+6.2f}, "
              f"delta={residual:+6.2f}, eff_lag={eff:6.2f}, m_h={m_h:+.1f}s")

    # Print spline params
    n_spline_per_hold = 8
    print(f"\n  Exp B -- Latent spline params:")
    for i, h in enumerate(fit_holds):
        sp_start = spline_offset_b + i * n_spline_per_hold
        sp = flat_b[sp_start : sp_start + n_spline_per_hold]
        values = np.zeros(8)
        values[0] = sp[0]
        for j in range(4):
            values[j + 1] = values[j] - sp[1 + j]
        for j in range(3):
            values[5 + j] = values[4 + j] + sp[5 + j]
        print(f"    {h['type']}#{h['id']}: x0={sp[0]:.1f}, dec=[{sp[1]:.1f},{sp[2]:.1f},"
              f"{sp[3]:.1f},{sp[4]:.1f}], inc=[{sp[5]:.1f},{sp[6]:.1f},{sp[7]:.1f}]")
        print(f"      -> knot values: [{', '.join(f'{v:.1f}' for v in values)}]")

    eval_b, latent_curves = evaluate_exp_b(flat_b, fit_holds, nadir_info, severities, end_slopes)

    # ══════════════════════════════════════════════════════════════════════════
    # EXP C: Frozen Sensor + Apnea-Only Physiology (19 params)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*140}")
    print("EXP C: Frozen Sensor + Apnea-Only Physiology (19 params)")
    print(f"{'='*140}")

    frozen_kernel = {
        "mean_lag": mean_lag_b,
        "cv": cv_b,
        "r_offset": r_offset_b,
        "beta_g": beta_g_b,
        "beta_s": beta_s_b,
    }

    flat_c, conv_c = run_exp_c(fit_holds, nadir_info, severities, end_slopes, frozen_kernel)

    phys_names_c = ["pvo2", "k_co2", "tau_reoxy", "gamma"]
    phys_bounds_c = [(15, 50), (0.02, 0.25), (5, 30), (0.8, 3.0)]
    print(f"\n  Exp C physiology params (sensor frozen from Exp B):")
    for name, val, (lo, hi) in zip(phys_names_c, flat_c[:4], phys_bounds_c):
        flag = " ** AT BOUND **" if is_at_bound(val, lo, hi) else ""
        print(f"    {name:>12s} = {val:8.4f}  [{lo:>6.2f}, {hi:>6.2f}]{flag}")

    n_phys_c = 4
    delta_offset_c = n_phys_c
    ic_offset_c = delta_offset_c + n_holds
    deltas_c = flat_c[delta_offset_c : delta_offset_c + n_holds]

    print(f"\n  Exp C -- Per-hold shifts:")
    for i, h in enumerate(fit_holds):
        sys_g = beta_g_b * severities[h["id"]]
        sys_s = beta_s_b * (-end_slopes[h["id"]])
        residual = deltas_c[i]
        total_shift = sys_g + sys_s + residual
        eff = mean_lag_b + total_shift
        print(f"    {h['type']}#{h['id']}: bg*g={sys_g:+6.2f}, bs*(-s)={sys_s:+6.2f}, "
              f"delta={residual:+6.2f}, eff_lag={eff:6.2f}")

    print_perhold_ics(flat_c, ic_offset_c, fit_holds, "Exp C")

    eval_c = evaluate_exp_c(flat_c, fit_holds, nadir_info, severities, end_slopes,
                            frozen_kernel, all_holds)

    # ── Comparison table ─────────────────────────────────────────────────────
    variant_names = ["A:2-covariate", "B:sensor-rebal", "C:frozen+apnea"]
    all_results = eval_a + eval_b + eval_c
    print_comparison_table(all_results, variant_names)

    # ── Cross-experiment parameter comparison ────────────────────────────────
    print(f"\n{'='*140}")
    print("PARAMETER COMPARISON (A vs C)")
    print(f"{'='*140}")

    print(f"\n  {'Param':<12s} | {'Exp A':>10s} | {'Exp C':>10s} | {'Frozen(B)':>10s}")
    print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for i, name in enumerate(SHARED_NAMES):
        va = flat_a[i]
        if name in phys_names_c:
            vc = flat_c[phys_names_c.index(name)]
        elif name == "mean_lag":
            vc = frozen_kernel["mean_lag"]
        elif name == "cv":
            vc = frozen_kernel["cv"]
        elif name == "r_offset":
            vc = frozen_kernel["r_offset"]
        else:
            vc = float("nan")
        # Frozen from B
        if name == "mean_lag":
            vf = frozen_kernel["mean_lag"]
        elif name == "cv":
            vf = frozen_kernel["cv"]
        elif name == "r_offset":
            vf = frozen_kernel["r_offset"]
        else:
            vf = float("nan")
        print(f"  {name:<12s} | {va:10.4f} | {vc:10.4f} | {vf:10.4f}")

    # Kernel comparison
    print(f"\n  Kernel comparison (A = co-trained, B = sensor-only, C = frozen from B):")
    for label, ml, cv_val in [
        ("A", flat_a[4], flat_a[5]),
        ("B", mean_lag_b, cv_b),
        ("C(frozen)", frozen_kernel["mean_lag"], frozen_kernel["cv"]),
    ]:
        k = 1.0 / (cv_val * cv_val)
        std = ml * cv_val
        print(f"    Exp {label}: mean_lag={ml:.2f}, cv={cv_val:.3f}, k={k:.2f}, std={std:.1f}s")

    # ── mean_lag profile ─────────────────────────────────────────────────────
    print(f"\n{'='*140}")
    print("MEAN_LAG PROFILE (9-point sweep, Exp A structure, two covariates)")
    print(f"{'='*140}")

    ml_sweep = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0]
    ml_profile = run_mean_lag_profile(fit_holds, nadir_info, severities, end_slopes, ml_sweep)

    print(f"\n  {'mean_lag':>8s} | {'loss':>10s} | {'pvo2':>8s} | {'k_co2':>8s} | "
          f"{'r_offset':>8s} | {'cv':>8s} | {'gamma':>8s} | {'beta_g':>8s} | {'beta_s':>8s}")
    print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for ml in sorted(ml_profile.keys()):
        r = ml_profile[ml]
        s = r["flat"]
        pvo2, k_co2, r_off, tau_re, mean_lag, cv, gamma_val = s[:N_SHARED]
        bg = s[N_SHARED]
        bs = s[N_SHARED + 1]
        print(f"  {ml:8.1f} | {r['loss']:10.2f} | {pvo2:8.2f} | {k_co2:8.4f} | "
              f"{r_off:8.4f} | {cv:8.4f} | {gamma_val:8.4f} | {bg:8.4f} | {bs:8.4f}")

    losses = [ml_profile[ml]["loss"] for ml in sorted(ml_profile.keys())]
    is_monotone_dec = all(losses[i] >= losses[i + 1] for i in range(len(losses) - 1))
    is_monotone_inc = all(losses[i] <= losses[i + 1] for i in range(len(losses) - 1))
    is_monotone = is_monotone_dec or is_monotone_inc
    min_ml = sorted(ml_profile.keys())[np.argmin(losses)]
    print(f"\n  Monotonically decreasing: {'YES (degenerate!)' if is_monotone_dec else 'NO'}")
    print(f"  Monotonically increasing: {'YES (degenerate!)' if is_monotone_inc else 'NO'}")
    print(f"  Non-monotone: {'YES (good!)' if not is_monotone else 'NO (degenerate)'}")
    print(f"  Minimum loss at mean_lag={min_ml:.1f}")

    # ── Plots ────────────────────────────────────────────────────────────────
    output_dir = Path(__file__).resolve().parent

    plot_per_hold_detail(
        all_results, all_holds, variant_names, nadir_info,
        output_dir / "exp_v6_07_slope_rebalance.png",
    )

    plot_kernel_shape(flat_a, flat_b, output_dir / "exp_v6_07_kernel_shape.png")

    plot_latent_splines(latent_curves, fit_holds, nadir_info,
                        output_dir / "exp_v6_07_latent.png")

    plot_mean_lag_profile(ml_profile, output_dir / "exp_v6_07_mean_lag_profile.png")

    # ── Success criteria ─────────────────────────────────────────────────────
    print(f"\n{'='*140}")
    print("SUCCESS CRITERIA CHECK")
    print(f"{'='*140}")

    # 1. Exp A: Slope covariate is used (beta_s significantly nonzero)
    print(f"\n  1. [Exp A] beta_s={beta_s_a:.4f} significantly nonzero: "
          f"{'PASS' if abs(beta_s_a) > 0.05 else 'FAIL (near zero)'}")

    # 2. Exp A: RV#4 delta reduced from v6.06's -15 (bound)
    rv4_idx = next((i for i, h in enumerate(fit_holds) if h["id"] == 4), None)
    if rv4_idx is not None:
        rv4_delta = deltas_a[rv4_idx]
        rv4_at_bound = is_at_bound(rv4_delta, *DELTA_BOUNDS_A)
        print(f"  2. [Exp A] RV#4 delta={rv4_delta:.2f} (not at bound={not rv4_at_bound}): "
              f"{'PASS' if not rv4_at_bound else 'FAIL (at bound)'}")

    # 3. Exp A: Timing error < 4s (improvement over v6.06's 4.0s)
    nerrs_a = [abs(r["nadir_err"]) for r in eval_a if not r["is_excluded"]]
    avg_nerr_a = np.mean(nerrs_a) if nerrs_a else float("nan")
    print(f"  3. [Exp A] Timing error avg={avg_nerr_a:.1f}s (< 4s): "
          f"{'PASS' if avg_nerr_a < 4 else 'FAIL'}")

    # 4. Exp B: cv interior (not at 0.10 bound)
    cv_b_interior = not is_at_bound(cv_b, 0.10, 1.2)
    print(f"  4. [Exp B] cv={cv_b:.4f} interior (not at bound): "
          f"{'PASS' if cv_b_interior else 'FAIL (at bound)'}")

    # 5. Exp B: beta_g < 2.0 (not at bound)
    bg_b_interior = not is_at_bound(beta_g_b, -2.0, 2.0)
    print(f"  5. [Exp B] beta_g={beta_g_b:.4f} interior (not at bound): "
          f"{'PASS' if bg_b_interior else 'FAIL (at bound)'}")

    # 6. Exp B: Timing error improved vs v6.06 Exp C (15.4s -> <10s)
    nerrs_b = [abs(r["nadir_err"]) for r in eval_b if not r["is_excluded"]]
    avg_nerr_b = np.mean(nerrs_b) if nerrs_b else float("nan")
    print(f"  6. [Exp B] Timing error avg={avg_nerr_b:.1f}s (< 10s, v6.06 was 15.4s): "
          f"{'PASS' if avg_nerr_b < 10 else 'FAIL'}")

    # 7. Exp B: deltas nontrivial (used for RV#4/FRC#5 mismatch)
    max_delta_b = float(np.max(np.abs(deltas_b)))
    print(f"  7. [Exp B] Max |delta|={max_delta_b:.2f} (nontrivial > 1): "
          f"{'PASS' if max_delta_b > 1 else 'FAIL (deltas near zero)'}")

    # 8. Exp C: Physiology params closer to co-trained than v6.06 Exp D
    pvo2_a_val, pvo2_c_val = flat_a[0], flat_c[0]
    gamma_a_val, gamma_c_val = flat_a[6], flat_c[3]
    pvo2_diff = abs(pvo2_a_val - pvo2_c_val)
    gamma_diff = abs(gamma_a_val - gamma_c_val)
    print(f"  8. [Exp C] Physiology vs co-trained: pvo2 diff={pvo2_diff:.2f}, "
          f"gamma diff={gamma_diff:.2f}")

    # 9. Exp C: tau_reoxy behavior (at bound -> confirms recovery model wrong)
    tau_reoxy_c = flat_c[2]
    tau_reoxy_at_bound = is_at_bound(tau_reoxy_c, 5, 30)
    print(f"  9. [Exp C] tau_reoxy={tau_reoxy_c:.2f} "
          f"({'at bound -> recovery model inadequate' if tau_reoxy_at_bound else 'interior'})")

    # 10. Profile: Non-monotone or less steep
    print(f" 10. [Profile] Non-monotone: {'PASS' if not is_monotone else 'FAIL (monotone)'}")
    print(f"     Minimum at mean_lag={min_ml:.1f}")

    # Exp C: R2 on apnea
    r2_apnea_c = [r["r2_apnea"] for r in eval_c if not r["is_excluded"] and r["r2_apnea"] is not None]
    avg_r2a_c = np.mean(r2_apnea_c) if r2_apnea_c else float("nan")
    print(f" 11. [Exp C] R2(apnea) avg={avg_r2a_c:.4f} (>= 0.85): "
          f"{'PASS' if avg_r2a_c >= 0.85 else 'FAIL'}")


if __name__ == "__main__":
    main()
