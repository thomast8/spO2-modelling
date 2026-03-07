"""
v6 Experiment 5: Gamma-Kernel Sensor Model.

v6.04 proved:
  1. The IIR filter is the bottleneck — tau_f saturates at every bound we set (8s, 30s).
     This isn't non-identifiability; it's model misspecification. The data wants more
     dispersion than a 1-pole IIR can provide without going extreme.
  2. Per-hold delay works — Exp C's hierarchical delta_d gave the best timing (5.2s avg)
     with an interpretable pattern.
  3. Recovery physics don't fix the sensor sponge — Exp D's two-timescale recovery gave
     high R2 (0.93) only by exploiting tau_f=30s, not better physiology.

Strategy: Replace `SaO2_delayed -> IIR filter` with `SaO2 * h(t)` where h is a
gamma-distribution impulse response. This gives independent control of:
  - Mean lag = k*theta (timing, replaces d)
  - Dispersion = k*theta^2 (smoothing, replaces tau_f)

Parameterized via (mean_lag, cv) instead of (k, theta) for better optimization:
  - mean_lag in (5, 35) — mean of kernel
  - cv in (0.15, 1.0) — coefficient of variation = 1/sqrt(k)
    cv=0.15 -> k~44, very peaked (near pure delay)
    cv=1.0  -> k=1, exponential (maximum dispersion)

Sub-experiments:
  A: Global gamma kernel (baseline) — 17 params
  B: Gamma kernel + per-hold shifts — 22 params
  C: Reduced nadir penalty (self-identification test) — 22 params

Usage:
    cd backend && uv run python -u scripts/exp_v6_05_kernel.py
"""

import csv
import io
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution
from scipy.signal import lfilter, lfilter_zi
from scipy.special import gammaln

DB_PATH = Path(__file__).resolve().parents[3] / "data" / "spo2.db"

P50_BASE = 26.6
P_EQ = 100.0
PACO2_NORMAL = 40.0
TAU_CLEAR_FIXED = 30.0
FIO2_PB_PH2O = 149.2  # FiO2 * (PB - PH2O) = 0.2093 * (760 - 47)
RQ = 0.8

EXCLUDED_IDS = {1}  # FL#1 excluded (only 2% SpO2 variation)

# ── Regularization strengths ─────────────────────────────────────────────────

LAMBDA_REG = 10.0  # per-hold IC -> type-mean
LAMBDA_NADIR = 1000.0  # nadir timing penalty per hold
LAMBDA_NADIR_REDUCED = 100.0  # Exp C: reduced nadir penalty
LAMBDA_K_CO2 = 2000.0  # k_co2 prior toward 0.06
LAMBDA_PACO2 = 1000.0  # paco2_0 prior toward 40
LAMBDA_GAMMA = 2000.0  # gamma prior toward 1.0
LAMBDA_MEAN_LAG = 2000.0  # mean_lag prior toward 15.5
LAMBDA_R_OFFSET = 500.0  # r_offset prior toward 0
LAMBDA_SHRINK = 300.0  # per-hold delta_i shrinkage (Exp B/C)

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

# Exp A shared: pvo2, k_co2, r_offset, tau_reoxy, mean_lag, cv, gamma
# NOTE: Aa removed (fixed at 0), r_offset widened to (-8, 8)
SHARED_A_BOUNDS = [
    (15, 50),      # pvo2
    (0.02, 0.25),  # k_co2
    (-8, 8),       # r_offset — widened (was -3, 3)
    (5, 30),       # tau_reoxy
    (5, 35),       # mean_lag — replaces d
    (0.15, 1.0),   # cv — kernel shape (0.15=peaked, 1.0=exponential)
    (0.8, 3.0),    # gamma — ODC steepness
]
SHARED_A_NAMES = ["pvo2", "k_co2", "r_offset", "tau_reoxy", "mean_lag", "cv", "gamma"]
N_SHARED_A = len(SHARED_A_BOUNDS)

# Exp B/C add per-hold delta_i
DELTA_BOUNDS = (-20, 20)


# ── Data loading (reused from v6.04) ─────────────────────────────────────────


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


# ── Physiology functions (reused from v6.04) ─────────────────────────────────


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


# ── Gamma kernel (delay + cascaded IIR = Erlang filter) ──────────────────────
#
# A gamma(k, theta) impulse response is equivalent to:
#   1. A cascade of k identical 1st-order IIR lowpass filters, each with tau = theta
#   2. The cascade output has mean delay = k*theta and std = sqrt(k)*theta
#
# For non-integer k, we use n_stages = round(k) IIR stages and adjust theta
# so the mean is preserved: theta_adj = mean_lag / n_stages.
#
# This is O(n_stages * signal_length) per call using lfilter, which is MUCH
# faster than explicit convolution for the DE optimizer.


def gamma_kernel_stats(mean_lag, cv):
    """Compute kernel statistics without building the full array.

    Returns: dict with k, theta, n_stages, theta_adj, std.
    """
    k = 1.0 / (cv * cv)
    theta = mean_lag * cv * cv
    n_stages = max(min(int(round(k)), 12), 1)  # match cascaded_iir_filter cap
    theta_adj = mean_lag / n_stages  # preserve mean delay exactly
    std = mean_lag * cv
    return {
        "k": k,
        "theta": theta,
        "n_stages": n_stages,
        "theta_adj": theta_adj,
        "mean": mean_lag,
        "std": std,
    }


def gamma_kernel_discrete(mean_lag, cv, max_support=60):
    """Compute normalized discrete gamma impulse response (for plotting only).

    Not used in the optimizer — use cascaded_iir_filter instead.
    """
    k = 1.0 / (cv * cv)
    theta = mean_lag * cv * cv
    std = mean_lag * cv
    support_len = min(int(mean_lag + 4.0 * std), max_support)
    support_len = max(support_len, 2)
    L = support_len + 1

    n = np.arange(L, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_h = (k - 1.0) * np.log(np.maximum(n, 1e-30)) - n / theta - k * np.log(theta) - gammaln(k)
    h = np.exp(log_h)
    h[0] = 0.0 if k > 1.0 else (1.0 / theta) if abs(k - 1.0) < 1e-6 else 0.0
    h = np.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
    total = h.sum()
    if total > 0:
        h /= total
    return h


def cascaded_iir_filter(signal, mean_lag, cv):
    """Apply gamma-equivalent sensor filter: cascaded 1st-order IIR stages.

    n_stages = round(1/cv^2) IIR lowpass filters in cascade.
    Each stage has tau = mean_lag / n_stages (preserves total mean delay).
    Preconditioned with signal[0] to avoid edge transients.
    """
    k = 1.0 / (cv * cv)
    n_stages = max(min(int(round(k)), 12), 1)  # cap at 12 for speed; >12 is indistinguishable
    tau_stage = mean_lag / n_stages

    dt = 1.0
    alpha = dt / (tau_stage + dt)
    b_coeff = np.array([alpha])
    a_coeff = np.array([1.0, -(1.0 - alpha)])
    zi_base = lfilter_zi(b_coeff, a_coeff)

    out = signal
    for _ in range(n_stages):
        out, _ = lfilter(b_coeff, a_coeff, out, zi=zi_base * out[0])
    return out


# ── Core predict function (gamma kernel via cascaded IIR) ─────────────────────


def predict_v5(t, pvo2, tau_washout, gamma, paco2_0, k_co2, r_offset,
               tau_reoxy, mean_lag, cv, t_end, shift=0.0):
    """Full sensor pipeline with gamma kernel (cascaded IIR implementation).

    Pipeline: physiology -> ODC -> cascaded IIR filter -> clip.

    Aa is fixed at 0 (removed from search space).
    shift: per-hold kernel mean shift (Exp B/C). Effective mean = mean_lag + shift.
    """
    aa = 0.0  # fixed
    pao2_0 = corrected_pao2_0(paco2_0, aa)
    pao2 = pao2_with_exp_recovery(t, pao2_0, pvo2, tau_washout, tau_reoxy, t_end)
    p50 = p50_with_exp_recovery(t, paco2_0, k_co2, TAU_CLEAR_FIXED, t_end)
    sa = odc_severinghaus(pao2, p50, gamma)

    eff_mean_lag = max(mean_lag + shift, 1.0)
    filtered = cascaded_iir_filter(sa, eff_mean_lag, cv)

    return np.clip(filtered + r_offset, 0.0, 100.0)


# ── Nadir + loss helpers (reused from v6.04) ─────────────────────────────────


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


def nadir_timing_penalty(t, pred, t_nadir_obs, lam=LAMBDA_NADIR):
    """Squared nadir timing penalty."""
    t_nadir_pred = t[np.argmin(pred)]
    return lam * (t_nadir_pred - t_nadir_obs) ** 2


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


# ── Exp A: Global gamma kernel (baseline) ────────────────────────────────────


def run_exp_a(fit_holds, nadir_info):
    """Exp A: Global gamma kernel replacing (d, tau_f).

    Shared (7): pvo2, k_co2, r_offset, tau_reoxy, mean_lag, cv, gamma
    Per-hold (2x5=10): tau_washout_i, paco2_0_i
    Total: 17 free params.
    """
    n_holds = len(fit_holds)
    bounds = list(SHARED_A_BOUNDS)
    for h in fit_holds:
        bounds.extend(PERHOLD_BOUNDS[h["type"]])
    n_total = len(bounds)

    type_groups = {}
    for i, h in enumerate(fit_holds):
        type_groups.setdefault(h["type"], []).append(i)

    print(f"\n  Exp A: {n_total} params ({N_SHARED_A} shared + {N_PH}x{n_holds} per-hold)")
    print(f"  Gamma kernel: mean_lag in (5, 35), cv in (0.15, 1.0)")
    print(f"  Aa fixed at 0, r_offset widened to (-8, 8)")

    # Precompute masks and weights (hold data is constant)
    masks = [nadir_window_mask(h["t"], h["t_end"]) for h in fit_holds]
    weights = [np.where(h["spo2"][m] < 95, 3.0, 1.0) for h, m in zip(fit_holds, masks)]
    nadir_ts = [nadir_info[h["id"]]["t_nadir"] for h in fit_holds]

    def objective(flat):
        pvo2, k_co2, r_offset, tau_reoxy, mean_lag, cv, gamma_val = flat[:N_SHARED_A]
        total = 0.0

        for i, h in enumerate(fit_holds):
            offset = N_SHARED_A + i * N_PH
            tau_washout, paco2_0 = flat[offset : offset + N_PH]

            pred = predict_v5(
                h["t"], pvo2, tau_washout, gamma_val,
                paco2_0, k_co2, r_offset, tau_reoxy, mean_lag, cv, h["t_end"],
            )
            m = masks[i]
            total += np.sum(weights[i] * (h["spo2"][m] - pred[m]) ** 2)

            total += nadir_timing_penalty(h["t"], pred, nadir_ts[i])
            total += LAMBDA_PACO2 * (paco2_0 - 40.0) ** 2

        total += shared_priors(k_co2, gamma_val, mean_lag, r_offset)
        total += ic_regularization(flat, N_SHARED_A, fit_holds, type_groups)

        return total

    res = differential_evolution(
        objective, bounds, maxiter=3000, seed=42, tol=1e-10,
        polish=True, popsize=30, mutation=(0.5, 1.5), recombination=0.9,
    )
    print(f"  Converged: {res.success}, fun={res.fun:.2f}, nfev={res.nfev}")
    return res.x, res.success


# ── Exp B: Gamma kernel + per-hold shifts ─────────────────────────────────────


def run_exp_b(fit_holds, nadir_info):
    """Exp B: Gamma kernel with per-hold shifts (bridge experiment).

    Shared (7): pvo2, k_co2, r_offset, tau_reoxy, mean_lag, cv, gamma
    Per-hold delta (5): delta_i in (-20, 20) with shrinkage
    Per-hold ICs (2x5=10): tau_washout_i, paco2_0_i
    Total: 22 free params.
    """
    n_holds = len(fit_holds)
    bounds = list(SHARED_A_BOUNDS)
    for _ in fit_holds:
        bounds.append(DELTA_BOUNDS)
    for h in fit_holds:
        bounds.extend(PERHOLD_BOUNDS[h["type"]])
    n_total = len(bounds)

    type_groups = {}
    for i, h in enumerate(fit_holds):
        type_groups.setdefault(h["type"], []).append(i)

    n_delta = n_holds
    ic_offset = N_SHARED_A + n_delta

    masks = [nadir_window_mask(h["t"], h["t_end"]) for h in fit_holds]
    weights = [np.where(h["spo2"][m] < 95, 3.0, 1.0) for h, m in zip(fit_holds, masks)]
    nadir_ts = [nadir_info[h["id"]]["t_nadir"] for h in fit_holds]

    print(f"\n  Exp B: {n_total} params ({N_SHARED_A} shared + {n_delta} delta + "
          f"{N_PH}x{n_holds} per-hold ICs)")
    print(f"  Per-hold kernel shifts with shrinkage lambda={LAMBDA_SHRINK}")

    def objective(flat):
        pvo2, k_co2, r_offset, tau_reoxy, mean_lag, cv, gamma_val = flat[:N_SHARED_A]
        deltas = flat[N_SHARED_A : N_SHARED_A + n_delta]
        total = 0.0

        for i, h in enumerate(fit_holds):
            ph_offset = ic_offset + i * N_PH
            tau_washout, paco2_0 = flat[ph_offset : ph_offset + N_PH]

            pred = predict_v5(
                h["t"], pvo2, tau_washout, gamma_val,
                paco2_0, k_co2, r_offset, tau_reoxy, mean_lag, cv, h["t_end"],
                shift=deltas[i],
            )
            m = masks[i]
            total += np.sum(weights[i] * (h["spo2"][m] - pred[m]) ** 2)

            total += nadir_timing_penalty(h["t"], pred, nadir_ts[i])
            total += LAMBDA_PACO2 * (paco2_0 - 40.0) ** 2

        total += LAMBDA_SHRINK * np.sum(deltas**2)
        total += shared_priors(k_co2, gamma_val, mean_lag, r_offset)
        total += ic_regularization(flat, N_SHARED_A, fit_holds, type_groups, ic_offset=ic_offset)

        return total

    res = differential_evolution(
        objective, bounds, maxiter=3000, seed=42, tol=1e-10,
        polish=True, popsize=30, mutation=(0.5, 1.5), recombination=0.9,
    )
    print(f"  Converged: {res.success}, fun={res.fun:.2f}, nfev={res.nfev}")
    return res.x, res.success


# ── Exp C: Reduced nadir penalty (self-identification test) ──────────────────


def run_exp_c(fit_holds, nadir_info):
    """Exp C: Same as B but lambda_nadir=100 (test kernel self-identification).

    Shared (7): pvo2, k_co2, r_offset, tau_reoxy, mean_lag, cv, gamma
    Per-hold delta (5): delta_i in (-20, 20) with shrinkage
    Per-hold ICs (2x5=10): tau_washout_i, paco2_0_i
    Total: 22 free params.
    """
    n_holds = len(fit_holds)
    bounds = list(SHARED_A_BOUNDS)
    for _ in fit_holds:
        bounds.append(DELTA_BOUNDS)
    for h in fit_holds:
        bounds.extend(PERHOLD_BOUNDS[h["type"]])
    n_total = len(bounds)

    type_groups = {}
    for i, h in enumerate(fit_holds):
        type_groups.setdefault(h["type"], []).append(i)

    n_delta = n_holds
    ic_offset = N_SHARED_A + n_delta

    masks = [nadir_window_mask(h["t"], h["t_end"]) for h in fit_holds]
    weights = [np.where(h["spo2"][m] < 95, 3.0, 1.0) for h, m in zip(fit_holds, masks)]
    nadir_ts = [nadir_info[h["id"]]["t_nadir"] for h in fit_holds]

    print(f"\n  Exp C: {n_total} params (same as B, but lambda_nadir={LAMBDA_NADIR_REDUCED})")
    print(f"  Testing: does the kernel's mean_lag self-identify from data shape alone?")

    def objective(flat):
        pvo2, k_co2, r_offset, tau_reoxy, mean_lag, cv, gamma_val = flat[:N_SHARED_A]
        deltas = flat[N_SHARED_A : N_SHARED_A + n_delta]
        total = 0.0

        for i, h in enumerate(fit_holds):
            ph_offset = ic_offset + i * N_PH
            tau_washout, paco2_0 = flat[ph_offset : ph_offset + N_PH]

            pred = predict_v5(
                h["t"], pvo2, tau_washout, gamma_val,
                paco2_0, k_co2, r_offset, tau_reoxy, mean_lag, cv, h["t_end"],
                shift=deltas[i],
            )
            m = masks[i]
            total += np.sum(weights[i] * (h["spo2"][m] - pred[m]) ** 2)

            # REDUCED nadir penalty
            total += nadir_timing_penalty(h["t"], pred, nadir_ts[i], lam=LAMBDA_NADIR_REDUCED)
            total += LAMBDA_PACO2 * (paco2_0 - 40.0) ** 2

        total += LAMBDA_SHRINK * np.sum(deltas**2)
        total += shared_priors(k_co2, gamma_val, mean_lag, r_offset)
        total += ic_regularization(flat, N_SHARED_A, fit_holds, type_groups, ic_offset=ic_offset)

        return total

    res = differential_evolution(
        objective, bounds, maxiter=3000, seed=42, tol=1e-10,
        polish=True, popsize=30, mutation=(0.5, 1.5), recombination=0.9,
    )
    print(f"  Converged: {res.success}, fun={res.fun:.2f}, nfev={res.nfev}")
    return res.x, res.success


# ── Mean-lag profile sweep ───────────────────────────────────────────────────


def run_mean_lag_profile(fit_holds, nadir_info, mean_lag_values):
    """Fix mean_lag at each value, re-optimize using Exp A structure."""
    mean_lag_idx = 4  # mean_lag is 5th shared param (0-indexed)

    type_groups = {}
    for i, h in enumerate(fit_holds):
        type_groups.setdefault(h["type"], []).append(i)

    masks = [nadir_window_mask(h["t"], h["t_end"]) for h in fit_holds]
    weights = [np.where(h["spo2"][m] < 95, 3.0, 1.0) for h, m in zip(fit_holds, masks)]
    nadir_ts = [nadir_info[h["id"]]["t_nadir"] for h in fit_holds]

    results = {}
    for ml_fixed in mean_lag_values:
        fixed_bounds = list(SHARED_A_BOUNDS)
        fixed_bounds[mean_lag_idx] = (ml_fixed - 0.01, ml_fixed + 0.01)
        bounds = list(fixed_bounds)
        for h in fit_holds:
            bounds.extend(PERHOLD_BOUNDS[h["type"]])

        def objective(flat, _tg=type_groups, _masks=masks, _weights=weights, _nadir=nadir_ts):
            pvo2, k_co2, r_offset, tau_reoxy, mean_lag, cv, gamma_val = flat[:N_SHARED_A]
            total = 0.0

            for i, h in enumerate(fit_holds):
                offset = N_SHARED_A + i * N_PH
                tau_washout, paco2_0 = flat[offset : offset + N_PH]

                pred = predict_v5(
                    h["t"], pvo2, tau_washout, gamma_val,
                    paco2_0, k_co2, r_offset, tau_reoxy, mean_lag, cv, h["t_end"],
                )
                m = _masks[i]
                total += np.sum(_weights[i] * (h["spo2"][m] - pred[m]) ** 2)

                total += nadir_timing_penalty(h["t"], pred, _nadir[i])
                total += LAMBDA_PACO2 * (paco2_0 - 40.0) ** 2

            total += LAMBDA_K_CO2 * (k_co2 - 0.06) ** 2
            total += LAMBDA_GAMMA * (gamma_val - 1.0) ** 2
            total += LAMBDA_R_OFFSET * r_offset ** 2

            total += ic_regularization(flat, N_SHARED_A, fit_holds, _tg)
            return total

        res = differential_evolution(
            objective, bounds, maxiter=2000, seed=42, tol=1e-10,
            polish=True, popsize=30, mutation=(0.5, 1.5), recombination=0.9,
        )
        results[ml_fixed] = {"flat": res.x, "loss": res.fun, "success": res.success}
        print(f"    mean_lag={ml_fixed:5.1f}: loss={res.fun:.2f}, "
              f"cv={res.x[5]:.3f}, nfev={res.nfev}", flush=True)

    return results


# ── Evaluation helpers ───────────────────────────────────────────────────────


def evaluate_exp(flat, fit_holds, label, nadir_info, all_holds=None,
                 delta_offset=None, ic_offset_override=None):
    """Generic evaluator for all experiments."""
    results = []
    target_holds = all_holds if all_holds is not None else fit_holds
    fit_ids = {h["id"] for h in fit_holds}
    ic_offset = ic_offset_override if ic_offset_override is not None else N_SHARED_A

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

        pvo2, k_co2, r_offset, tau_reoxy, mean_lag, cv, gamma_val = flat[:N_SHARED_A]

        # Per-hold shift
        shift = 0.0
        delta_val = 0.0
        if delta_offset is not None and hold_idx is not None:
            delta_val = flat[delta_offset + hold_idx]
            shift = delta_val

        pred_full = predict_v5(
            h["t"], pvo2, tau_washout, gamma_val,
            paco2_0, k_co2, r_offset, tau_reoxy, mean_lag, cv, h["t_end"],
            shift=shift,
        )
        pred_apnea = predict_v5(
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
            pred_rec = predict_v5(
                h["t_recovery"], pvo2, tau_washout, gamma_val,
                paco2_0, k_co2, r_offset, tau_reoxy, mean_lag, cv, h["t_end"],
                shift=shift,
            )
            r2_recovery = compute_r2(h["spo2_recovery"], pred_rec)

        t_nadir_obs = nadir_info[h["id"]]["t_nadir"]
        t_nadir_pred = h["t"][np.argmin(pred_full)]
        nadir_err = t_nadir_pred - t_nadir_obs

        ph_bounds = PERHOLD_BOUNDS[h["type"]]
        at_bounds = []
        for val, (lo, hi), name in zip([tau_washout, paco2_0], ph_bounds, PERHOLD_NAMES):
            if is_at_bound(val, lo, hi):
                at_bounds.append(f"{name}={'lo' if abs(val - lo) < 1e-3 else 'hi'}")

        effective_mean_lag = max(mean_lag + shift, 1.0)
        pao2_0 = corrected_pao2_0(paco2_0, 0.0)  # Aa=0

        rec = {
            "variant": label,
            "hold_id": h["id"],
            "hold_type": h["type"],
            "r2_full": r2_full,
            "r2_apnea": r2_apnea,
            "r2_nadir": r2_nadir,
            "r2_recovery": r2_recovery,
            "at_bounds": at_bounds,
            "pred_full": pred_full,
            "pao2_0": pao2_0,
            "tau_washout": tau_washout,
            "paco2_0": paco2_0,
            "nadir_err": nadir_err,
            "t_nadir_pred": t_nadir_pred,
            "is_excluded": is_excl or h["id"] in EXCLUDED_IDS,
            "effective_mean_lag": effective_mean_lag,
        }
        if delta_offset is not None and hold_idx is not None:
            rec["delta"] = delta_val
        results.append(rec)
    return results


# ── Output functions ─────────────────────────────────────────────────────────


def print_shared_params(flat, label):
    """Print shared parameter values with bounds + kernel stats."""
    print(f"\n  {label}:")
    for name, val, (lo, hi) in zip(SHARED_A_NAMES, flat[:N_SHARED_A], SHARED_A_BOUNDS):
        flag = " ** AT BOUND **" if is_at_bound(val, lo, hi) else ""
        print(f"    {name:>12s} = {val:8.4f}  [{lo:>6.2f}, {hi:>6.2f}]{flag}")

    # Kernel stats
    mean_lag, cv = flat[4], flat[5]
    stats = gamma_kernel_stats(mean_lag, cv)
    print(f"\n    Kernel: k={stats['k']:.2f}, theta={stats['theta']:.2f}, "
          f"n_stages={stats['n_stages']}, tau_stage={stats['theta_adj']:.2f}s, "
          f"mean={stats['mean']:.1f}s, std={stats['std']:.1f}s")


def print_perhold_ics(flat, ic_offset, fit_holds, label):
    """Print per-hold IC parameters."""
    print(f"\n  {label} — Per-hold ICs (Aa=0 fixed):")
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
                r2a = f"{r['r2_apnea']:.4f}"
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
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]

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

    fig.suptitle("v6.05: Gamma-Kernel Sensor Model — Per-Hold Detail",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {output_path}")


def plot_kernel_shape(flat_a, flat_b, output_path):
    """Plot gamma kernel shapes for best fits."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Exp A kernel
    mean_lag_a, cv_a = flat_a[4], flat_a[5]
    h_a = gamma_kernel_discrete(mean_lag_a, cv_a)
    stats_a = gamma_kernel_stats(mean_lag_a, cv_a)
    t_a = np.arange(len(h_a))
    axes[0].fill_between(t_a, h_a, alpha=0.3, color="#1f77b4")
    axes[0].plot(t_a, h_a, color="#1f77b4", linewidth=2)
    axes[0].axvline(x=mean_lag_a, color="red", linestyle="--", alpha=0.7,
                    label=f"mean={mean_lag_a:.1f}s")
    axes[0].set_title(f"Exp A: Global Kernel (k={stats_a['k']:.1f}, "
                      f"cv={cv_a:.3f})", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Time lag (s)")
    axes[0].set_ylabel("h(t)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Exp B kernel
    mean_lag_b, cv_b = flat_b[4], flat_b[5]
    h_b = gamma_kernel_discrete(mean_lag_b, cv_b)
    stats_b = gamma_kernel_stats(mean_lag_b, cv_b)
    t_b = np.arange(len(h_b))
    axes[1].fill_between(t_b, h_b, alpha=0.3, color="#2ca02c")
    axes[1].plot(t_b, h_b, color="#2ca02c", linewidth=2, label="Base kernel")
    axes[1].axvline(x=mean_lag_b, color="red", linestyle="--", alpha=0.7,
                    label=f"mean={mean_lag_b:.1f}s")

    # Show shifted kernels for Exp B
    n_holds = 5  # we know there are 5 fit holds
    deltas = flat_b[N_SHARED_A : N_SHARED_A + n_holds]
    for i, delta in enumerate(deltas):
        if abs(delta) > 0.5:
            shifted_h = gamma_kernel_discrete(mean_lag_b, cv_b)
            shift_int = int(round(delta))
            if shift_int > 0:
                shifted_h = np.concatenate([np.zeros(shift_int), shifted_h])
            elif shift_int < 0:
                trim = min(-shift_int, len(shifted_h) - 1)
                shifted_h = shifted_h[trim:]
                if shifted_h.sum() > 0:
                    shifted_h /= shifted_h.sum()
            t_sh = np.arange(len(shifted_h))
            axes[1].plot(t_sh, shifted_h, linewidth=1, alpha=0.5,
                         label=f"delta={delta:+.1f}")

    axes[1].set_title(f"Exp B: Kernel + Shifts (k={stats_b['k']:.1f}, "
                      f"cv={cv_b:.3f})", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Time lag (s)")
    axes[1].set_ylabel("h(t)")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("v6.05: Gamma Kernel Impulse Responses",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Kernel shape plot saved to {output_path}")


def plot_mean_lag_profile(results_per_ml, output_path):
    """Plot loss and key params vs fixed mean_lag."""
    ml_list = sorted(results_per_ml.keys())

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    configs = [
        (None, "loss", "Total Loss"),
        (0, "pvo2", "PvO2 (mmHg)"),
        (1, "k_co2", "k_CO2 (mmHg/s)"),
        (2, "r_offset", "r_offset"),
        (5, "cv", "cv (kernel shape)"),
        (6, "gamma", "gamma"),
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

    fig.suptitle("v6.05: mean_lag Profile (Exp A structure)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"mean_lag profile plot saved to {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    print("=" * 140)
    print("v6.05: Gamma-Kernel Sensor Model")
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

    # ── Summary of changes ───────────────────────────────────────────────────
    print(f"\nChanges from v6.04:")
    print(f"  1. Replace delay+IIR with gamma (Erlang) convolution kernel")
    print(f"  2. Parameterize via (mean_lag, cv) for independent lag/dispersion control")
    print(f"  3. Fix Aa=0 (always at bound in v6.04)")
    print(f"  4. Widen r_offset to (-8, 8) with prior toward 0 (lambda={LAMBDA_R_OFFSET})")
    print(f"  5. Per-hold kernel shifts (Exp B/C) with shrinkage (lambda={LAMBDA_SHRINK})")
    print(f"  6. Reduced nadir penalty test (Exp C: lambda={LAMBDA_NADIR_REDUCED})")

    # ── Exp A: Global gamma kernel ───────────────────────────────────────────
    print(f"\n{'='*140}")
    print("EXP A: Global Gamma Kernel (17 params)")
    print(f"{'='*140}")

    flat_a, conv_a = run_exp_a(fit_holds, nadir_info)
    print_shared_params(flat_a, "Exp A")
    print_perhold_ics(flat_a, N_SHARED_A, fit_holds, "Exp A")

    eval_a = evaluate_exp(flat_a, fit_holds, "A:kernel", nadir_info, all_holds)

    # ── Exp B: Kernel + per-hold shifts ──────────────────────────────────────
    print(f"\n{'='*140}")
    print("EXP B: Gamma Kernel + Per-Hold Shifts (22 params)")
    print(f"{'='*140}")

    flat_b, conv_b = run_exp_b(fit_holds, nadir_info)
    n_holds = len(fit_holds)
    ic_offset_b = N_SHARED_A + n_holds

    print_shared_params(flat_b, "Exp B (shared)")
    print_perhold_ics(flat_b, ic_offset_b, fit_holds, "Exp B")

    # Print per-hold effective mean lags
    mean_lag_b = flat_b[4]
    print(f"\n  Exp B — Per-hold effective mean lags (base mean_lag={mean_lag_b:.2f}):")
    for i, h in enumerate(fit_holds):
        delta = flat_b[N_SHARED_A + i]
        eff = mean_lag_b + delta
        bound_str = " *BOUND*" if is_at_bound(delta, *DELTA_BOUNDS) else ""
        print(f"    {h['type']}#{h['id']}: delta={delta:+6.2f}, "
              f"effective_mean_lag={eff:6.2f}{bound_str}")

    eval_b = evaluate_exp(
        flat_b, fit_holds, "B:kernel+shift", nadir_info, all_holds,
        delta_offset=N_SHARED_A, ic_offset_override=ic_offset_b,
    )

    # ── Exp C: Reduced nadir penalty ─────────────────────────────────────────
    print(f"\n{'='*140}")
    print(f"EXP C: Reduced Nadir Penalty (lambda={LAMBDA_NADIR_REDUCED}) — Self-ID Test (22 params)")
    print(f"{'='*140}")

    flat_c, conv_c = run_exp_c(fit_holds, nadir_info)
    ic_offset_c = N_SHARED_A + n_holds

    print_shared_params(flat_c, "Exp C (shared)")
    print_perhold_ics(flat_c, ic_offset_c, fit_holds, "Exp C")

    mean_lag_c = flat_c[4]
    print(f"\n  Exp C — Per-hold effective mean lags (base mean_lag={mean_lag_c:.2f}):")
    for i, h in enumerate(fit_holds):
        delta = flat_c[N_SHARED_A + i]
        eff = mean_lag_c + delta
        bound_str = " *BOUND*" if is_at_bound(delta, *DELTA_BOUNDS) else ""
        print(f"    {h['type']}#{h['id']}: delta={delta:+6.2f}, "
              f"effective_mean_lag={eff:6.2f}{bound_str}")

    eval_c = evaluate_exp(
        flat_c, fit_holds, "C:reduced_nadir", nadir_info, all_holds,
        delta_offset=N_SHARED_A, ic_offset_override=ic_offset_c,
    )

    # ── Comparison table ─────────────────────────────────────────────────────
    variant_names = ["A:kernel", "B:kernel+shift", "C:reduced_nadir"]
    all_results = eval_a + eval_b + eval_c
    print_comparison_table(all_results, variant_names)

    # ── Cross-experiment parameter comparison ────────────────────────────────
    print(f"\n{'='*140}")
    print("PARAMETER COMPARISON (A vs B vs C)")
    print(f"{'='*140}")

    print(f"\n  {'Param':<12s} | {'Exp A':>10s} | {'Exp B':>10s} | {'Exp C':>10s}")
    print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for i, name in enumerate(SHARED_A_NAMES):
        va = flat_a[i]
        vb = flat_b[i]
        vc = flat_c[i]
        print(f"  {name:<12s} | {va:10.4f} | {vb:10.4f} | {vc:10.4f}")

    # Kernel stats comparison
    print(f"\n  Derived kernel stats:")
    for label, flat in [("A", flat_a), ("B", flat_b), ("C", flat_c)]:
        stats = gamma_kernel_stats(flat[4], flat[5])
        print(f"    Exp {label}: k={stats['k']:.2f}, theta={stats['theta']:.2f}, "
              f"mean={stats['mean']:.1f}s, std={stats['std']:.1f}s")

    # ── mean_lag profile ─────────────────────────────────────────────────────
    print(f"\n{'='*140}")
    print("MEAN_LAG PROFILE (5-point sweep, Exp A structure)")
    print(f"{'='*140}")

    ml_sweep = [8.0, 12.0, 16.0, 20.0, 24.0]
    ml_profile = run_mean_lag_profile(fit_holds, nadir_info, ml_sweep)

    print(f"\n  {'mean_lag':>8s} | {'loss':>10s} | {'pvo2':>8s} | {'k_co2':>8s} | "
          f"{'r_offset':>8s} | {'cv':>8s} | {'gamma':>8s}")
    print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for ml in sorted(ml_profile.keys()):
        r = ml_profile[ml]
        s = r["flat"][:N_SHARED_A]
        pvo2, k_co2, r_off, tau_re, mean_lag, cv, gamma_val = s
        print(f"  {ml:8.1f} | {r['loss']:10.2f} | {pvo2:8.2f} | {k_co2:8.4f} | "
              f"{r_off:8.4f} | {cv:8.4f} | {gamma_val:8.4f}")

    losses = [ml_profile[ml]["loss"] for ml in sorted(ml_profile.keys())]
    is_monotone = all(losses[i] >= losses[i + 1] for i in range(len(losses) - 1))
    min_ml = sorted(ml_profile.keys())[np.argmin(losses)]
    print(f"\n  Loss monotonically decreasing: {'YES (degenerate!)' if is_monotone else 'NO (good!)'}")
    print(f"  Minimum loss at mean_lag={min_ml:.1f}")

    # ── Plots ────────────────────────────────────────────────────────────────
    output_dir = Path(__file__).resolve().parent

    plot_per_hold_detail(
        all_results, all_holds, variant_names, nadir_info,
        output_dir / "exp_v6_05_kernel.png",
    )

    plot_kernel_shape(flat_a, flat_b, output_dir / "exp_v6_05_kernel_shape.png")

    plot_mean_lag_profile(ml_profile, output_dir / "exp_v6_05_mean_lag_profile.png")

    # ── Success criteria ─────────────────────────────────────────────────────
    print(f"\n{'='*140}")
    print("SUCCESS CRITERIA CHECK")
    print(f"{'='*140}")

    # 1. Exp A: mean_lag interior
    mean_lag_a = flat_a[4]
    ml_pass = not is_at_bound(mean_lag_a, 5, 35)
    print(f"\n  1. [Exp A] mean_lag={mean_lag_a:.2f} (interior): "
          f"{'PASS' if ml_pass else 'FAIL (at bound)'}")

    # 2. Exp A: cv interior
    cv_a = flat_a[5]
    cv_pass = not is_at_bound(cv_a, 0.15, 1.0)
    print(f"  2. [Exp A] cv={cv_a:.4f} (interior): "
          f"{'PASS' if cv_pass else 'FAIL (at bound)'}")

    # 3. Exp A: R2(apnea) avg >= 0.85
    r2_apnea_a = [r["r2_apnea"] for r in eval_a if not r["is_excluded"]]
    avg_r2a_a = np.mean(r2_apnea_a)
    print(f"  3. [Exp A] R2(apnea) avg={avg_r2a_a:.4f} (>= 0.85): "
          f"{'PASS' if avg_r2a_a >= 0.85 else 'FAIL'}")

    # 4. Exp B: timing error avg <= 5s
    nerrs_b = [abs(r["nadir_err"]) for r in eval_b if not r["is_excluded"]]
    avg_nerr_b = np.mean(nerrs_b)
    print(f"  4. [Exp B] Timing error avg={avg_nerr_b:.1f}s (<= 5s): "
          f"{'PASS' if avg_nerr_b <= 5 else 'FAIL'}")

    # 5. Exp B: R2(apnea) avg >= 0.90
    r2_apnea_b = [r["r2_apnea"] for r in eval_b if not r["is_excluded"]]
    avg_r2a_b = np.mean(r2_apnea_b)
    print(f"  5. [Exp B] R2(apnea) avg={avg_r2a_b:.4f} (>= 0.90): "
          f"{'PASS' if avg_r2a_b >= 0.90 else 'FAIL'}")

    # 6. Exp B: No kernel params at absurd bounds
    cv_b = flat_b[5]
    ml_b_interior = not is_at_bound(flat_b[4], 5, 35)
    cv_b_interior = not is_at_bound(cv_b, 0.15, 1.0)
    print(f"  6. [Exp B] Kernel params interior (mean_lag={flat_b[4]:.2f}, cv={cv_b:.4f}): "
          f"{'PASS' if ml_b_interior and cv_b_interior else 'FAIL'}")

    # 7. Exp C: mean_lag within ±3s of Exp B
    mean_lag_b_val = flat_b[4]
    mean_lag_c_val = flat_c[4]
    ml_diff = abs(mean_lag_c_val - mean_lag_b_val)
    print(f"  7. [Exp C] mean_lag={mean_lag_c_val:.2f} vs Exp B={mean_lag_b_val:.2f} "
          f"(diff={ml_diff:.2f}, <= 3s): {'PASS' if ml_diff <= 3 else 'FAIL'}")

    # 8. Exp C: timing error avg <= 8s
    nerrs_c = [abs(r["nadir_err"]) for r in eval_c if not r["is_excluded"]]
    avg_nerr_c = np.mean(nerrs_c)
    print(f"  8. [Exp C] Timing error avg={avg_nerr_c:.1f}s (<= 8s): "
          f"{'PASS' if avg_nerr_c <= 8 else 'FAIL'}")

    # 9. All exps: k_co2 interior (0.04-0.20)
    all_k_co2 = [flat_a[1], flat_b[1], flat_c[1]]
    all_interior = all(0.04 <= k <= 0.20 for k in all_k_co2)
    print(f"  9. [All] k_co2 values: {[f'{k:.4f}' for k in all_k_co2]} "
          f"(all in 0.04-0.20): {'PASS' if all_interior else 'FAIL'}")

    # 10. All exps: gamma interior
    all_gamma = [flat_a[6], flat_b[6], flat_c[6]]
    gamma_interior = all(not is_at_bound(g, 0.8, 3.0) for g in all_gamma)
    print(f" 10. [All] gamma values: {[f'{g:.4f}' for g in all_gamma]} "
          f"(all interior): {'PASS' if gamma_interior else 'FAIL'}")

    # 11. mean_lag profile: non-monotone
    print(f" 11. [Profile] Non-monotone: {'PASS' if not is_monotone else 'FAIL'}, "
          f"minimum at mean_lag={min_ml:.1f}")


if __name__ == "__main__":
    main()
