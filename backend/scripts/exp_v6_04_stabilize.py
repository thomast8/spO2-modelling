"""
v6 Experiment 4: Stabilize Sensor Pipeline + Recovery Physiology.

v6.03 broke the d-degeneracy: d converges to ~15.5s (interior) with a non-monotone
profile-likelihood minimum. The nadir timing penalty is the key structural win.

However, three problems remain:
  1. tau_f always at 30s bound — IIR filter absorbs physiology model error
  2. Heterogeneous nadir timing — single global d forces ~17s effective lag
  3. gamma hits 1.5 bound — ODC steepness insufficient, bound too tight

Strategy: freeze v6.03 wins (d~15.5, CO2 priors, nadir-window approach), then
address remaining issues in four sub-experiments:

  A: Constrained tau_f (1-8s) + widened gamma (0.8-3.0) — baseline fix
  B: Huber nadir penalty — robust to timing outliers (RV#4, FRC#5)
  C: Hierarchical per-hold delay — controlled flexibility with shrinkage
  D: Two-timescale recovery — physiological fix for sponge filter

Usage:
    cd backend && uv run python -u scripts/exp_v6_04_stabilize.py
"""

import csv
import io
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution
from scipy.signal import lfilter, lfilter_zi

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "spo2.db"

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
LAMBDA_K_CO2 = 2000.0  # k_co2 prior toward 0.06
LAMBDA_PACO2 = 1000.0  # paco2_0 prior toward 40
LAMBDA_GAMMA = 2000.0  # gamma prior toward 1.0 (reduced from v6.03's 5000)
LAMBDA_D_PRIOR = 2000.0  # d prior toward 15.5
LAMBDA_SHRINK = 500.0  # per-hold delta_d shrinkage (Exp C)

D_PRIOR_CENTER = 15.5  # v6.03 finding
NADIR_WINDOW_AFTER = 45  # seconds after t_end for loss window
HUBER_DELTA = 5.0  # Huber loss transition (seconds)

# ── Bounds ───────────────────────────────────────────────────────────────────

# Per-hold ICs: tau_washout, paco2_0
PERHOLD_BOUNDS = {
    "FL": [(50, 250), (20, 50)],
    "FRC": [(20, 100), (25, 50)],
    "RV": [(10, 80), (30, 55)],
}
PERHOLD_NAMES = ["tau_washout", "paco2_0"]
N_PH = len(PERHOLD_NAMES)

# Exp A shared: Aa, pvo2, k_co2, r_offset, tau_reoxy, d, tau_f, gamma
SHARED_A_BOUNDS = [
    (0, 30),       # Aa
    (15, 50),      # pvo2
    (0.02, 0.25),  # k_co2
    (-3, 3),       # r_offset
    (5, 30),       # tau_reoxy
    (1, 30),       # d
    (1, 8),        # tau_f — constrained (was 1-30)
    (0.8, 3.0),    # gamma — widened (was 0.8-1.5)
]
SHARED_A_NAMES = ["Aa", "pvo2", "k_co2", "r_offset", "tau_reoxy", "d", "tau_f", "gamma"]
N_SHARED_A = len(SHARED_A_BOUNDS)

# Exp C adds per-hold delta_d (5 extra params)
DELTA_D_BOUNDS = (-10, 10)

# Exp D shared: same as A but with tau_slow and A_fast instead of constrained tau_f
SHARED_D_BOUNDS = [
    (0, 30),       # Aa
    (15, 50),      # pvo2
    (0.02, 0.25),  # k_co2
    (-3, 3),       # r_offset
    (5, 30),       # tau_reoxy (= tau_fast)
    (1, 30),       # d
    (1, 30),       # tau_f — unconstrained (test if physiology fix makes it interior)
    (0.8, 3.0),    # gamma
    (0.3, 0.9),    # A_fast — fraction of fast recovery component
    (30, 120),     # tau_slow — slow recovery time constant
]
SHARED_D_NAMES = [
    "Aa", "pvo2", "k_co2", "r_offset", "tau_reoxy", "d", "tau_f", "gamma",
    "A_fast", "tau_slow",
]
N_SHARED_D = len(SHARED_D_BOUNDS)


# ── Data loading (reused from v6.03) ─────────────────────────────────────────


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


# ── Physiology functions (reused from v6.03) ─────────────────────────────────


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


def pao2_two_timescale_recovery(t, pao2_0, pvo2, tau_washout, tau_fast, tau_slow, a_fast, t_end):
    """Piecewise PAO2: exponential decay during apnea, two-timescale rise during recovery.

    Recovery: PaO2(t) = P_EQ - A*exp(-(t-t_end)/tau_fast) - (1-A)*exp(-(t-t_end)/tau_slow)
    where the deficit is split between a fast and slow component.
    """
    pao2_end = pvo2 + (pao2_0 - pvo2) * np.exp(-t_end / max(tau_washout, 0.01))
    deficit = P_EQ - pao2_end

    return np.where(
        t <= t_end,
        pvo2 + (pao2_0 - pvo2) * np.exp(-t / max(tau_washout, 0.01)),
        P_EQ
        - a_fast * deficit * np.exp(-(t - t_end) / max(tau_fast, 0.01))
        - (1.0 - a_fast) * deficit * np.exp(-(t - t_end) / max(tau_slow, 0.01)),
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


# ── Core predict functions ───────────────────────────────────────────────────


def predict_v3(t, aa, pvo2, tau_washout, gamma, paco2_0, k_co2, r_offset, tau_reoxy, d, tau_f, t_end):
    """Full sensor pipeline (reused from v6.03).

    Pipeline: physiology -> ODC -> delay -> IIR filter -> clip.
    """
    pao2_0 = corrected_pao2_0(paco2_0, aa)
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


def predict_v4(t, aa, pvo2, tau_washout, gamma, paco2_0, k_co2, r_offset,
               tau_fast, tau_slow, a_fast, d, tau_f, t_end):
    """Full sensor pipeline with two-timescale recovery (Exp D).

    Pipeline: two-timescale physiology -> ODC -> delay -> IIR filter -> clip.
    """
    pao2_0 = corrected_pao2_0(paco2_0, aa)
    pao2 = pao2_two_timescale_recovery(t, pao2_0, pvo2, tau_washout, tau_fast, tau_slow, a_fast, t_end)
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


def huber_loss(x, delta):
    """Huber loss: quadratic for |x| <= delta, linear beyond."""
    abs_x = np.abs(x)
    return np.where(abs_x <= delta, x**2, delta * (2.0 * abs_x - delta))


def nadir_timing_penalty_squared(t, pred, t_nadir_obs):
    """Squared nadir timing penalty (v6.03 style)."""
    t_nadir_pred = t[np.argmin(pred)]
    return LAMBDA_NADIR * (t_nadir_pred - t_nadir_obs) ** 2


def nadir_timing_penalty_huber(t, pred, t_nadir_obs):
    """Huber nadir timing penalty — robust to timing outliers."""
    t_nadir_pred = t[np.argmin(pred)]
    return LAMBDA_NADIR * huber_loss(t_nadir_pred - t_nadir_obs, HUBER_DELTA)


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


def shared_priors(k_co2, gamma, d):
    """CO2 prior + gamma prior + d prior (anchored to v6.03 finding)."""
    return (
        LAMBDA_K_CO2 * (k_co2 - 0.06) ** 2
        + LAMBDA_GAMMA * (gamma - 1.0) ** 2
        + LAMBDA_D_PRIOR * (d - D_PRIOR_CENTER) ** 2
    )


def ic_regularization(flat, n_shared, fit_holds, type_groups):
    """Per-hold IC regularization toward type means."""
    total = 0.0
    for ht, indices in type_groups.items():
        if len(indices) < 2:
            continue
        for p_off in range(N_PH):
            values = [flat[n_shared + idx * N_PH + p_off] for idx in indices]
            mean_val = np.mean(values)
            total += LAMBDA_REG * sum((v - mean_val) ** 2 for v in values)
    return total


# ── Exp A: Constrained tau_f + widened gamma ─────────────────────────────────


def run_exp_a(fit_holds, nadir_info):
    """Exp A: Constrained tau_f (1-8s), widened gamma (0.8-3.0), d prior at 15.5.

    Shared (8): Aa, pvo2, k_co2, r_offset, tau_reoxy, d, tau_f, gamma
    Per-hold (2x5=10): tau_washout_i, paco2_0_i
    Total: 18 free params.
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
    print(f"  tau_f constrained to (1, 8), gamma widened to (0.8, 3.0)")
    print(f"  Priors: gamma->1.0 (lambda={LAMBDA_GAMMA}), d->15.5 (lambda={LAMBDA_D_PRIOR})")

    def objective(flat):
        aa, pvo2, k_co2, r_offset, tau_reoxy, d, tau_f, gamma = flat[:N_SHARED_A]
        total = 0.0

        for i, h in enumerate(fit_holds):
            offset = N_SHARED_A + i * N_PH
            tau_washout, paco2_0 = flat[offset : offset + N_PH]

            pred = predict_v3(
                h["t"], aa, pvo2, tau_washout, gamma,
                paco2_0, k_co2, r_offset, tau_reoxy, d, tau_f, h["t_end"],
            )
            mask = nadir_window_mask(h["t"], h["t_end"])
            w = np.where(h["spo2"][mask] < 95, 3.0, 1.0)
            total += np.sum(w * (h["spo2"][mask] - pred[mask]) ** 2)

            total += nadir_timing_penalty_squared(h["t"], pred, nadir_info[h["id"]]["t_nadir"])
            total += LAMBDA_PACO2 * (paco2_0 - 40.0) ** 2

        total += shared_priors(k_co2, gamma, d)
        total += ic_regularization(flat, N_SHARED_A, fit_holds, type_groups)

        return total

    res = differential_evolution(
        objective, bounds, maxiter=6000, seed=42, tol=1e-10,
        polish=True, popsize=60, mutation=(0.5, 1.5), recombination=0.9,
    )
    print(f"  Converged: {res.success}, fun={res.fun:.2f}, nfev={res.nfev}")
    return res.x, res.success


# ── Exp B: Huber nadir penalty ───────────────────────────────────────────────


def run_exp_b(fit_holds, nadir_info):
    """Exp B: Same as A but with Huber nadir timing penalty (delta=5s).

    Shared (8): Aa, pvo2, k_co2, r_offset, tau_reoxy, d, tau_f, gamma
    Per-hold (2x5=10): tau_washout_i, paco2_0_i
    Total: 18 free params.
    """
    n_holds = len(fit_holds)
    bounds = list(SHARED_A_BOUNDS)
    for h in fit_holds:
        bounds.extend(PERHOLD_BOUNDS[h["type"]])
    n_total = len(bounds)

    type_groups = {}
    for i, h in enumerate(fit_holds):
        type_groups.setdefault(h["type"], []).append(i)

    print(f"\n  Exp B: {n_total} params (same structure as A)")
    print(f"  Huber nadir penalty with delta={HUBER_DELTA}s (linear beyond)")

    def objective(flat):
        aa, pvo2, k_co2, r_offset, tau_reoxy, d, tau_f, gamma = flat[:N_SHARED_A]
        total = 0.0

        for i, h in enumerate(fit_holds):
            offset = N_SHARED_A + i * N_PH
            tau_washout, paco2_0 = flat[offset : offset + N_PH]

            pred = predict_v3(
                h["t"], aa, pvo2, tau_washout, gamma,
                paco2_0, k_co2, r_offset, tau_reoxy, d, tau_f, h["t_end"],
            )
            mask = nadir_window_mask(h["t"], h["t_end"])
            w = np.where(h["spo2"][mask] < 95, 3.0, 1.0)
            total += np.sum(w * (h["spo2"][mask] - pred[mask]) ** 2)

            total += nadir_timing_penalty_huber(h["t"], pred, nadir_info[h["id"]]["t_nadir"])
            total += LAMBDA_PACO2 * (paco2_0 - 40.0) ** 2

        total += shared_priors(k_co2, gamma, d)
        total += ic_regularization(flat, N_SHARED_A, fit_holds, type_groups)

        return total

    res = differential_evolution(
        objective, bounds, maxiter=6000, seed=42, tol=1e-10,
        polish=True, popsize=60, mutation=(0.5, 1.5), recombination=0.9,
    )
    print(f"  Converged: {res.success}, fun={res.fun:.2f}, nfev={res.nfev}")
    return res.x, res.success


# ── Exp C: Hierarchical per-hold delay ───────────────────────────────────────


def run_exp_c(fit_holds, nadir_info):
    """Exp C: Per-hold delta_d with shrinkage.

    Shared (8): Aa, pvo2, k_co2, r_offset, tau_reoxy, d_0, tau_f, gamma
    Per-hold delta_d (5): delta_d_i in (-10, 10)
    Per-hold ICs (2x5=10): tau_washout_i, paco2_0_i
    Total: 23 free params.

    Effective delay for hold i: d_i = d_0 + delta_d_i
    """
    n_holds = len(fit_holds)
    bounds = list(SHARED_A_BOUNDS)
    # Per-hold delta_d
    for _ in fit_holds:
        bounds.append(DELTA_D_BOUNDS)
    # Per-hold ICs
    for h in fit_holds:
        bounds.extend(PERHOLD_BOUNDS[h["type"]])
    n_total = len(bounds)

    type_groups = {}
    for i, h in enumerate(fit_holds):
        type_groups.setdefault(h["type"], []).append(i)

    n_delta_d = n_holds
    ic_offset = N_SHARED_A + n_delta_d

    print(f"\n  Exp C: {n_total} params ({N_SHARED_A} shared + {n_delta_d} delta_d + "
          f"{N_PH}x{n_holds} per-hold ICs)")
    print(f"  d_i = d_0 + delta_d_i, shrinkage lambda={LAMBDA_SHRINK}")

    def objective(flat):
        aa, pvo2, k_co2, r_offset, tau_reoxy, d_0, tau_f, gamma = flat[:N_SHARED_A]
        delta_ds = flat[N_SHARED_A : N_SHARED_A + n_delta_d]
        total = 0.0

        for i, h in enumerate(fit_holds):
            d_i = max(d_0 + delta_ds[i], 0.1)  # effective delay must be positive
            ph_offset = ic_offset + i * N_PH
            tau_washout, paco2_0 = flat[ph_offset : ph_offset + N_PH]

            pred = predict_v3(
                h["t"], aa, pvo2, tau_washout, gamma,
                paco2_0, k_co2, r_offset, tau_reoxy, d_i, tau_f, h["t_end"],
            )
            mask = nadir_window_mask(h["t"], h["t_end"])
            w = np.where(h["spo2"][mask] < 95, 3.0, 1.0)
            total += np.sum(w * (h["spo2"][mask] - pred[mask]) ** 2)

            total += nadir_timing_penalty_squared(h["t"], pred, nadir_info[h["id"]]["t_nadir"])
            total += LAMBDA_PACO2 * (paco2_0 - 40.0) ** 2

        # Shrinkage on delta_d
        total += LAMBDA_SHRINK * np.sum(delta_ds**2)

        total += shared_priors(k_co2, gamma, d_0)

        # IC regularization (offset by delta_d block)
        for ht, indices in type_groups.items():
            if len(indices) < 2:
                continue
            for p_off in range(N_PH):
                values = [flat[ic_offset + idx * N_PH + p_off] for idx in indices]
                mean_val = np.mean(values)
                total += LAMBDA_REG * sum((v - mean_val) ** 2 for v in values)

        return total

    res = differential_evolution(
        objective, bounds, maxiter=8000, seed=42, tol=1e-10,
        polish=True, popsize=80, mutation=(0.5, 1.5), recombination=0.9,
    )
    print(f"  Converged: {res.success}, fun={res.fun:.2f}, nfev={res.nfev}")
    return res.x, res.success


# ── Exp D: Two-timescale recovery ────────────────────────────────────────────


def run_exp_d(fit_holds, nadir_info):
    """Exp D: Two-timescale recovery model with unconstrained tau_f.

    Shared (10): Aa, pvo2, k_co2, r_offset, tau_reoxy(=tau_fast), d, tau_f, gamma,
                 A_fast, tau_slow
    Per-hold (2x5=10): tau_washout_i, paco2_0_i
    Total: 20 free params.
    """
    n_holds = len(fit_holds)
    bounds = list(SHARED_D_BOUNDS)
    for h in fit_holds:
        bounds.extend(PERHOLD_BOUNDS[h["type"]])
    n_total = len(bounds)

    type_groups = {}
    for i, h in enumerate(fit_holds):
        type_groups.setdefault(h["type"], []).append(i)

    print(f"\n  Exp D: {n_total} params ({N_SHARED_D} shared + {N_PH}x{n_holds} per-hold)")
    print(f"  Two-timescale recovery: tau_fast=tau_reoxy, tau_slow in (30,120), A_fast in (0.3,0.9)")
    print(f"  tau_f unconstrained (1-30) — test if physiology fix makes it interior")

    def objective(flat):
        aa, pvo2, k_co2, r_offset, tau_fast, d, tau_f, gamma, a_fast, tau_slow = flat[:N_SHARED_D]
        total = 0.0

        for i, h in enumerate(fit_holds):
            offset = N_SHARED_D + i * N_PH
            tau_washout, paco2_0 = flat[offset : offset + N_PH]

            pred = predict_v4(
                h["t"], aa, pvo2, tau_washout, gamma,
                paco2_0, k_co2, r_offset, tau_fast, tau_slow, a_fast,
                d, tau_f, h["t_end"],
            )
            mask = nadir_window_mask(h["t"], h["t_end"])
            w = np.where(h["spo2"][mask] < 95, 3.0, 1.0)
            total += np.sum(w * (h["spo2"][mask] - pred[mask]) ** 2)

            total += nadir_timing_penalty_squared(h["t"], pred, nadir_info[h["id"]]["t_nadir"])
            total += LAMBDA_PACO2 * (paco2_0 - 40.0) ** 2

        total += LAMBDA_K_CO2 * (k_co2 - 0.06) ** 2
        total += LAMBDA_GAMMA * (gamma - 1.0) ** 2
        total += LAMBDA_D_PRIOR * (d - D_PRIOR_CENTER) ** 2
        total += ic_regularization(flat, N_SHARED_D, fit_holds, type_groups)

        return total

    res = differential_evolution(
        objective, bounds, maxiter=8000, seed=42, tol=1e-10,
        polish=True, popsize=80, mutation=(0.5, 1.5), recombination=0.9,
    )
    print(f"  Converged: {res.success}, fun={res.fun:.2f}, nfev={res.nfev}")
    return res.x, res.success


# ── d-Profile (Exp A structure) ──────────────────────────────────────────────


def run_d_profile(fit_holds, nadir_info, d_values):
    """Fix d at each value, re-optimize using Exp A structure."""
    d_idx = 5  # d is 6th shared param (0-indexed)

    type_groups = {}
    for i, h in enumerate(fit_holds):
        type_groups.setdefault(h["type"], []).append(i)

    results = {}
    for d_fixed in d_values:
        fixed_shared = list(SHARED_A_BOUNDS)
        fixed_shared[d_idx] = (d_fixed - 0.01, d_fixed + 0.01)
        bounds = list(fixed_shared)
        for h in fit_holds:
            bounds.extend(PERHOLD_BOUNDS[h["type"]])

        def objective(flat, _tg=type_groups):
            aa, pvo2, k_co2, r_offset, tau_reoxy, d, tau_f, gamma = flat[:N_SHARED_A]
            total = 0.0

            for i, h in enumerate(fit_holds):
                offset = N_SHARED_A + i * N_PH
                tau_washout, paco2_0 = flat[offset : offset + N_PH]

                pred = predict_v3(
                    h["t"], aa, pvo2, tau_washout, gamma,
                    paco2_0, k_co2, r_offset, tau_reoxy, d, tau_f, h["t_end"],
                )
                mask = nadir_window_mask(h["t"], h["t_end"])
                w = np.where(h["spo2"][mask] < 95, 3.0, 1.0)
                total += np.sum(w * (h["spo2"][mask] - pred[mask]) ** 2)

                total += nadir_timing_penalty_squared(
                    h["t"], pred, nadir_info[h["id"]]["t_nadir"],
                )
                total += LAMBDA_PACO2 * (paco2_0 - 40.0) ** 2

            # Same priors as Exp A but without d prior (d is fixed)
            total += LAMBDA_K_CO2 * (k_co2 - 0.06) ** 2
            total += LAMBDA_GAMMA * (gamma - 1.0) ** 2

            for ht, indices in _tg.items():
                if len(indices) < 2:
                    continue
                for p_off in range(N_PH):
                    values = [flat[N_SHARED_A + idx * N_PH + p_off] for idx in indices]
                    mean_val = np.mean(values)
                    total += LAMBDA_REG * sum((v - mean_val) ** 2 for v in values)
            return total

        res = differential_evolution(
            objective, bounds, maxiter=4000, seed=42, tol=1e-10,
            polish=True, popsize=40, mutation=(0.5, 1.5), recombination=0.9,
        )
        results[d_fixed] = {"flat": res.x, "loss": res.fun, "success": res.success}
        print(f"    d={d_fixed:5.1f}: loss={res.fun:.2f}, nfev={res.nfev}", flush=True)

    return results


# ── Evaluation helpers ───────────────────────────────────────────────────────


def evaluate_exp(flat, fit_holds, n_shared, shared_names, label, nadir_info,
                 all_holds=None, predict_fn=None, extra_shared=None,
                 delta_d_offset=None, ic_offset_override=None):
    """Generic evaluator for all experiments.

    For Exp C: delta_d_offset gives index where delta_d values start,
    ic_offset_override gives where per-hold ICs start.
    For Exp D: predict_fn=predict_v4, extra_shared contains tau_slow/A_fast.
    """
    if predict_fn is None:
        predict_fn = predict_v3
    results = []
    target_holds = all_holds if all_holds is not None else fit_holds
    fit_ids = {h["id"] for h in fit_holds}
    ic_offset = ic_offset_override if ic_offset_override is not None else n_shared

    for h in target_holds:
        if h["id"] not in fit_ids:
            # Excluded hold — use type-mean ICs
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

        # Extract shared params
        shared = flat[:n_shared]
        aa = shared[0]
        pvo2 = shared[1]
        k_co2 = shared[2]
        r_offset = shared[3]
        tau_reoxy = shared[4]
        d = shared[5]
        tau_f = shared[6]
        gamma = shared[7]

        # Handle per-hold delay (Exp C)
        if delta_d_offset is not None and hold_idx is not None:
            delta_d = flat[delta_d_offset + hold_idx]
            d_eff = max(d + delta_d, 0.1)
        elif delta_d_offset is not None:
            d_eff = d  # excluded holds use base d
        else:
            d_eff = d
            delta_d = 0.0 if delta_d_offset is None else 0.0

        pao2_0 = corrected_pao2_0(paco2_0, aa)

        # Build prediction args based on model
        if predict_fn == predict_v4:
            a_fast = shared[8]
            tau_slow = shared[9]
            pred_full = predict_v4(
                h["t"], aa, pvo2, tau_washout, gamma,
                paco2_0, k_co2, r_offset, tau_reoxy, tau_slow, a_fast,
                d_eff, tau_f, h["t_end"],
            )
            pred_apnea = predict_v4(
                h["t_apnea"], aa, pvo2, tau_washout, gamma,
                paco2_0, k_co2, r_offset, tau_reoxy, tau_slow, a_fast,
                d_eff, tau_f, h["t_end"],
            )
        else:
            pred_full = predict_v3(
                h["t"], aa, pvo2, tau_washout, gamma,
                paco2_0, k_co2, r_offset, tau_reoxy, d_eff, tau_f, h["t_end"],
            )
            pred_apnea = predict_v3(
                h["t_apnea"], aa, pvo2, tau_washout, gamma,
                paco2_0, k_co2, r_offset, tau_reoxy, d_eff, tau_f, h["t_end"],
            )

        r2_full = compute_r2(h["spo2"], pred_full)
        r2_apnea = compute_r2(h["spo2_apnea"], pred_apnea)

        # Nadir window
        mask = nadir_window_mask(h["t"], h["t_end"])
        r2_nadir = compute_r2(h["spo2"][mask], pred_full[mask]) if mask.sum() > 3 else None

        # Recovery only
        r2_recovery = None
        if len(h["t_recovery"]) > 3:
            if predict_fn == predict_v4:
                pred_rec = predict_v4(
                    h["t_recovery"], aa, pvo2, tau_washout, gamma,
                    paco2_0, k_co2, r_offset, tau_reoxy, tau_slow, a_fast,
                    d_eff, tau_f, h["t_end"],
                )
            else:
                pred_rec = predict_v3(
                    h["t_recovery"], aa, pvo2, tau_washout, gamma,
                    paco2_0, k_co2, r_offset, tau_reoxy, d_eff, tau_f, h["t_end"],
                )
            r2_recovery = compute_r2(h["spo2_recovery"], pred_rec)

        # Nadir timing error
        t_nadir_obs = nadir_info[h["id"]]["t_nadir"]
        t_nadir_pred = h["t"][np.argmin(pred_full)]
        nadir_err = t_nadir_pred - t_nadir_obs

        # Bound checks on per-hold params
        ph_bounds = PERHOLD_BOUNDS[h["type"]]
        at_bounds = []
        for val, (lo, hi), name in zip([tau_washout, paco2_0], ph_bounds, PERHOLD_NAMES):
            if is_at_bound(val, lo, hi):
                at_bounds.append(f"{name}={'lo' if abs(val - lo) < 1e-3 else 'hi'}")

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
            "d_eff": d_eff,
        }
        if delta_d_offset is not None and hold_idx is not None:
            rec["delta_d"] = delta_d
        results.append(rec)
    return results


# ── Output functions ─────────────────────────────────────────────────────────


def print_shared_params(flat, bounds, names, label):
    """Print shared parameter values with bounds."""
    n = len(bounds)
    shared = flat[:n]
    print(f"\n  {label}:")
    for name, val, (lo, hi) in zip(names, shared, bounds):
        flag = " ** AT BOUND **" if is_at_bound(val, lo, hi) else ""
        print(f"    {name:>12s} = {val:8.4f}  [{lo:>6.2f}, {hi:>6.2f}]{flag}")


def print_perhold_ics(flat, ic_offset, fit_holds, label):
    """Print per-hold IC parameters with derived PaO2_0."""
    aa = flat[0]  # Aa is always first shared param
    print(f"\n  {label} — Per-hold ICs (Aa={aa:.2f} shared):")
    for i, h in enumerate(fit_holds):
        offset = ic_offset + i * N_PH
        tau_washout, paco2_0 = flat[offset : offset + N_PH]
        pao2_0 = corrected_pao2_0(paco2_0, aa)
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

    # Averages (fitted only)
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
    """Per-hold detail plots: 4 panels (A, B, C, D) per hold."""
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
            elif len(r["pred_full"]) == len(h["t_apnea"]):
                t_plot = h["t_apnea"]
            else:
                t_plot = np.arange(len(r["pred_full"]))

            r2_str = f"R2a={r['r2_apnea']:.3f}"
            if r["r2_nadir"] is not None:
                r2_str += f",nw={r['r2_nadir']:.3f}"
            r2_str += f",ne={r['nadir_err']:+.1f}s"
            ax.plot(t_plot, r["pred_full"], color=colors[i % len(colors)],
                    linewidth=1.5, alpha=0.8, label=f"{vn} ({r2_str})")

            if len(r["pred_full"]) == len(h["t"]):
                ax.plot(r["t_nadir_pred"], r["pred_full"][np.argmin(r["pred_full"])],
                        marker="v", color=colors[i % len(colors)], markersize=8, zorder=4)

        tag = " [EXCLUDED]" if hid in EXCLUDED_IDS else ""
        ax.set_title(f"{h['type']}#{hid}{tag}", fontsize=13, fontweight="bold",
                     color="red" if hid in EXCLUDED_IDS else "black")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("SpO2 (%)")
        ax.set_ylim(30, 105)
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(True, alpha=0.3)

    fig.suptitle("v6.04: Stabilize Sensor Pipeline — Per-Hold Detail",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {output_path}")


def plot_d_profile(results_per_d, output_path):
    """Plot loss and key params vs fixed d."""
    d_list = sorted(results_per_d.keys())

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    configs = [
        (None, "loss", "Total Loss"),
        (0, "Aa", "A-a Gradient (mmHg)"),
        (1, "pvo2", "PvO2 (mmHg)"),
        (2, "k_co2", "k_CO2 (mmHg/s)"),
        (6, "tau_f", "tau_f (filter, s)"),
        (7, "gamma", "gamma"),
    ]

    for i, (pidx, name, title) in enumerate(configs):
        ax = axes[i]
        if pidx is not None:
            vals = [results_per_d[d]["flat"][pidx] for d in d_list]
        else:
            vals = [results_per_d[d]["loss"] for d in d_list]
        ax.plot(d_list, vals, "o-", color="#1f77b4", linewidth=2, markersize=6)
        ax.set_xlabel("d (fixed delay, s)")
        ax.set_ylabel(name)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)

    fig.suptitle("v6.04: d-Profile Check (Exp A structure)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"d-profile plot saved to {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    print("=" * 140)
    print("v6.04: Stabilize Sensor Pipeline + Recovery Physiology")
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
    print(f"\nChanges from v6.03:")
    print(f"  1. tau_f constrained to (1, 8) [was (1, 30)]")
    print(f"  2. gamma widened to (0.8, 3.0) [was (0.8, 1.5)]")
    print(f"  3. gamma prior reduced to lambda={LAMBDA_GAMMA} [was 5000]")
    print(f"  4. d prior toward {D_PRIOR_CENTER} (lambda={LAMBDA_D_PRIOR})")
    print(f"  5. Huber nadir penalty option (delta={HUBER_DELTA}s)")
    print(f"  6. Per-hold hierarchical delay option (shrinkage={LAMBDA_SHRINK})")
    print(f"  7. Two-timescale recovery option")

    # ── Exp A: Constrained tau_f + widened gamma ─────────────────────────────
    print(f"\n{'='*140}")
    print("EXP A: Constrained tau_f + Widened Gamma (18 params)")
    print(f"{'='*140}")

    flat_a, conv_a = run_exp_a(fit_holds, nadir_info)
    print_shared_params(flat_a, SHARED_A_BOUNDS, SHARED_A_NAMES, "Exp A")
    print_perhold_ics(flat_a, N_SHARED_A, fit_holds, "Exp A")

    eval_a = evaluate_exp(
        flat_a, fit_holds, N_SHARED_A, SHARED_A_NAMES, "A:constrained",
        nadir_info, all_holds,
    )

    # ── Exp B: Huber nadir penalty ───────────────────────────────────────────
    print(f"\n{'='*140}")
    print("EXP B: Huber Nadir Penalty (18 params)")
    print(f"{'='*140}")

    flat_b, conv_b = run_exp_b(fit_holds, nadir_info)
    print_shared_params(flat_b, SHARED_A_BOUNDS, SHARED_A_NAMES, "Exp B")
    print_perhold_ics(flat_b, N_SHARED_A, fit_holds, "Exp B")

    eval_b = evaluate_exp(
        flat_b, fit_holds, N_SHARED_A, SHARED_A_NAMES, "B:huber",
        nadir_info, all_holds,
    )

    # ── Exp C: Hierarchical per-hold delay ───────────────────────────────────
    print(f"\n{'='*140}")
    print("EXP C: Hierarchical Per-Hold Delay (23 params)")
    print(f"{'='*140}")

    flat_c, conv_c = run_exp_c(fit_holds, nadir_info)
    n_holds = len(fit_holds)
    ic_offset_c = N_SHARED_A + n_holds

    print_shared_params(flat_c, SHARED_A_BOUNDS, SHARED_A_NAMES, "Exp C (shared)")
    print_perhold_ics(flat_c, ic_offset_c, fit_holds, "Exp C")

    # Print per-hold effective delays
    d_0_c = flat_c[5]
    print(f"\n  Exp C — Per-hold effective delays (d_0={d_0_c:.2f}):")
    for i, h in enumerate(fit_holds):
        delta_d = flat_c[N_SHARED_A + i]
        d_eff = max(d_0_c + delta_d, 0.1)
        bound_str = " *BOUND*" if is_at_bound(delta_d, *DELTA_D_BOUNDS) else ""
        print(f"    {h['type']}#{h['id']}: delta_d={delta_d:+6.2f}, "
              f"d_eff={d_eff:6.2f}{bound_str}")

    eval_c = evaluate_exp(
        flat_c, fit_holds, N_SHARED_A, SHARED_A_NAMES, "C:hierarchical",
        nadir_info, all_holds, delta_d_offset=N_SHARED_A, ic_offset_override=ic_offset_c,
    )

    # ── Exp D: Two-timescale recovery ────────────────────────────────────────
    print(f"\n{'='*140}")
    print("EXP D: Two-Timescale Recovery (20 params)")
    print(f"{'='*140}")

    flat_d, conv_d = run_exp_d(fit_holds, nadir_info)
    print_shared_params(flat_d, SHARED_D_BOUNDS, SHARED_D_NAMES, "Exp D")
    print_perhold_ics(flat_d, N_SHARED_D, fit_holds, "Exp D")

    eval_d = evaluate_exp(
        flat_d, fit_holds, N_SHARED_D, SHARED_D_NAMES, "D:two_timescale",
        nadir_info, all_holds, predict_fn=predict_v4,
    )

    # ── Comparison table ─────────────────────────────────────────────────────
    variant_names = ["A:constrained", "B:huber", "C:hierarchical", "D:two_timescale"]
    all_results = eval_a + eval_b + eval_c + eval_d
    print_comparison_table(all_results, variant_names)

    # ── Cross-experiment parameter comparison ────────────────────────────────
    print(f"\n{'='*140}")
    print("PARAMETER COMPARISON (A vs B vs C vs D)")
    print(f"{'='*140}")

    print(f"\n  {'Param':<12s} | {'Exp A':>10s} | {'Exp B':>10s} | {'Exp C':>10s} | {'Exp D':>10s}")
    print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for i, name in enumerate(SHARED_A_NAMES):
        va = flat_a[i]
        vb = flat_b[i]
        vc = flat_c[i]
        vd = flat_d[i] if i < N_SHARED_D else float("nan")
        print(f"  {name:<12s} | {va:10.4f} | {vb:10.4f} | {vc:10.4f} | {vd:10.4f}")
    # Exp D extra params
    if N_SHARED_D > N_SHARED_A:
        for i in range(N_SHARED_A, N_SHARED_D):
            name = SHARED_D_NAMES[i]
            vd = flat_d[i]
            print(f"  {name:<12s} | {'---':>10s} | {'---':>10s} | {'---':>10s} | {vd:10.4f}")

    # ── d-Profile check ──────────────────────────────────────────────────────
    print(f"\n{'='*140}")
    print("d-PROFILE CHECK (4-point sweep, Exp A structure)")
    print(f"{'='*140}")

    d_sweep = [10.0, 13.0, 16.0, 20.0]
    d_profile = run_d_profile(fit_holds, nadir_info, d_sweep)

    print(f"\n  {'d':>5s} | {'loss':>10s} | {'Aa':>8s} | {'pvo2':>8s} | "
          f"{'k_co2':>8s} | {'tau_f':>8s} | {'gamma':>8s}")
    print(f"  {'-'*5}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for d_fixed in sorted(d_profile.keys()):
        r = d_profile[d_fixed]
        s = r["flat"][:N_SHARED_A]
        aa, pvo2, k_co2, r_off, tau_re, d, tau_f, gamma = s
        print(f"  {d_fixed:5.1f} | {r['loss']:10.2f} | {aa:8.2f} | {pvo2:8.2f} | "
              f"{k_co2:8.4f} | {tau_f:8.2f} | {gamma:8.4f}")

    losses = [d_profile[d]["loss"] for d in sorted(d_profile.keys())]
    is_monotone = all(losses[i] >= losses[i + 1] for i in range(len(losses) - 1))
    min_d = sorted(d_profile.keys())[np.argmin(losses)]
    print(f"\n  Loss monotonically decreasing: {'YES (degenerate!)' if is_monotone else 'NO (good!)'}")
    print(f"  Minimum loss at d={min_d:.1f}")

    # ── Plots ────────────────────────────────────────────────────────────────
    output_dir = Path(__file__).resolve().parent

    plot_per_hold_detail(
        all_results, all_holds, variant_names, nadir_info,
        output_dir / "exp_v6_04_stabilize.png",
    )

    plot_d_profile(d_profile, output_dir / "exp_v6_04_d_profile.png")

    # ── Success criteria ─────────────────────────────────────────────────────
    print(f"\n{'='*140}")
    print("SUCCESS CRITERIA CHECK")
    print(f"{'='*140}")

    # 1. Exp A: tau_f interior (< 10s)
    tau_f_a = flat_a[6]
    print(f"\n  1. [Exp A] tau_f={tau_f_a:.2f} (interior, < 8): "
          f"{'PASS' if not is_at_bound(tau_f_a, 1, 8) else 'FAIL (at bound)'}")

    # 2. Exp A: gamma < 2.5
    gamma_a = flat_a[7]
    print(f"  2. [Exp A] gamma={gamma_a:.4f} (< 2.5): "
          f"{'PASS' if gamma_a < 2.5 else 'FAIL'}")

    # 3. Exp A: d stays 13-18
    d_a = flat_a[5]
    print(f"  3. [Exp A] d={d_a:.2f} (13-18): "
          f"{'PASS' if 13 <= d_a <= 18 else 'FAIL'}")

    # 4. Exp A: R2(apnea) avg >= 0.93
    r2_apnea_a = [r["r2_apnea"] for r in eval_a if not r["is_excluded"]]
    avg_r2a_a = np.mean(r2_apnea_a)
    print(f"  4. [Exp A] R2(apnea) avg={avg_r2a_a:.4f} (>= 0.93): "
          f"{'PASS' if avg_r2a_a >= 0.93 else 'FAIL'}")

    # 5. Exp B: Timing error avg < 8s
    nerrs_b = [abs(r["nadir_err"]) for r in eval_b if not r["is_excluded"]]
    avg_nerr_b = np.mean(nerrs_b)
    print(f"  5. [Exp B] Nadir timing error avg={avg_nerr_b:.1f}s (< 8s): "
          f"{'PASS' if avg_nerr_b < 8 else 'FAIL'}")

    # 6. Exp C: Timing error avg < 5s
    nerrs_c = [abs(r["nadir_err"]) for r in eval_c if not r["is_excluded"]]
    avg_nerr_c = np.mean(nerrs_c)
    print(f"  6. [Exp C] Nadir timing error avg={avg_nerr_c:.1f}s (< 5s): "
          f"{'PASS' if avg_nerr_c < 5 else 'FAIL'}")

    # 7. Exp C: delta_d spread interpretable
    delta_ds = [flat_c[N_SHARED_A + i] for i in range(n_holds)]
    print(f"  7. [Exp C] delta_d range=[{min(delta_ds):+.2f}, {max(delta_ds):+.2f}] "
          f"(interpretable: some variation expected)")

    # 8. Exp D: tau_f interior (< 10s) without explicit constraint
    tau_f_d = flat_d[6]
    print(f"  8. [Exp D] tau_f={tau_f_d:.2f} (interior, < 10 without constraint): "
          f"{'PASS' if tau_f_d < 10 else 'FAIL (still maximized)'}")

    # 9. All exps: k_co2 interior (0.04-0.20)
    all_k_co2 = [flat_a[2], flat_b[2], flat_c[2], flat_d[2]]
    all_interior = all(0.04 <= k <= 0.20 for k in all_k_co2)
    print(f"  9. [All] k_co2 values: {[f'{k:.4f}' for k in all_k_co2]} "
          f"(all in 0.04-0.20): {'PASS' if all_interior else 'FAIL'}")

    # 10. d-profile: non-monotone with minimum near 15
    print(f" 10. [d-profile] Non-monotone: {'PASS' if not is_monotone else 'FAIL'}, "
          f"minimum at d={min_d:.1f} "
          f"({'PASS' if 12 <= min_d <= 18 else 'FAIL'} — near 15)")


if __name__ == "__main__":
    main()
