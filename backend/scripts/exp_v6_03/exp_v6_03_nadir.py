"""
v6 Experiment 3: Nadir-Window Sensor Identification + Structural Degeneracy Breaking.

v6.02 found a bifurcation at d~15-20 where the optimizer switches between two
competing regimes (low-d: high k_co2/low gamma vs high-d: low k_co2/high gamma).
This proves model class non-identifiability.

v6.03 targets the nadir timing — the V-shape where SpO2 continues falling after
apnea ends due to sensor delay — as the only feature that should identify d.

Five structural changes:
  1. Corrected AGE sign: PaO2_0 = 149.2 - PaCO2_0/0.8 - Aa (Aa >= 0, shared globally)
  2. Gamma fixed at 1.0 (Exp A/B) or free with strong prior (Exp C)
  3. Strong CO2 priors (k_co2 toward 0.06, paco2_0 toward 40)
  4. Nadir-timing penalty (predicted vs observed nadir time)
  5. Nadir-window loss (apnea + first 45s post-apnea, no late recovery)

Sub-experiments:
  A: Two-stage sensor identification (per-hold 2-param d/tau_f + global physiology)
  B: Combined model, gamma=1.0 (17 free params)
  C: Gamma as sensor calibration (18 free params)

Usage:
    cd backend && uv run python -u scripts/exp_v6_03_nadir.py
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

# ── Regularization strengths ─────────────────────────────────────────────────

LAMBDA_REG = 10.0  # per-hold IC -> type-mean
LAMBDA_NADIR = 1000.0  # nadir timing penalty per hold
LAMBDA_K_CO2 = 2000.0  # k_co2 prior toward 0.06
LAMBDA_PACO2 = 1000.0  # paco2_0 prior toward 40
LAMBDA_GAMMA_C = 5000.0  # gamma prior toward 1.0 (Exp C only)

NADIR_WINDOW_AFTER = 45  # seconds after t_end for loss window
NADIR_STAGE1_BEFORE = 30  # seconds before t_end for Stage 1 loss window

# ── Fixed physiology for Exp A Stage 1 ───────────────────────────────────────

FIXED_PVO2 = 25.0
FIXED_GAMMA = 1.0
FIXED_K_CO2 = 0.06
FIXED_R_OFFSET = 0.0
FIXED_TAU_REOXY = 10.0
FIXED_AA = 10.0
FIXED_PACO2_0 = 40.0
FIXED_TAU_WASHOUT = {"FL": 80.0, "FRC": 60.0, "RV": 20.0}

# ── Bounds ───────────────────────────────────────────────────────────────────

# Per-hold ICs: tau_washout, paco2_0
PERHOLD_BOUNDS = {
    "FL": [(50, 250), (20, 50)],
    "FRC": [(20, 100), (25, 50)],
    "RV": [(10, 80), (30, 55)],
}
PERHOLD_NAMES = ["tau_washout", "paco2_0"]
N_PH = len(PERHOLD_NAMES)

# Exp B shared: Aa, pvo2, k_co2, r_offset, tau_reoxy, d, tau_f
SHARED_B_BOUNDS = [(0, 30), (15, 50), (0.02, 0.25), (-3, 3), (5, 30), (1, 30), (1, 30)]
SHARED_B_NAMES = ["Aa", "pvo2", "k_co2", "r_offset", "tau_reoxy", "d", "tau_f"]
N_SHARED_B = len(SHARED_B_BOUNDS)

# Exp C shared: same + gamma
SHARED_C_BOUNDS = SHARED_B_BOUNDS + [(0.8, 1.5)]
SHARED_C_NAMES = SHARED_B_NAMES + ["gamma"]
N_SHARED_C = len(SHARED_C_BOUNDS)

# Exp A Stage 2 shared: Aa, pvo2, k_co2, r_offset, tau_reoxy (d, tau_f fixed)
SHARED_A2_BOUNDS = [(0, 30), (15, 50), (0.02, 0.25), (-3, 3), (5, 30)]
SHARED_A2_NAMES = ["Aa", "pvo2", "k_co2", "r_offset", "tau_reoxy"]
N_SHARED_A2 = len(SHARED_A2_BOUNDS)


# ── Data loading (reused from v6.02) ─────────────────────────────────────────


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


# ── Physiology functions (reused from v6.02) ─────────────────────────────────


def corrected_pao2_0(paco2_0, aa):
    """Derive initial PaO2 from PaCO2 and A-a gradient (corrected sign).

    PaO2_0 = FIO2*(PB-PH2O) - PaCO2_0/RQ - Aa
    where Aa >= 0 (always, since PaO2 <= PAO2).
    """
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


# ── Core predict function (v3: corrected AGE, gamma as arg) ─────────────────


def predict_v3(t, aa, pvo2, tau_washout, gamma, paco2_0, k_co2, r_offset, tau_reoxy, d, tau_f, t_end):
    """Full sensor pipeline with corrected AGE.

    PaO2_0 derived from paco2_0 + Aa via corrected alveolar gas equation.
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


# ── Nadir helpers ────────────────────────────────────────────────────────────


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


def stage1_window_mask(t, t_end, before=NADIR_STAGE1_BEFORE, after=NADIR_WINDOW_AFTER):
    """Boolean mask for narrow nadir window: [t_end - before, t_end + after]."""
    return (t >= t_end - before) & (t <= t_end + after)


def nadir_timing_penalty(t, pred, t_nadir_obs):
    """Penalty for mismatch between predicted and observed nadir timing."""
    t_nadir_pred = t[np.argmin(pred)]
    return LAMBDA_NADIR * (t_nadir_pred - t_nadir_obs) ** 2


# ── Metrics ──────────────────────────────────────────────────────────────────


def compute_r2(obs, pred):
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def compute_rmse(obs, pred):
    return float(np.sqrt(np.mean((obs - pred) ** 2)))


def is_at_bound(val, lo, hi, tol=1e-3):
    return abs(val - lo) < tol or abs(val - hi) < tol


# ── Exp A: Two-Stage Sensor Identification ───────────────────────────────────


def run_exp_a_stage1(holds, nadir_info):
    """Stage 1: Per-hold 2-param fits (d, tau_f) with fixed physiology.

    Fits only on nadir window [t_end-30, t_end+45] with nadir timing penalty.
    """
    print("\n  Stage 1: Per-hold d/tau_f identification (2 params each)")
    print(f"  Fixed physiology: pvo2={FIXED_PVO2}, gamma={FIXED_GAMMA}, k_co2={FIXED_K_CO2}, "
          f"r_offset={FIXED_R_OFFSET}, tau_reoxy={FIXED_TAU_REOXY}, Aa={FIXED_AA}, paco2_0={FIXED_PACO2_0}")
    print(f"  Fixed tau_washout by type: {FIXED_TAU_WASHOUT}")
    print(f"  Loss window: [t_end-{NADIR_STAGE1_BEFORE}, t_end+{NADIR_WINDOW_AFTER}]")

    bounds_s1 = [(1, 30), (1, 30)]  # d, tau_f
    results = {}

    for h in holds:
        t_nadir_obs = nadir_info[h["id"]]["t_nadir"]
        tau_w = FIXED_TAU_WASHOUT[h["type"]]

        def objective(x, _h=h, _tw=tau_w, _tno=t_nadir_obs):
            d, tau_f = x
            pred = predict_v3(
                _h["t"], FIXED_AA, FIXED_PVO2, _tw, FIXED_GAMMA,
                FIXED_PACO2_0, FIXED_K_CO2, FIXED_R_OFFSET, FIXED_TAU_REOXY,
                d, tau_f, _h["t_end"],
            )
            mask = stage1_window_mask(_h["t"], _h["t_end"])
            if mask.sum() < 3:
                return 1e10
            w = np.where(_h["spo2"][mask] < 95, 3.0, 1.0)
            sse = np.sum(w * (_h["spo2"][mask] - pred[mask]) ** 2)
            sse += nadir_timing_penalty(_h["t"], pred, _tno)
            return sse

        res = differential_evolution(
            objective, bounds_s1, maxiter=2000, seed=42, tol=1e-10,
            polish=True, popsize=15, mutation=(0.5, 1.5), recombination=0.9,
        )
        d_fit, tau_f_fit = res.x
        results[h["id"]] = {
            "d": d_fit, "tau_f": tau_f_fit, "loss": res.fun,
            "success": res.success, "hold": h,
        }
        d_bound = " *BOUND*" if is_at_bound(d_fit, 1, 30) else ""
        tf_bound = " *BOUND*" if is_at_bound(tau_f_fit, 1, 30) else ""
        print(f"    {h['type']}#{h['id']}: d={d_fit:6.2f}{d_bound}, "
              f"tau_f={tau_f_fit:6.2f}{tf_bound}, loss={res.fun:.1f}")

    return results


def run_exp_a_stage2(holds, hold_types, d_fixed, tau_f_fixed, nadir_info):
    """Stage 2: Global physiology fit with fixed sensor params.

    Shared: Aa, pvo2, k_co2, r_offset, tau_reoxy (5 params)
    Per-type: tau_washout, paco2_0 (2 per type)
    Gamma fixed at 1.0, d/tau_f from Stage 1.
    Loss on apnea only.
    """
    n_types = len(hold_types)
    bounds = list(SHARED_A2_BOUNDS)
    for ht in hold_types:
        bounds.extend(PERHOLD_BOUNDS[ht])
    n_total = len(bounds)

    holds_by_type = {}
    for h in holds:
        holds_by_type.setdefault(h["type"], []).append(h)

    print(f"\n  Stage 2: Global physiology fit ({n_total} params = "
          f"{N_SHARED_A2} shared + {N_PH}x{n_types} per-type)")
    print(f"  Fixed sensor: d={d_fixed:.2f}, tau_f={tau_f_fixed:.2f}, gamma={FIXED_GAMMA}")
    print(f"  Loss on apnea only")

    def objective(flat):
        aa, pvo2, k_co2, r_offset, tau_reoxy = flat[:N_SHARED_A2]
        total = 0.0

        for ti, ht in enumerate(hold_types):
            offset = N_SHARED_A2 + ti * N_PH
            tau_washout, paco2_0 = flat[offset : offset + N_PH]
            for h in holds_by_type.get(ht, []):
                pred = predict_v3(
                    h["t_apnea"], aa, pvo2, tau_washout, FIXED_GAMMA,
                    paco2_0, k_co2, r_offset, tau_reoxy, d_fixed, tau_f_fixed, h["t_end"],
                )
                w = np.where(h["spo2_apnea"] < 95, 3.0, 1.0)
                total += np.sum(w * (h["spo2_apnea"] - pred) ** 2)

            # CO2 prior on paco2_0
            total += LAMBDA_PACO2 * (paco2_0 - 40.0) ** 2

        # k_co2 prior
        total += LAMBDA_K_CO2 * (k_co2 - 0.06) ** 2
        return total

    res = differential_evolution(
        objective, bounds, maxiter=4000, seed=42, tol=1e-10,
        polish=True, popsize=40, mutation=(0.5, 1.5), recombination=0.9,
    )
    print(f"  Converged: {res.success}, fun={res.fun:.2f}, nfev={res.nfev}")
    return res.x, res.success


# ── Exp B: Combined model, gamma=1.0 (17 params) ────────────────────────────


def run_exp_b(fit_holds, nadir_info):
    """Exp B: Combined fit with corrected AGE, gamma=1.0, nadir penalties.

    Shared (7): Aa, pvo2, k_co2, r_offset, tau_reoxy, d, tau_f
    Per-hold (2x5=10): tau_washout_i, paco2_0_i
    Total: 17 free params.
    """
    n_holds = len(fit_holds)
    bounds = list(SHARED_B_BOUNDS)
    for h in fit_holds:
        bounds.extend(PERHOLD_BOUNDS[h["type"]])
    n_total = len(bounds)

    type_groups = {}
    for i, h in enumerate(fit_holds):
        type_groups.setdefault(h["type"], []).append(i)

    print(f"\n  Exp B: {n_total} params ({N_SHARED_B} shared + {N_PH}x{n_holds} per-hold)")
    print(f"  gamma fixed at 1.0")
    print(f"  Loss on apnea + {NADIR_WINDOW_AFTER}s recovery (nadir window)")
    print(f"  Penalties: nadir timing, CO2 priors, per-hold IC regularization")

    def objective(flat):
        aa, pvo2, k_co2, r_offset, tau_reoxy, d, tau_f = flat[:N_SHARED_B]
        total = 0.0

        for i, h in enumerate(fit_holds):
            offset = N_SHARED_B + i * N_PH
            tau_washout, paco2_0 = flat[offset : offset + N_PH]

            pred = predict_v3(
                h["t"], aa, pvo2, tau_washout, 1.0,
                paco2_0, k_co2, r_offset, tau_reoxy, d, tau_f, h["t_end"],
            )
            mask = nadir_window_mask(h["t"], h["t_end"])
            w = np.where(h["spo2"][mask] < 95, 3.0, 1.0)
            total += np.sum(w * (h["spo2"][mask] - pred[mask]) ** 2)

            # Nadir timing penalty
            total += nadir_timing_penalty(h["t"], pred, nadir_info[h["id"]]["t_nadir"])

            # paco2_0 prior per hold
            total += LAMBDA_PACO2 * (paco2_0 - 40.0) ** 2

        # k_co2 prior (shared)
        total += LAMBDA_K_CO2 * (k_co2 - 0.06) ** 2

        # Per-hold IC regularization toward type means
        for ht, indices in type_groups.items():
            if len(indices) < 2:
                continue
            for p_off in range(N_PH):
                values = [flat[N_SHARED_B + idx * N_PH + p_off] for idx in indices]
                mean_val = np.mean(values)
                total += LAMBDA_REG * sum((v - mean_val) ** 2 for v in values)

        return total

    res = differential_evolution(
        objective, bounds, maxiter=6000, seed=42, tol=1e-10,
        polish=True, popsize=60, mutation=(0.5, 1.5), recombination=0.9,
    )
    print(f"  Converged: {res.success}, fun={res.fun:.2f}, nfev={res.nfev}")
    return res.x, res.success


# ── Exp C: Gamma as sensor calibration (18 params) ──────────────────────────


def run_exp_c(fit_holds, nadir_info):
    """Exp C: Same as B but gamma free in (0.8, 1.5) with strong prior.

    Shared (8): Aa, pvo2, k_co2, r_offset, tau_reoxy, d, tau_f, gamma
    Per-hold (2x5=10): tau_washout_i, paco2_0_i
    Total: 18 free params.
    """
    n_holds = len(fit_holds)
    bounds = list(SHARED_C_BOUNDS)
    for h in fit_holds:
        bounds.extend(PERHOLD_BOUNDS[h["type"]])
    n_total = len(bounds)

    type_groups = {}
    for i, h in enumerate(fit_holds):
        type_groups.setdefault(h["type"], []).append(i)

    print(f"\n  Exp C: {n_total} params ({N_SHARED_C} shared + {N_PH}x{n_holds} per-hold)")
    print(f"  gamma free in (0.8, 1.5) with prior lambda={LAMBDA_GAMMA_C} toward 1.0")

    def objective(flat):
        aa, pvo2, k_co2, r_offset, tau_reoxy, d, tau_f, gamma = flat[:N_SHARED_C]
        total = 0.0

        for i, h in enumerate(fit_holds):
            offset = N_SHARED_C + i * N_PH
            tau_washout, paco2_0 = flat[offset : offset + N_PH]

            pred = predict_v3(
                h["t"], aa, pvo2, tau_washout, gamma,
                paco2_0, k_co2, r_offset, tau_reoxy, d, tau_f, h["t_end"],
            )
            mask = nadir_window_mask(h["t"], h["t_end"])
            w = np.where(h["spo2"][mask] < 95, 3.0, 1.0)
            total += np.sum(w * (h["spo2"][mask] - pred[mask]) ** 2)

            # Nadir timing penalty
            total += nadir_timing_penalty(h["t"], pred, nadir_info[h["id"]]["t_nadir"])

            # paco2_0 prior per hold
            total += LAMBDA_PACO2 * (paco2_0 - 40.0) ** 2

        # k_co2 prior
        total += LAMBDA_K_CO2 * (k_co2 - 0.06) ** 2

        # gamma prior toward 1.0
        total += LAMBDA_GAMMA_C * (gamma - 1.0) ** 2

        # Per-hold IC regularization toward type means
        for ht, indices in type_groups.items():
            if len(indices) < 2:
                continue
            for p_off in range(N_PH):
                values = [flat[N_SHARED_C + idx * N_PH + p_off] for idx in indices]
                mean_val = np.mean(values)
                total += LAMBDA_REG * sum((v - mean_val) ** 2 for v in values)

        return total

    res = differential_evolution(
        objective, bounds, maxiter=6000, seed=42, tol=1e-10,
        polish=True, popsize=60, mutation=(0.5, 1.5), recombination=0.9,
    )
    print(f"  Converged: {res.success}, fun={res.fun:.2f}, nfev={res.nfev}")
    return res.x, res.success


# ── d-Profile check ─────────────────────────────────────────────────────────


def run_d_profile(fit_holds, nadir_info, d_values):
    """Fix d at each value, re-optimize using Exp B structure. Check monotonicity."""
    d_idx = 5  # d is 6th shared param (0-indexed) in SHARED_B

    type_groups = {}
    for i, h in enumerate(fit_holds):
        type_groups.setdefault(h["type"], []).append(i)

    results = {}
    for d_fixed in d_values:
        fixed_shared = list(SHARED_B_BOUNDS)
        fixed_shared[d_idx] = (d_fixed - 0.01, d_fixed + 0.01)
        bounds = list(fixed_shared)
        for h in fit_holds:
            bounds.extend(PERHOLD_BOUNDS[h["type"]])

        def objective(flat, _tg=type_groups):
            aa, pvo2, k_co2, r_offset, tau_reoxy, d, tau_f = flat[:N_SHARED_B]
            total = 0.0

            for i, h in enumerate(fit_holds):
                offset = N_SHARED_B + i * N_PH
                tau_washout, paco2_0 = flat[offset : offset + N_PH]

                pred = predict_v3(
                    h["t"], aa, pvo2, tau_washout, 1.0,
                    paco2_0, k_co2, r_offset, tau_reoxy, d, tau_f, h["t_end"],
                )
                mask = nadir_window_mask(h["t"], h["t_end"])
                w = np.where(h["spo2"][mask] < 95, 3.0, 1.0)
                total += np.sum(w * (h["spo2"][mask] - pred[mask]) ** 2)

                total += nadir_timing_penalty(h["t"], pred, nadir_info[h["id"]]["t_nadir"])
                total += LAMBDA_PACO2 * (paco2_0 - 40.0) ** 2

            total += LAMBDA_K_CO2 * (k_co2 - 0.06) ** 2

            for ht, indices in _tg.items():
                if len(indices) < 2:
                    continue
                for p_off in range(N_PH):
                    values = [flat[N_SHARED_B + idx * N_PH + p_off] for idx in indices]
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


def evaluate_bc(flat, fit_holds, n_shared, gamma_fixed, label, nadir_info, all_holds=None):
    """Evaluate Exp B or C on all holds.

    If gamma_fixed is not None, uses that value. Otherwise extracts gamma from shared params.
    """
    results = []
    target_holds = all_holds if all_holds is not None else fit_holds
    fit_ids = {h["id"] for h in fit_holds}

    for h in target_holds:
        if h["id"] not in fit_ids:
            # Excluded hold — use type-mean ICs
            shared = flat[:n_shared]
            type_indices = [i for i, fh in enumerate(fit_holds) if fh["type"] == h["type"]]
            if not type_indices:
                continue
            avg_ph = np.mean(
                [flat[n_shared + idx * N_PH : n_shared + (idx + 1) * N_PH] for idx in type_indices],
                axis=0,
            )
            tau_washout, paco2_0 = avg_ph
            is_excl = True
        else:
            idx = next(i for i, fh in enumerate(fit_holds) if fh["id"] == h["id"])
            shared = flat[:n_shared]
            ph_offset = n_shared + idx * N_PH
            tau_washout, paco2_0 = flat[ph_offset : ph_offset + N_PH]
            is_excl = False

        if n_shared == N_SHARED_B:
            aa, pvo2, k_co2, r_offset, tau_reoxy, d, tau_f = shared
            gamma = gamma_fixed if gamma_fixed is not None else 1.0
        else:  # N_SHARED_C
            aa, pvo2, k_co2, r_offset, tau_reoxy, d, tau_f, gamma = shared

        pao2_0 = corrected_pao2_0(paco2_0, aa)

        # Full prediction
        pred_full = predict_v3(
            h["t"], aa, pvo2, tau_washout, gamma,
            paco2_0, k_co2, r_offset, tau_reoxy, d, tau_f, h["t_end"],
        )
        r2_full = compute_r2(h["spo2"], pred_full)

        # Apnea only
        pred_apnea = predict_v3(
            h["t_apnea"], aa, pvo2, tau_washout, gamma,
            paco2_0, k_co2, r_offset, tau_reoxy, d, tau_f, h["t_end"],
        )
        r2_apnea = compute_r2(h["spo2_apnea"], pred_apnea)

        # Nadir window
        mask = nadir_window_mask(h["t"], h["t_end"])
        r2_nadir = compute_r2(h["spo2"][mask], pred_full[mask]) if mask.sum() > 3 else None

        # Recovery only
        r2_recovery = None
        if len(h["t_recovery"]) > 3:
            pred_rec = predict_v3(
                h["t_recovery"], aa, pvo2, tau_washout, gamma,
                paco2_0, k_co2, r_offset, tau_reoxy, d, tau_f, h["t_end"],
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

        results.append({
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
        })
    return results


def evaluate_a2(flat_a2, hold_types, fit_holds, d_fixed, tau_f_fixed, nadir_info, all_holds):
    """Evaluate Exp A Stage 2 on all holds."""
    aa, pvo2, k_co2, r_offset, tau_reoxy = flat_a2[:N_SHARED_A2]
    results = []
    fit_ids = {h["id"] for h in fit_holds}

    for h in all_holds:
        # Find type-specific ICs
        if h["type"] in hold_types:
            ti = hold_types.index(h["type"])
            offset = N_SHARED_A2 + ti * N_PH
            tau_washout, paco2_0 = flat_a2[offset : offset + N_PH]
        else:
            continue

        pao2_0 = corrected_pao2_0(paco2_0, aa)

        pred_full = predict_v3(
            h["t"], aa, pvo2, tau_washout, FIXED_GAMMA,
            paco2_0, k_co2, r_offset, tau_reoxy, d_fixed, tau_f_fixed, h["t_end"],
        )
        r2_full = compute_r2(h["spo2"], pred_full)

        pred_apnea = predict_v3(
            h["t_apnea"], aa, pvo2, tau_washout, FIXED_GAMMA,
            paco2_0, k_co2, r_offset, tau_reoxy, d_fixed, tau_f_fixed, h["t_end"],
        )
        r2_apnea = compute_r2(h["spo2_apnea"], pred_apnea)

        mask = nadir_window_mask(h["t"], h["t_end"])
        r2_nadir = compute_r2(h["spo2"][mask], pred_full[mask]) if mask.sum() > 3 else None

        r2_recovery = None
        if len(h["t_recovery"]) > 3:
            pred_rec = predict_v3(
                h["t_recovery"], aa, pvo2, tau_washout, FIXED_GAMMA,
                paco2_0, k_co2, r_offset, tau_reoxy, d_fixed, tau_f_fixed, h["t_end"],
            )
            r2_recovery = compute_r2(h["spo2_recovery"], pred_rec)

        t_nadir_obs = nadir_info[h["id"]]["t_nadir"]
        t_nadir_pred = h["t"][np.argmin(pred_full)]

        results.append({
            "variant": "A:TwoStage",
            "hold_id": h["id"],
            "hold_type": h["type"],
            "r2_full": r2_full,
            "r2_apnea": r2_apnea,
            "r2_nadir": r2_nadir,
            "r2_recovery": r2_recovery,
            "at_bounds": [],
            "pred_full": pred_full,
            "pao2_0": pao2_0,
            "tau_washout": tau_washout,
            "paco2_0": paco2_0,
            "nadir_err": t_nadir_pred - t_nadir_obs,
            "t_nadir_pred": t_nadir_pred,
            "is_excluded": h["id"] in EXCLUDED_IDS,
        })
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


def print_perhold_ics(flat, n_shared, fit_holds, label):
    """Print per-hold IC parameters with derived PaO2_0."""
    aa = flat[0]  # Aa is always first shared param in B/C
    print(f"\n  {label} — Per-hold ICs (Aa={aa:.2f} shared):")
    for i, h in enumerate(fit_holds):
        offset = n_shared + i * N_PH
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
    print(f"\n{'='*130}")
    print("PER-HOLD COMPARISON")
    print(f"{'='*130}")

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
    """Per-hold detail plots: apnea + nadir window with nadir markers."""
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

        # Observed data
        ax.plot(h["t"], h["spo2"], "k.", markersize=2, alpha=0.5, label="Observed")
        ax.axvline(x=h["t_end"], color="red", linestyle="--", alpha=0.5, label="Apnea end")

        # Nadir window bounds
        ax.axvline(x=h["t_end"] + NADIR_WINDOW_AFTER, color="gray", linestyle=":",
                   alpha=0.3, label=f"Nadir window (+{NADIR_WINDOW_AFTER}s)")

        # Observed nadir marker
        ax.plot(ni["t_nadir"], ni["spo2_nadir"], "r*", markersize=12, zorder=5,
                label=f"Obs nadir (t={ni['t_nadir']:.0f}s)")

        # Model predictions
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

            # Predicted nadir marker
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

    fig.suptitle("v6.03: Nadir-Window Sensor Identification — Per-Hold Detail",
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
        (4, "tau_reoxy", "tau_reoxy (s)"),
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

    fig.suptitle("v6.03: d-Profile Check (Exp B structure with nadir penalties)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"d-profile plot saved to {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    print("=" * 130)
    print("v6.03: Nadir-Window Sensor Identification + Structural Degeneracy Breaking")
    print("=" * 130)

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
    hold_types = sorted(set(h["type"] for h in fit_holds))
    holds_by_type = {}
    for h in fit_holds:
        holds_by_type.setdefault(h["type"], []).append(h)

    print(f"\nFitting on {len(fit_holds)} holds (types: {hold_types}), "
          f"excluding {sum(1 for h in all_holds if h['id'] in EXCLUDED_IDS)} hold(s)")

    # ── Compute nadir info ───────────────────────────────────────────────────
    print(f"\n{'='*130}")
    print("OBSERVED NADIR TIMING")
    print(f"{'='*130}")

    nadir_info = {}
    for h in all_holds:
        ni = compute_nadir_info(h)
        nadir_info[h["id"]] = ni
        tag = " [EXCLUDED]" if h["id"] in EXCLUDED_IDS else ""
        loc = "recovery" if ni["in_recovery"] else "apnea"
        print(f"  {h['type']}#{h['id']}{tag}: nadir at t={ni['t_nadir']:.0f}s "
              f"(SpO2={ni['spo2_nadir']:.0f}%, {loc}, "
              f"delay_from_end={ni['delay_from_end']:+.0f}s)")

    # ── Structural changes summary ───────────────────────────────────────────
    print(f"\nStructural changes (v6.03):")
    print(f"  1. Corrected AGE: PaO2_0 = {FIO2_PB_PH2O} - PaCO2_0/{RQ} - Aa (Aa >= 0, shared)")
    print(f"  2. Gamma fixed at 1.0 (Exp A/B) or free with prior lambda={LAMBDA_GAMMA_C} (Exp C)")
    print(f"  3. CO2 priors: k_co2 toward 0.06 (lambda={LAMBDA_K_CO2}), "
          f"paco2_0 toward 40 (lambda={LAMBDA_PACO2})")
    print(f"  4. Nadir timing penalty (lambda={LAMBDA_NADIR})")
    print(f"  5. Nadir window loss: apnea + {NADIR_WINDOW_AFTER}s recovery (no late recovery)")

    # ── Exp A: Two-Stage ─────────────────────────────────────────────────────
    print(f"\n{'='*130}")
    print("EXP A: Two-Stage Sensor Identification")
    print(f"{'='*130}")

    s1_results = run_exp_a_stage1(fit_holds, nadir_info)

    # Compute median d, tau_f from Stage 1
    d_vals = [r["d"] for r in s1_results.values()]
    tf_vals = [r["tau_f"] for r in s1_results.values()]
    d_median = float(np.median(d_vals))
    tau_f_median = float(np.median(tf_vals))
    d_spread = max(d_vals) - min(d_vals)

    print(f"\n  Stage 1 summary:")
    print(f"    d:     median={d_median:.2f}, range=[{min(d_vals):.2f}, {max(d_vals):.2f}], "
          f"spread={d_spread:.2f}")
    print(f"    tau_f: median={tau_f_median:.2f}, range=[{min(tf_vals):.2f}, {max(tf_vals):.2f}]")

    # Stage 2
    flat_a2, conv_a2 = run_exp_a_stage2(
        fit_holds, hold_types, d_median, tau_f_median, nadir_info,
    )
    print_shared_params(flat_a2, SHARED_A2_BOUNDS, SHARED_A2_NAMES, "Exp A Stage 2")

    # Print type-specific ICs
    aa_a2 = flat_a2[0]
    print(f"\n  Exp A Stage 2 — Type-specific ICs (Aa={aa_a2:.2f}):")
    for ti, ht in enumerate(hold_types):
        offset = N_SHARED_A2 + ti * N_PH
        tw, pc = flat_a2[offset : offset + N_PH]
        p0 = corrected_pao2_0(pc, aa_a2)
        print(f"    {ht}: tau_w={tw:.1f}, paco2_0={pc:.1f}, PaO2_0={p0:.1f}")

    eval_a = evaluate_a2(flat_a2, hold_types, fit_holds, d_median, tau_f_median,
                         nadir_info, all_holds)

    # ── Exp B: Combined, gamma=1.0 ──────────────────────────────────────────
    print(f"\n{'='*130}")
    print("EXP B: Combined Model, gamma=1.0 (17 params)")
    print(f"{'='*130}")

    flat_b, conv_b = run_exp_b(fit_holds, nadir_info)
    print_shared_params(flat_b, SHARED_B_BOUNDS, SHARED_B_NAMES, "Exp B")
    print_perhold_ics(flat_b, N_SHARED_B, fit_holds, "Exp B")

    eval_b = evaluate_bc(flat_b, fit_holds, N_SHARED_B, 1.0, "B:gamma=1.0",
                         nadir_info, all_holds)

    # ── Exp C: Gamma free ────────────────────────────────────────────────────
    print(f"\n{'='*130}")
    print("EXP C: Gamma as Sensor Calibration (18 params)")
    print(f"{'='*130}")

    flat_c, conv_c = run_exp_c(fit_holds, nadir_info)
    print_shared_params(flat_c, SHARED_C_BOUNDS, SHARED_C_NAMES, "Exp C")
    print_perhold_ics(flat_c, N_SHARED_C, fit_holds, "Exp C")

    eval_c = evaluate_bc(flat_c, fit_holds, N_SHARED_C, None, "C:gamma_free",
                         nadir_info, all_holds)

    # ── Comparison table ─────────────────────────────────────────────────────
    variant_names = ["A:TwoStage", "B:gamma=1.0", "C:gamma_free"]
    all_results = eval_a + eval_b + eval_c
    print_comparison_table(all_results, variant_names)

    # B vs C param comparison
    print(f"\n{'='*130}")
    print("EXP B vs C: PARAMETER COMPARISON")
    print(f"{'='*130}")
    shared_b = flat_b[:N_SHARED_B]
    shared_c = flat_c[:N_SHARED_C]
    print(f"\n  {'Param':<12s} | {'Exp B':>10s} | {'Exp C':>10s} | {'Diff':>10s}")
    print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for i, name in enumerate(SHARED_B_NAMES):
        vb = shared_b[i]
        vc = shared_c[i]
        print(f"  {name:<12s} | {vb:10.4f} | {vc:10.4f} | {vc - vb:+10.4f}")
    gamma_c = shared_c[7]
    print(f"  {'gamma':<12s} | {'1.0 (fixed)':>10s} | {gamma_c:10.4f} |")

    # ── d-Profile check ──────────────────────────────────────────────────────
    print(f"\n{'='*130}")
    print("d-PROFILE CHECK (4-point sweep, Exp B structure)")
    print(f"{'='*130}")

    d_sweep = [10.0, 15.0, 20.0, 25.0]
    d_profile = run_d_profile(fit_holds, nadir_info, d_sweep)

    print(f"\n  {'d':>5s} | {'loss':>10s} | {'Aa':>8s} | {'pvo2':>8s} | "
          f"{'k_co2':>8s} | {'tau_f':>8s} | {'tau_reoxy':>10s}")
    print(f"  {'-'*5}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}")
    for d_fixed in sorted(d_profile.keys()):
        r = d_profile[d_fixed]
        s = r["flat"][:N_SHARED_B]
        aa, pvo2, k_co2, r_off, tau_re, d, tau_f = s
        print(f"  {d_fixed:5.1f} | {r['loss']:10.2f} | {aa:8.2f} | {pvo2:8.2f} | "
              f"{k_co2:8.4f} | {tau_f:8.2f} | {tau_re:10.2f}")

    # Check monotonicity
    losses = [d_profile[d]["loss"] for d in sorted(d_profile.keys())]
    is_monotone = all(losses[i] >= losses[i + 1] for i in range(len(losses) - 1))
    print(f"\n  Loss monotonically decreasing with d: {'YES (still degenerate)' if is_monotone else 'NO (nadir penalty breaks degeneracy!)'}")

    # ── Plots ────────────────────────────────────────────────────────────────
    output_dir = Path(__file__).resolve().parent

    plot_per_hold_detail(
        all_results, all_holds, variant_names, nadir_info,
        output_dir / "exp_v6_03_nadir.png",
    )

    plot_d_profile(d_profile, output_dir / "exp_v6_03_d_profile.png")

    # ── Success criteria ─────────────────────────────────────────────────────
    print(f"\n{'='*130}")
    print("SUCCESS CRITERIA CHECK")
    print(f"{'='*130}")

    # 1. Exp A Stage 1: d converges to finite values (not at bounds)
    d_at_bound = sum(1 for v in d_vals if is_at_bound(v, 1, 30))
    print(f"\n  1. [Exp A S1] d at finite values: {len(d_vals) - d_at_bound}/{len(d_vals)} "
          f"{'PASS' if d_at_bound < len(d_vals) // 2 else 'FAIL'}")

    # 2. Exp A: d spread < 10s
    print(f"  2. [Exp A S1] d spread={d_spread:.2f}s (< 10s): "
          f"{'PASS' if d_spread < 10 else 'FAIL'}")

    # 3. Exp B: d not at upper bound
    d_b = flat_b[5]
    d_interior = not is_at_bound(d_b, 1, 30)
    print(f"  3. [Exp B] d={d_b:.2f} (not at bound): "
          f"{'PASS' if d_interior else 'FAIL'}")

    # 4. Exp B: k_co2 not at extremes
    k_co2_b = flat_b[2]
    k_interior = not is_at_bound(k_co2_b, 0.02, 0.25)
    print(f"  4. [Exp B] k_co2={k_co2_b:.4f} (interior): "
          f"{'PASS' if k_interior else 'FAIL'}")

    # 5. Exp B: R2(apnea) avg >= 0.93
    r2_apnea_b = [r["r2_apnea"] for r in eval_b if not r["is_excluded"]]
    avg_r2a = np.mean(r2_apnea_b)
    print(f"  5. [Exp B] R2(apnea) avg={avg_r2a:.4f} (>= 0.93): "
          f"{'PASS' if avg_r2a >= 0.93 else 'FAIL'}")

    # 6. Exp B: Nadir timing error < 5s avg
    nerrs_b = [abs(r["nadir_err"]) for r in eval_b if not r["is_excluded"]]
    avg_nerr = np.mean(nerrs_b)
    print(f"  6. [Exp B] Nadir timing error avg={avg_nerr:.1f}s (< 5s): "
          f"{'PASS' if avg_nerr < 5 else 'FAIL'}")

    # 7. Exp C vs B: gamma < 1.3, params consistent
    gamma_c = flat_c[7]
    print(f"  7. [Exp C] gamma={gamma_c:.4f} (< 1.3): "
          f"{'PASS' if gamma_c < 1.3 else 'FAIL'}")

    # Param consistency: check d, pvo2, k_co2 differ by <20%
    d_c = flat_c[5]
    pvo2_b, pvo2_c = flat_b[1], flat_c[1]
    k_co2_c = flat_c[2]
    d_diff = abs(d_b - d_c) / max(d_b, 0.01) * 100
    pvo2_diff = abs(pvo2_b - pvo2_c) / max(pvo2_b, 0.01) * 100
    k_diff = abs(k_co2_b - k_co2_c) / max(k_co2_b, 0.01) * 100
    consistent = d_diff < 20 and pvo2_diff < 20 and k_diff < 50
    print(f"     B vs C consistency: d diff={d_diff:.1f}%, pvo2 diff={pvo2_diff:.1f}%, "
          f"k_co2 diff={k_diff:.1f}%: {'PASS' if consistent else 'FAIL'}")

    # 8. d-profile: loss no longer monotonically decreasing
    print(f"  8. [d-profile] Non-monotone loss: "
          f"{'PASS' if not is_monotone else 'FAIL'}")


if __name__ == "__main__":
    main()
