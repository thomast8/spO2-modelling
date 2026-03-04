"""
v6 Experiment 1: Global Recovery Fit with Exponential Re-oxygenation.

v5 proved that on apnea-only data with free per-hold initial conditions, sensor
observation models (delay, filter, beat-based) are structurally unidentifiable.
The pao2_0 <-> delay confounding is inherent to any monotone descent with a
free start state.

The fix: include recovery data with a realistic (not step) recovery model, and
make sensor parameters global/shared across all holds. The reversal (breathing
resumes -> SpO2 keeps falling -> then recovers) creates a feature that cannot be
absorbed by rescaling initial conditions.

Variants:
  A: CO2-Bohr baseline (global, apnea only)          — 13 params
  B: CO2-Bohr+Recovery (global, exp recovery)         — 16 params
  C: CO2-Bohr+Recovery+Sensor (global, d + tau_f)     — 18 params  [main]
  D: CO2-Bohr+Recovery+Sensor (gamma=1.0)             — 18 params

Usage:
    cd backend && uv run python -u scripts/exp_v6_01_global_recovery.py
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
P_EQ = 100.0  # atmospheric PAO2 equilibrium
PACO2_NORMAL = 40.0  # normal PaCO2
TAU_CLEAR_FIXED = 30.0  # CO2 clearance time constant (fixed)

EXCLUDED_IDS = {1}  # FL#1 excluded from fit (only 2% SpO2 variation)


# ── CSV recovery data extraction ────────────────────────────────────────────


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

        duration = _parse_time_to_seconds(block[-1]["abs_time"]) - \
                   _parse_time_to_seconds(block[0]["abs_time"])
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


def pao2_apnea(t, pao2_0, pvo2, tau_washout):
    """Exponential O2 decay during apnea."""
    return pvo2 + (pao2_0 - pvo2) * np.exp(-t / max(tau_washout, 0.01))


def pao2_with_exp_recovery(t, pao2_0, pvo2, tau_washout, tau_reoxy, t_end):
    """Piecewise PAO2: exponential decay during apnea, exponential rise during recovery."""
    pao2_end = pvo2 + (pao2_0 - pvo2) * np.exp(-t_end / max(tau_washout, 0.01))
    return np.where(
        t <= t_end,
        pvo2 + (pao2_0 - pvo2) * np.exp(-t / max(tau_washout, 0.01)),
        P_EQ - (P_EQ - pao2_end) * np.exp(-(t - t_end) / max(tau_reoxy, 0.01)),
    )


def p50_apnea(t, paco2_0, k_co2):
    """P50 shift from linear CO2 rise during apnea."""
    paco2 = paco2_0 + k_co2 * t
    return P50_BASE + 0.48 * (paco2 - PACO2_NORMAL)


def p50_with_exp_recovery(t, paco2_0, k_co2, tau_clear, t_end):
    """Piecewise P50: linear CO2 rise during apnea, exponential clearance during recovery."""
    paco2_end = paco2_0 + k_co2 * t_end
    paco2 = np.where(
        t <= t_end,
        paco2_0 + k_co2 * t,
        PACO2_NORMAL + (paco2_end - PACO2_NORMAL) * np.exp(-(t - t_end) / max(tau_clear, 0.01)),
    )
    return P50_BASE + 0.48 * (paco2 - PACO2_NORMAL)


def odc_severinghaus(pao2, p50_eff, gamma):
    """Severinghaus ODC with Bohr shift and gamma steepness."""
    pao2_virtual = pao2 * (P50_BASE / np.maximum(p50_eff, 0.01))
    pao2_adj = P50_BASE * (np.maximum(pao2_virtual, 0.01) / P50_BASE) ** gamma
    x = np.maximum(pao2_adj, 0.01)
    return 100.0 / (1.0 + 23400.0 / (x**3 + 150.0 * x))


# ── Predict functions ────────────────────────────────────────────────────────


def predict_co2bohr_apnea(t, params):
    """CO2-Bohr: 7 params, apnea only (no sensor, no recovery)."""
    pao2_0, pvo2, tau_washout, gamma, paco2_0, k_co2, r_offset = params
    pao2 = pao2_apnea(t, pao2_0, pvo2, tau_washout)
    p50 = p50_apnea(t, paco2_0, k_co2)
    sa = odc_severinghaus(pao2, p50, gamma)
    return np.clip(sa + r_offset, 0.0, 100.0)


def predict_co2bohr_recovery(t, params, t_end):
    """CO2-Bohr+Recovery: 8 params (7 + tau_reoxy), exponential recovery, no sensor."""
    pao2_0, pvo2, tau_washout, gamma, paco2_0, k_co2, r_offset, tau_reoxy = params
    pao2 = pao2_with_exp_recovery(t, pao2_0, pvo2, tau_washout, tau_reoxy, t_end)
    p50 = p50_with_exp_recovery(t, paco2_0, k_co2, TAU_CLEAR_FIXED, t_end)
    sa = odc_severinghaus(pao2, p50, gamma)
    return np.clip(sa + r_offset, 0.0, 100.0)


def predict_recovery_sensor(t, params, t_end):
    """CO2-Bohr+Recovery+Sensor: 10 params (7 + tau_reoxy + d + tau_f).

    Full pipeline: piecewise physiology -> ODC -> delay -> preconditioned IIR filter -> clip.
    """
    pao2_0, pvo2, tau_washout, gamma, paco2_0, k_co2, r_offset, tau_reoxy, d, tau_f = params
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

# Type-specific params: pao2_0, tau_washout, paco2_0
TYPE_SPECIFIC_BOUNDS = {
    "FL": [(100, 250), (50, 250), (20, 50)],
    "FRC": [(80, 140), (20, 100), (25, 50)],
    "RV": [(70, 110), (10, 80), (30, 55)],
}
TYPE_SPECIFIC_NAMES = ["pao2_0", "tau_washout", "paco2_0"]

# Variant A: CO2-Bohr apnea baseline (global, no recovery)
# Shared: pvo2, gamma, k_co2, r_offset
VARIANT_A_SHARED_BOUNDS = [(15, 50), (0.8, 2.5), (0.02, 0.25), (-3, 3)]
VARIANT_A_SHARED_NAMES = ["pvo2", "gamma", "k_co2", "r_offset"]

# Variant B: CO2-Bohr+Recovery (global, exponential recovery, no sensor)
# Shared: pvo2, gamma, k_co2, r_offset, tau_reoxy
VARIANT_B_SHARED_BOUNDS = [(15, 50), (0.8, 2.5), (0.02, 0.25), (-3, 3), (3, 60)]
VARIANT_B_SHARED_NAMES = ["pvo2", "gamma", "k_co2", "r_offset", "tau_reoxy"]

# Variant C: CO2-Bohr+Recovery+Sensor (global, d + tau_f)
# Shared: pvo2, gamma, k_co2, r_offset, tau_reoxy, d, tau_f
VARIANT_C_SHARED_BOUNDS = [(15, 50), (0.8, 2.5), (0.02, 0.25), (-3, 3), (3, 60), (1, 30), (1, 30)]
VARIANT_C_SHARED_NAMES = ["pvo2", "gamma", "k_co2", "r_offset", "tau_reoxy", "d", "tau_f"]

# Variant D: same as C but gamma fixed at 1.0
VARIANT_D_SHARED_BOUNDS = [
    (15, 50), (0.99, 1.01), (0.02, 0.25), (-3, 3), (3, 60), (1, 30), (1, 30),
]
VARIANT_D_SHARED_NAMES = VARIANT_C_SHARED_NAMES


# ── Global fit machinery ────────────────────────────────────────────────────


def build_global_bounds(shared_bounds, type_specific_bounds, hold_types):
    """Build flat bounds vector: shared + type-specific for each type."""
    bounds = list(shared_bounds)
    for ht in hold_types:
        bounds.extend(type_specific_bounds[ht])
    return bounds


def unpack_global(flat, n_shared, n_ts, hold_types, hold_type):
    """Unpack flat global vector into (shared_params, type_specific_params)."""
    shared = flat[:n_shared]
    type_idx = hold_types.index(hold_type)
    offset = n_shared + type_idx * n_ts
    specific = flat[offset : offset + n_ts]
    return shared, specific


def assemble_params_a(shared, specific):
    """Variant A: [pao2_0, pvo2, tau_washout, gamma, paco2_0, k_co2, r_offset]."""
    pvo2, gamma, k_co2, r_offset = shared
    pao2_0, tau_washout, paco2_0 = specific
    return np.array([pao2_0, pvo2, tau_washout, gamma, paco2_0, k_co2, r_offset])


def assemble_params_b(shared, specific):
    """Variant B: [pao2_0, pvo2, tau_washout, gamma, paco2_0, k_co2, r_offset, tau_reoxy]."""
    pvo2, gamma, k_co2, r_offset, tau_reoxy = shared
    pao2_0, tau_washout, paco2_0 = specific
    return np.array([pao2_0, pvo2, tau_washout, gamma, paco2_0, k_co2, r_offset, tau_reoxy])


def assemble_params_cd(shared, specific):
    """Variant C/D: [pao2_0, pvo2, tau_washout, gamma, paco2_0, k_co2, r_offset, tau_reoxy, d, tau_f]."""
    pvo2, gamma, k_co2, r_offset, tau_reoxy, d, tau_f = shared
    pao2_0, tau_washout, paco2_0 = specific
    return np.array([
        pao2_0, pvo2, tau_washout, gamma, paco2_0, k_co2, r_offset, tau_reoxy, d, tau_f,
    ])


def global_fit(
    holds_by_type, hold_types, shared_bounds, shared_names,
    predict_fn, assemble_fn, label, uses_recovery=True,
):
    """Run global fit with shared + type-specific parameter split."""
    n_shared = len(shared_bounds)
    n_ts = len(TYPE_SPECIFIC_NAMES)
    bounds = build_global_bounds(shared_bounds, TYPE_SPECIFIC_BOUNDS, hold_types)
    n_total = len(bounds)

    def objective(flat):
        total_sse = 0.0
        for ht in hold_types:
            shared, specific = unpack_global(flat, n_shared, n_ts, hold_types, ht)
            params = assemble_fn(shared, specific)
            for h in holds_by_type[ht]:
                if uses_recovery:
                    pred = predict_fn(h["t"], params, h["t_end"])
                else:
                    pred = predict_fn(h["t_apnea"], params)
                obs = h["spo2"] if uses_recovery else h["spo2_apnea"]
                w = np.where(obs < 95, 3.0, 1.0)
                total_sse += np.sum(w * (obs - pred) ** 2)
        return total_sse

    n_types = len(hold_types)
    print(f"\n  {label}: {n_total} params "
          f"({n_shared} shared + {n_ts}x{n_types} type-specific)", flush=True)

    result = differential_evolution(
        objective, bounds, maxiter=6000, seed=42, tol=1e-10,
        polish=True, popsize=60, mutation=(0.5, 1.5), recombination=0.9,
    )
    print(f"  Converged: {result.success}, fun={result.fun:.2f}, nfev={result.nfev}", flush=True)
    return result.x, result.success


# ── Evaluation ──────────────────────────────────────────────────────────────


def evaluate_variant(
    flat, holds, hold_types, shared_bounds, shared_names,
    predict_fn, assemble_fn, variant_name, uses_recovery=True,
):
    """Evaluate a fitted variant on all holds, return per-hold metrics."""
    n_shared = len(shared_bounds)
    n_ts = len(TYPE_SPECIFIC_NAMES)
    global_bounds = build_global_bounds(shared_bounds, TYPE_SPECIFIC_BOUNDS, hold_types)

    results = []
    for h in holds:
        shared, specific = unpack_global(flat, n_shared, n_ts, hold_types, h["type"])
        params = assemble_fn(shared, specific)

        # Full trace prediction (uses recovery for recovery variants)
        if uses_recovery:
            pred_full = predict_fn(h["t"], params, h["t_end"])
        else:
            # For apnea-only variant, predict on full trace by just running apnea formula
            pred_full = predict_fn(h["t"], params)

        r2_full = compute_r2(h["spo2"], pred_full)
        rmse_full = compute_rmse(h["spo2"], pred_full)

        # Apnea-only metrics
        if uses_recovery:
            pred_apnea = predict_fn(h["t_apnea"], params, h["t_end"])
        else:
            pred_apnea = predict_fn(h["t_apnea"], params)
        r2_apnea = compute_r2(h["spo2_apnea"], pred_apnea)

        # Recovery-only metrics
        r2_recovery = None
        rmse_recovery = None
        if len(h["t_recovery"]) > 3 and uses_recovery:
            pred_rec = predict_fn(h["t_recovery"], params, h["t_end"])
            r2_recovery = compute_r2(h["spo2_recovery"], pred_rec)
            rmse_recovery = compute_rmse(h["spo2_recovery"], pred_rec)

        # Bounds check on type-specific params
        ts_bounds = TYPE_SPECIFIC_BOUNDS[h["type"]]
        at_bounds = []
        for val, (lo, hi), name in zip(specific, ts_bounds, TYPE_SPECIFIC_NAMES, strict=True):
            if is_at_bound(val, lo, hi):
                at_bounds.append(f"{name}={'lo' if abs(val - lo) < 1e-3 else 'hi'}")

        results.append({
            "variant": variant_name,
            "hold_id": h["id"],
            "hold_type": h["type"],
            "r2_full": r2_full,
            "rmse_full": rmse_full,
            "r2_apnea": r2_apnea,
            "r2_recovery": r2_recovery,
            "rmse_recovery": rmse_recovery,
            "at_bounds_ts": at_bounds,
            "params": params,
            "pred_full": pred_full,
            "shared": shared,
            "specific": specific,
            "is_excluded": h["id"] in EXCLUDED_IDS,
        })
    return results


# ── Profile likelihood for d ─────────────────────────────────────────────────


def profile_likelihood_d(
    holds_by_type, hold_types, base_flat, d_values,
):
    """Fix d at each value, re-optimize other params, return loss curve."""
    n_shared_c = len(VARIANT_C_SHARED_BOUNDS)
    n_ts = len(TYPE_SPECIFIC_NAMES)
    d_idx_in_shared = 5  # d is 6th shared param (0-indexed)

    losses = []
    for d_fixed in d_values:
        # Build bounds with d fixed
        fixed_shared = list(VARIANT_C_SHARED_BOUNDS)
        fixed_shared[d_idx_in_shared] = (d_fixed - 0.01, d_fixed + 0.01)
        bounds = build_global_bounds(fixed_shared, TYPE_SPECIFIC_BOUNDS, hold_types)

        def objective(flat):
            total_sse = 0.0
            for ht in hold_types:
                shared, specific = unpack_global(flat, n_shared_c, n_ts, hold_types, ht)
                params = assemble_params_cd(shared, specific)
                for h in holds_by_type[ht]:
                    pred = predict_recovery_sensor(h["t"], params, h["t_end"])
                    w = np.where(h["spo2"] < 95, 3.0, 1.0)
                    total_sse += np.sum(w * (h["spo2"] - pred) ** 2)
            return total_sse

        result = differential_evolution(
            objective, bounds, maxiter=3000, seed=42, tol=1e-8,
            polish=True, popsize=40, mutation=(0.5, 1.5), recombination=0.9,
        )
        losses.append(result.fun)
        print(f"    d={d_fixed:5.1f} -> loss={result.fun:.2f}", flush=True)

    return np.array(losses)


# ── Output ──────────────────────────────────────────────────────────────────


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
        short = vn[:12]
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


def print_shared_params(variant_configs):
    """Print shared parameter values with bounds for all variants."""
    print(f"\n{'='*120}")
    print("SHARED PARAMETER VALUES")
    print(f"{'='*120}")

    for vname, flat, shared_bounds, shared_names in variant_configs:
        n_shared = len(shared_bounds)
        shared = flat[:n_shared]
        print(f"\n  {vname}:")
        for name, val, (lo, hi) in zip(shared_names, shared, shared_bounds):
            flag = " ** AT BOUND **" if is_at_bound(val, lo, hi) else ""
            print(f"    {name:>12s} = {val:8.4f}  [{lo:>6.2f}, {hi:>6.2f}]{flag}")


def print_type_specific_params(all_eval_results, variant_names):
    """Print type-specific params per hold for each variant."""
    print(f"\n{'='*120}")
    print("TYPE-SPECIFIC PARAMETERS")
    print(f"{'='*120}")

    by_variant = {}
    for r in all_eval_results:
        by_variant.setdefault(r["variant"], []).append(r)

    for vn in variant_names:
        print(f"\n  --- {vn} ---")
        for r in sorted(by_variant.get(vn, []), key=lambda x: x["hold_id"]):
            tag = " (excl)" if r["is_excluded"] else ""
            label = f"{r['hold_type']}#{r['hold_id']}{tag}"
            ts_bounds = TYPE_SPECIFIC_BOUNDS[r["hold_type"]]
            print(f"    {label}:", end="")
            for name, val, (lo, hi) in zip(TYPE_SPECIFIC_NAMES, r["specific"], ts_bounds):
                flag = "*" if is_at_bound(val, lo, hi) else ""
                print(f"  {name}={val:.2f}{flag}", end="")
            if r["at_bounds_ts"]:
                print(f"  [{', '.join(r['at_bounds_ts'])}]", end="")
            print()


def plot_results(all_eval_results, variant_names, output_path):
    """Plot per-hold observed vs predicted for all variants."""
    by_hold = {}
    for r in all_eval_results:
        key = (r["hold_id"], r["hold_type"])
        by_hold.setdefault(key, {})[r["variant"]] = r

    holds_sorted = sorted(by_hold.keys(), key=lambda x: ({"FL": 0, "FRC": 1, "RV": 2}[x[1]], x[0]))
    n_holds = len(holds_sorted)
    fig, axes = plt.subplots(n_holds, 1, figsize=(16, 4.5 * n_holds), squeeze=False)

    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

    for idx, (hid, htype) in enumerate(holds_sorted):
        ax = axes[idx, 0]
        variants = by_hold[(hid, htype)]

        # Get observed data from first variant result
        first_r = next(iter(variants.values()))
        # We need the holds data; reconstruct from eval results
        # Use Variant A if available (it has pred_full for full time range of apnea-only)
        # Use any variant that has the full trace
        ref = variants.get(variant_names[1]) or first_r  # prefer recovery variant

        # Plot observed from a recovery variant if available
        for vn in variant_names[1:]:
            if vn in variants:
                ref = variants[vn]
                break

        ax.plot(ref["pred_full"], alpha=0)  # dummy for x range

        for i, vn in enumerate(variant_names):
            r = variants.get(vn)
            if not r:
                continue
            r2_str = f"R2f={r['r2_full']:.4f}"
            if r["r2_recovery"] is not None:
                r2_str += f", R2r={r['r2_recovery']:.4f}"
            label = f"{vn} ({r2_str})"
            ax.plot(
                np.arange(len(r["pred_full"])), r["pred_full"],
                color=colors[i % len(colors)], linewidth=1.5, alpha=0.8, label=label,
            )

        tag = " [EXCLUDED]" if hid in EXCLUDED_IDS else ""
        ax.set_title(f"{htype}#{hid}{tag}", fontsize=13, fontweight="bold",
                     color="red" if hid in EXCLUDED_IDS else "black")
        ax.set_ylabel("SpO2 (%)")
        ax.set_ylim(30, 105)
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Sample index")
    fig.suptitle("v6 Exp 1: Global Recovery Fit — Exponential Re-oxygenation + Sensor",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {output_path}")


def plot_per_hold_detail(all_eval_results, holds, variant_names, output_path):
    """Detailed per-hold plots with time axis and apnea end marker."""
    by_hold_results = {}
    for r in all_eval_results:
        by_hold_results.setdefault(r["hold_id"], {})[r["variant"]] = r

    holds_dict = {h["id"]: h for h in holds}
    hold_ids = sorted(by_hold_results.keys())

    n_holds = len(hold_ids)
    fig, axes = plt.subplots(n_holds, 1, figsize=(16, 4.5 * n_holds), squeeze=False)

    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

    for idx, hid in enumerate(hold_ids):
        ax = axes[idx, 0]
        h = holds_dict[hid]
        variants = by_hold_results[hid]

        # Plot observed data
        ax.plot(h["t"], h["spo2"], "k.", markersize=2, alpha=0.5, label="Observed")
        ax.axvline(x=h["t_end"], color="red", linestyle="--", alpha=0.5, label="Apnea end")

        # Plot each variant prediction
        for i, vn in enumerate(variant_names):
            r = variants.get(vn)
            if not r:
                continue

            # Get time array matching prediction length
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
            ax.plot(t_plot, r["pred_full"], color=colors[i % len(colors)],
                    linewidth=1.5, alpha=0.8, label=label)

        tag = " [EXCLUDED]" if hid in EXCLUDED_IDS else ""
        ax.set_title(f"{h['type']}#{hid}{tag}", fontsize=13, fontweight="bold",
                     color="red" if hid in EXCLUDED_IDS else "black")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("SpO2 (%)")
        ax.set_ylim(30, 105)
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(True, alpha=0.3)

    fig.suptitle("v6 Exp 1: Per-Hold Fit Detail (time axis)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Detail plot saved to {output_path}")


def plot_profile_likelihood(d_values, losses, output_path):
    """Plot profile likelihood curve for d."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(d_values, losses, "o-", color="#1f77b4", linewidth=2, markersize=6)
    ax.set_xlabel("d (fixed delay, seconds)", fontsize=12)
    ax.set_ylabel("Weighted SSE (optimized over other params)", fontsize=12)
    ax.set_title("Profile Likelihood for Sensor Delay d", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Mark minimum
    best_idx = np.argmin(losses)
    ax.axvline(x=d_values[best_idx], color="red", linestyle="--", alpha=0.5,
               label=f"Best d={d_values[best_idx]:.1f}")
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Profile likelihood plot saved to {output_path}")


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    print("=" * 120)
    print("v6 EXPERIMENT 1: Global Recovery Fit — Exponential Re-oxygenation + Sensor")
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
            rec_range = f", recovery SpO2 {h['spo2_recovery'].min():.0f}-{h['spo2_recovery'][-1]:.0f}%"
        print(
            f"  {h['type']}#{h['id']}{tag}: {n_apnea} apnea + {n_rec} recovery pts "
            f"(t_end={h['t_end']:.0f}s{rec_range})"
        )

    # Partition holds
    fit_holds = [h for h in all_holds if h["id"] not in EXCLUDED_IDS]
    hold_types = sorted(set(h["type"] for h in fit_holds))
    holds_by_type = {}
    for h in fit_holds:
        holds_by_type.setdefault(h["type"], []).append(h)

    print(f"\nFitting on {len(fit_holds)} holds (types: {hold_types}), "
          f"excluding {sum(1 for h in all_holds if h['id'] in EXCLUDED_IDS)} hold(s)")

    # ── Variant A: CO2-Bohr apnea baseline ───────────────────────────────────
    print(f"\n{'='*120}")
    print("VARIANT A: CO2-Bohr (global, apnea only)")
    print(f"{'='*120}")

    flat_a, conv_a = global_fit(
        holds_by_type, hold_types,
        VARIANT_A_SHARED_BOUNDS, VARIANT_A_SHARED_NAMES,
        predict_co2bohr_apnea, assemble_params_a,
        "A: CO2-Bohr (apnea)", uses_recovery=False,
    )

    # ── Variant B: CO2-Bohr+Recovery ─────────────────────────────────────────
    print(f"\n{'='*120}")
    print("VARIANT B: CO2-Bohr+Recovery (global, exponential recovery, no sensor)")
    print(f"{'='*120}")

    flat_b, conv_b = global_fit(
        holds_by_type, hold_types,
        VARIANT_B_SHARED_BOUNDS, VARIANT_B_SHARED_NAMES,
        predict_co2bohr_recovery, assemble_params_b,
        "B: CO2-Bohr+Recovery (exp)",
    )

    # ── Variant C: CO2-Bohr+Recovery+Sensor ──────────────────────────────────
    print(f"\n{'='*120}")
    print("VARIANT C: CO2-Bohr+Recovery+Sensor (global, d + tau_f)")
    print(f"{'='*120}")

    flat_c, conv_c = global_fit(
        holds_by_type, hold_types,
        VARIANT_C_SHARED_BOUNDS, VARIANT_C_SHARED_NAMES,
        predict_recovery_sensor, assemble_params_cd,
        "C: CO2-Bohr+Recovery+Sensor",
    )

    # ── Variant D: CO2-Bohr+Recovery+Sensor (gamma=1.0) ─────────────────────
    print(f"\n{'='*120}")
    print("VARIANT D: CO2-Bohr+Recovery+Sensor (gamma=1.0)")
    print(f"{'='*120}")

    flat_d, conv_d = global_fit(
        holds_by_type, hold_types,
        VARIANT_D_SHARED_BOUNDS, VARIANT_D_SHARED_NAMES,
        predict_recovery_sensor, assemble_params_cd,
        "D: CO2-Bohr+Recovery+Sensor (gamma=1.0)",
    )

    # ── Evaluate all variants on all holds ───────────────────────────────────
    print(f"\n{'='*120}")
    print("EVALUATION")
    print(f"{'='*120}")

    variant_names = ["A:CO2-Bohr", "B:Recovery", "C:Recov+Sensor", "D:Recov+Sens(g=1)"]
    all_eval = []

    eval_a = evaluate_variant(
        flat_a, all_holds, hold_types,
        VARIANT_A_SHARED_BOUNDS, VARIANT_A_SHARED_NAMES,
        predict_co2bohr_apnea, assemble_params_a,
        "A:CO2-Bohr", uses_recovery=False,
    )
    all_eval.extend(eval_a)

    eval_b = evaluate_variant(
        flat_b, all_holds, hold_types,
        VARIANT_B_SHARED_BOUNDS, VARIANT_B_SHARED_NAMES,
        predict_co2bohr_recovery, assemble_params_b,
        "B:Recovery",
    )
    all_eval.extend(eval_b)

    eval_c = evaluate_variant(
        flat_c, all_holds, hold_types,
        VARIANT_C_SHARED_BOUNDS, VARIANT_C_SHARED_NAMES,
        predict_recovery_sensor, assemble_params_cd,
        "C:Recov+Sensor",
    )
    all_eval.extend(eval_c)

    eval_d = evaluate_variant(
        flat_d, all_holds, hold_types,
        VARIANT_D_SHARED_BOUNDS, VARIANT_D_SHARED_NAMES,
        predict_recovery_sensor, assemble_params_cd,
        "D:Recov+Sens(g=1)",
    )
    all_eval.extend(eval_d)

    # ── Print results ────────────────────────────────────────────────────────
    print_comparison_table(all_eval, variant_names)

    print_shared_params([
        ("A:CO2-Bohr", flat_a, VARIANT_A_SHARED_BOUNDS, VARIANT_A_SHARED_NAMES),
        ("B:Recovery", flat_b, VARIANT_B_SHARED_BOUNDS, VARIANT_B_SHARED_NAMES),
        ("C:Recov+Sensor", flat_c, VARIANT_C_SHARED_BOUNDS, VARIANT_C_SHARED_NAMES),
        ("D:Recov+Sens(g=1)", flat_d, VARIANT_D_SHARED_BOUNDS, VARIANT_D_SHARED_NAMES),
    ])

    print_type_specific_params(all_eval, variant_names)

    # ── Identifiability summary ──────────────────────────────────────────────
    print(f"\n{'='*120}")
    print("IDENTIFIABILITY SUMMARY")
    print(f"{'='*120}")

    for vname, flat, shared_bounds, shared_names in [
        ("A:CO2-Bohr", flat_a, VARIANT_A_SHARED_BOUNDS, VARIANT_A_SHARED_NAMES),
        ("B:Recovery", flat_b, VARIANT_B_SHARED_BOUNDS, VARIANT_B_SHARED_NAMES),
        ("C:Recov+Sensor", flat_c, VARIANT_C_SHARED_BOUNDS, VARIANT_C_SHARED_NAMES),
        ("D:Recov+Sens(g=1)", flat_d, VARIANT_D_SHARED_BOUNDS, VARIANT_D_SHARED_NAMES),
    ]:
        n_shared = len(shared_bounds)
        shared = flat[:n_shared]
        n_total = len(flat)
        n_at = 0
        at_list = []

        for name, val, (lo, hi) in zip(shared_names, shared, shared_bounds):
            if is_at_bound(val, lo, hi):
                n_at += 1
                at_list.append(f"{name}={'lo' if abs(val - lo) < 1e-3 else 'hi'}")

        # Check type-specific
        n_ts = len(TYPE_SPECIFIC_NAMES)
        for ht in hold_types:
            _, specific = unpack_global(flat, n_shared, n_ts, hold_types, ht)
            ts_bounds = TYPE_SPECIFIC_BOUNDS[ht]
            for name, val, (lo, hi) in zip(TYPE_SPECIFIC_NAMES, specific, ts_bounds):
                if is_at_bound(val, lo, hi):
                    n_at += 1
                    at_list.append(f"{ht}/{name}={'lo' if abs(val - lo) < 1e-3 else 'hi'}")

        print(f"\n  {vname}: {n_at}/{n_total} at bounds")
        if at_list:
            for ab in at_list:
                print(f"    {ab}")

    # ── Profile likelihood for d ─────────────────────────────────────────────
    print(f"\n{'='*120}")
    print("PROFILE LIKELIHOOD FOR d (Variant C)")
    print(f"{'='*120}")

    d_values = np.array([2, 5, 8, 10, 12, 15, 18, 20, 25, 29])
    print(f"  Testing d at: {d_values}")

    profile_losses = profile_likelihood_d(holds_by_type, hold_types, flat_c, d_values)

    best_d_idx = np.argmin(profile_losses)
    print(f"\n  Best d={d_values[best_d_idx]:.1f} (loss={profile_losses[best_d_idx]:.2f})")
    print(f"  Loss range: {profile_losses.min():.2f} — {profile_losses.max():.2f}")
    print(f"  Ratio max/min: {profile_losses.max()/profile_losses.min():.3f}")

    # ── Plots ────────────────────────────────────────────────────────────────
    output_dir = Path(__file__).resolve().parent

    plot_per_hold_detail(
        all_eval, all_holds, variant_names,
        output_dir / "exp_v6_01_global_recovery.png",
    )

    plot_profile_likelihood(
        d_values, profile_losses,
        output_dir / "exp_v6_01_profile_d.png",
    )

    # ── Success criteria check ───────────────────────────────────────────────
    print(f"\n{'='*120}")
    print("SUCCESS CRITERIA CHECK")
    print(f"{'='*120}")

    shared_c = flat_c[:len(VARIANT_C_SHARED_BOUNDS)]
    d_val = shared_c[5]
    tau_f_val = shared_c[6]
    gamma_c = shared_c[1]
    tau_reoxy_c = shared_c[4]

    # 1. d and tau_f at interior values
    d_interior = not is_at_bound(d_val, 1, 30)
    tf_interior = not is_at_bound(tau_f_val, 1, 30)
    print(f"\n  1. d={d_val:.2f} interior: {'PASS' if d_interior else 'FAIL'}")
    print(f"     tau_f={tau_f_val:.2f} interior: {'PASS' if tf_interior else 'FAIL'}")

    # 2. tau_reoxy plausible (5-30s)
    reoxy_ok = 5 <= tau_reoxy_c <= 30
    print(f"  2. tau_reoxy={tau_reoxy_c:.2f} in [5,30]: {'PASS' if reoxy_ok else f'MARGINAL ({tau_reoxy_c:.1f}s)'}")

    # 3. R2 on recovery segments > 0
    rec_r2s = [r["r2_recovery"] for r in eval_c if r["r2_recovery"] is not None and not r["is_excluded"]]
    rec_avg = np.mean(rec_r2s) if rec_r2s else float("nan")
    rec_pass = all(r > 0 for r in rec_r2s) if rec_r2s else False
    print(f"  3. R2(recovery) avg={rec_avg:.4f}, all>0: {'PASS' if rec_pass else 'FAIL'}")

    # 4. R2 on apnea segments >= 0.95
    apn_r2s = [r["r2_apnea"] for r in eval_c if not r["is_excluded"]]
    apn_avg = np.mean(apn_r2s) if apn_r2s else float("nan")
    apn_pass = apn_avg >= 0.95
    print(f"  4. R2(apnea) avg={apn_avg:.4f} >= 0.95: {'PASS' if apn_pass else 'FAIL'}")

    # 5. gamma closer to 1.0 than v4 (1.76)
    gamma_a = flat_a[1]  # gamma from baseline
    g_improvement = abs(gamma_a - 1.0) - abs(gamma_c - 1.0)
    print(f"  5. gamma: A={gamma_a:.4f}, C={gamma_c:.4f} "
          f"(improvement={g_improvement:.4f}): {'PASS' if g_improvement > 0 else 'FAIL'}")

    # 6. Profile likelihood shows peak
    profile_ratio = profile_losses.max() / profile_losses.min()
    peak_exists = profile_ratio > 1.05
    print(f"  6. Profile likelihood ratio={profile_ratio:.3f}: "
          f"{'PASS (peak exists)' if peak_exists else 'FLAT (no clear peak)'}")


if __name__ == "__main__":
    main()
