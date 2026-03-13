"""
v5 Experiment 3: Recovery Phase — Full Trace Modeling.

The game changer: including 30-60s of recovery data after breathing resumes
should break the delay/pao2_0 symmetry and make both d and tau_f identifiable.

During recovery, PAO2 jumps to ~100 mmHg almost instantly (first breath), but
the sensor reads the delayed+filtered signal, so SpO2 keeps falling for d
seconds, then slowly rises over tau_f seconds. This creates a distinctive
V-shaped nadir that directly constrains the sensor model.

Model (9p): CO2-Bohr (7p) + delay d + filter tau_f.
    Piecewise PAO2: exponential decay during apnea, jump to 100 mmHg at recovery.
    Same delay + filter observation model as Exp 2.

Usage:
    cd backend && uv run python -u scripts/exp_v5_03_recovery.py
"""

import csv
import io
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution
from scipy.signal import lfilter

DB_PATH = Path(__file__).resolve().parents[4] / "data" / "spo2.db"

P50_BASE = 26.6


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
    """Load holds from DB with recovery data appended from raw CSV.

    For each apnea hold, appends the subsequent Rest/Cooldown interval
    (truncated at recovery_max_s or when SpO2 reaches recovery_spo2_ceiling).

    Returns list of dicts with keys:
        id, type, t, spo2, hr, t_end (apnea end time relative to hold start),
        t_apnea, spo2_apnea, t_recovery, spo2_recovery
    """
    conn = sqlite3.connect(DB_PATH)
    holds_db = conn.execute(
        "SELECT id, hold_type FROM holds WHERE hold_type != 'untagged' ORDER BY id"
    ).fetchall()

    # Load apnea-only data from DB (same as before)
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

    # Load raw CSV from DB to get recovery data
    csv_blob = conn.execute("SELECT csv_blob FROM sessions WHERE id = 1").fetchone()[0]
    conn.close()

    csv_text = csv_blob.decode("utf-8-sig")
    if csv_text.startswith("\ufeff"):
        csv_text = csv_text[1:]
    reader = csv.reader(io.StringIO(csv_text))
    rows = list(reader)

    # Find biometrics section
    bio_start = None
    for i, row in enumerate(rows):
        if row and row[0].strip() == "Biometrics":
            bio_start = i + 2  # skip header
            break

    if bio_start is None:
        raise ValueError("No Biometrics section in CSV")

    # Parse all intervals
    intervals = []
    current_type = None
    current_block = []

    for row in rows[bio_start:]:
        if not row or len(row) < 5:
            continue
        abs_time = row[0].strip()
        int_time = row[1].strip()
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
            "abs_time": abs_time,
            "int_time": int_time,
            "type": itype,
            "hr": hr,
            "spo2": spo2,
        })

    if current_block:
        intervals.append((current_type, current_block))

    # Match apnea intervals to DB holds and extract recovery
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
        hold_id = apnea_idx  # 1-indexed

        if hold_id not in hold_data_db:
            continue

        db_hold = hold_data_db[hold_id]
        t_end = float(db_hold["t"][-1])  # apnea end time

        # Get recovery interval (next Rest or Cooldown)
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

        # Combine apnea + recovery
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


def pao2_exponential(t, pao2_0, pvo2, tau):
    return pvo2 + (pao2_0 - pvo2) * np.exp(-t / max(tau, 0.01))


def pao2_with_recovery(t, pao2_0, pvo2, tau, t_end):
    """Piecewise PAO2: exponential decay during apnea, jump to 100 at recovery."""
    return np.where(
        t <= t_end,
        pvo2 + (pao2_0 - pvo2) * np.exp(-t / max(tau, 0.01)),
        100.0,  # atmospheric PAO2 restored instantly on first breath
    )


def p50_linear_co2(t, paco2_0, k_co2):
    paco2 = paco2_0 + k_co2 * t
    return P50_BASE + 0.48 * (paco2 - 40.0)


def p50_with_recovery(t, paco2_0, k_co2, t_end):
    """CO2 rises during apnea, returns to 40 mmHg at recovery."""
    paco2 = np.where(
        t <= t_end,
        paco2_0 + k_co2 * t,
        40.0,  # PaCO2 normalizes on breathing
    )
    return P50_BASE + 0.48 * (paco2 - 40.0)


def odc_severinghaus(pao2, p50_eff, gamma):
    pao2_virtual = pao2 * (P50_BASE / np.maximum(p50_eff, 0.01))
    pao2_adj = P50_BASE * (np.maximum(pao2_virtual, 0.01) / P50_BASE) ** gamma
    x = np.maximum(pao2_adj, 0.01)
    return 100.0 / (1.0 + 23400.0 / (x**3 + 150.0 * x))


# ── Predict functions ────────────────────────────────────────────────────────


def predict_co2bohr_apnea_only(t, hr, params, t_end):
    """CO2-Bohr: 7 params, apnea only (no sensor model)."""
    pao2_0, pvo2, tau_washout, gamma, paco2_0, k_co2, r_offset = params
    pao2 = pao2_exponential(t, pao2_0, pvo2, tau_washout)
    p50 = p50_linear_co2(t, paco2_0, k_co2)
    sa = odc_severinghaus(pao2, p50, gamma)
    return np.clip(sa + r_offset, 0.0, 100.0)


def predict_co2bohr_recovery(t, hr, params, t_end):
    """CO2-Bohr with recovery: 7 params, piecewise PAO2, no sensor model."""
    pao2_0, pvo2, tau_washout, gamma, paco2_0, k_co2, r_offset = params
    pao2 = pao2_with_recovery(t, pao2_0, pvo2, tau_washout, t_end)
    p50 = p50_with_recovery(t, paco2_0, k_co2, t_end)
    sa = odc_severinghaus(pao2, p50, gamma)
    return np.clip(sa + r_offset, 0.0, 100.0)


def predict_delay_filter_recovery(t, hr, params, t_end):
    """CO2-Bohr+D+F with recovery: 9 params, piecewise PAO2 + delay + filter."""
    pao2_0, pvo2, tau_washout, gamma, paco2_0, k_co2, r_offset, d, tau_f = params
    pao2 = pao2_with_recovery(t, pao2_0, pvo2, tau_washout, t_end)
    p50 = p50_with_recovery(t, paco2_0, k_co2, t_end)
    sa = odc_severinghaus(pao2, p50, gamma)
    # Delay
    sa_delayed = np.interp(t - d, t, sa, left=sa[0])
    # IIR filter
    dt = 1.0
    alpha = dt / (max(tau_f, 0.01) + dt)
    s_meas = lfilter([alpha], [1.0, -(1.0 - alpha)], sa_delayed)
    return np.clip(s_meas + r_offset, 0.0, 100.0)


def predict_delay_only_recovery(t, hr, params, t_end):
    """CO2-Bohr+Delay with recovery: 8 params, piecewise PAO2 + delay only."""
    pao2_0, pvo2, tau_washout, gamma, paco2_0, k_co2, r_offset, d = params
    pao2 = pao2_with_recovery(t, pao2_0, pvo2, tau_washout, t_end)
    p50 = p50_with_recovery(t, paco2_0, k_co2, t_end)
    sa = odc_severinghaus(pao2, p50, gamma)
    sa_delayed = np.interp(t - d, t, sa, left=sa[0])
    return np.clip(sa_delayed + r_offset, 0.0, 100.0)


def predict_filter_only_recovery(t, hr, params, t_end):
    """CO2-Bohr+Filter with recovery: 8 params, piecewise PAO2 + filter only."""
    pao2_0, pvo2, tau_washout, gamma, paco2_0, k_co2, r_offset, tau_f = params
    pao2 = pao2_with_recovery(t, pao2_0, pvo2, tau_washout, t_end)
    p50 = p50_with_recovery(t, paco2_0, k_co2, t_end)
    sa = odc_severinghaus(pao2, p50, gamma)
    dt = 1.0
    alpha = dt / (max(tau_f, 0.01) + dt)
    s_meas = lfilter([alpha], [1.0, -(1.0 - alpha)], sa)
    return np.clip(s_meas + r_offset, 0.0, 100.0)


def predict_richards(t, hr, params, t_end):
    s_max, s_min, t50, k, nu = params
    z = np.clip((t - t50) / max(k, 0.01), -500, 500)
    base = 1.0 + nu * np.exp(z)
    return np.clip(
        s_min + (s_max - s_min) / np.power(np.maximum(base, 1e-10), 1.0 / nu), 0.0, 100.0
    )


# ── Loss / metrics ──────────────────────────────────────────────────────────


def loss_weighted_sse(obs, pred, alpha=3.0):
    weights = np.where(obs < 95, alpha, 1.0)
    return np.sum(weights * (obs - pred) ** 2)


def compute_r2(obs, pred):
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def compute_rmse(obs, pred):
    return float(np.sqrt(np.mean((obs - pred) ** 2)))


def check_bounds(param_values, bounds, param_names, tol=1e-3):
    at_bounds = []
    for val, (lo, hi), name in zip(param_values, bounds, param_names, strict=True):
        if abs(val - lo) < tol or abs(val - hi) < tol:
            at_bounds.append(f"{name}={'lo' if abs(val - lo) < tol else 'hi'}")
    return at_bounds


# ── Bounds ───────────────────────────────────────────────────────────────────

CO2BOHR_BOUNDS = {
    "FL": [(100, 250), (20, 50), (50, 250), (0.8, 2.0), (25, 45), (0.02, 0.15), (-3, 3)],
    "FRC": [(80, 120), (20, 50), (20, 100), (0.8, 2.0), (30, 45), (0.02, 0.15), (-3, 3)],
    "RV": [(70, 110), (20, 50), (10, 80), (0.8, 2.0), (35, 50), (0.02, 0.15), (-3, 3)],
}

CO2BOHR_DELAY_BOUNDS = {
    "FL": [
        (100, 250), (20, 50), (50, 250), (0.8, 2.0), (25, 45), (0.02, 0.15), (-3, 3), (3, 30),
    ],
    "FRC": [
        (80, 120), (20, 50), (20, 100), (0.8, 2.0), (30, 45), (0.02, 0.15), (-3, 3), (3, 30),
    ],
    "RV": [
        (70, 110), (20, 50), (10, 80), (0.8, 2.0), (35, 50), (0.02, 0.15), (-3, 3), (3, 30),
    ],
}

CO2BOHR_FILTER_BOUNDS = {
    "FL": [
        (100, 250), (20, 50), (50, 250), (0.8, 2.0), (25, 45), (0.02, 0.15), (-3, 3), (1, 20),
    ],
    "FRC": [
        (80, 120), (20, 50), (20, 100), (0.8, 2.0), (30, 45), (0.02, 0.15), (-3, 3), (1, 20),
    ],
    "RV": [
        (70, 110), (20, 50), (10, 80), (0.8, 2.0), (35, 50), (0.02, 0.15), (-3, 3), (1, 20),
    ],
}

CO2BOHR_DF_BOUNDS = {
    "FL": [
        (100, 250), (20, 50), (50, 250), (0.8, 2.0), (25, 45), (0.02, 0.15), (-3, 3),
        (3, 30), (1, 20),
    ],
    "FRC": [
        (80, 120), (20, 50), (20, 100), (0.8, 2.0), (30, 45), (0.02, 0.15), (-3, 3),
        (3, 30), (1, 20),
    ],
    "RV": [
        (70, 110), (20, 50), (10, 80), (0.8, 2.0), (35, 50), (0.02, 0.15), (-3, 3),
        (3, 30), (1, 20),
    ],
}

RICHARDS_BOUNDS = {
    "FL": [(96, 101), (0, 96), (50, 500), (5, 80), (0.1, 10.0)],
    "FRC": [(96, 101), (0, 96), (20, 300), (3, 60), (0.1, 10.0)],
    "RV": [(96, 101), (0, 96), (10, 250), (3, 60), (0.1, 10.0)],
}


# ── Model variant definitions ───────────────────────────────────────────────


@dataclass
class ModelVariant:
    name: str
    param_names: list[str]
    bounds_by_type: dict[str, list[tuple[float, float]]]
    predict_fn: callable  # (t, hr, params, t_end) -> pred
    loss_fn: callable
    uses_recovery: bool  # Whether to fit on full trace or apnea only


VARIANTS = [
    ModelVariant(
        name="CO2-Bohr (apnea)",
        param_names=["pao2_0", "pvo2", "tau_washout", "gamma", "paco2_0", "k_co2", "r_offset"],
        bounds_by_type=CO2BOHR_BOUNDS,
        predict_fn=predict_co2bohr_apnea_only,
        loss_fn=lambda obs, pred: loss_weighted_sse(obs, pred, alpha=3.0),
        uses_recovery=False,
    ),
    ModelVariant(
        name="CO2-Bohr (full)",
        param_names=["pao2_0", "pvo2", "tau_washout", "gamma", "paco2_0", "k_co2", "r_offset"],
        bounds_by_type=CO2BOHR_BOUNDS,
        predict_fn=predict_co2bohr_recovery,
        loss_fn=lambda obs, pred: loss_weighted_sse(obs, pred, alpha=3.0),
        uses_recovery=True,
    ),
    ModelVariant(
        name="Delay (full)",
        param_names=[
            "pao2_0", "pvo2", "tau_washout", "gamma", "paco2_0", "k_co2", "r_offset", "d",
        ],
        bounds_by_type=CO2BOHR_DELAY_BOUNDS,
        predict_fn=predict_delay_only_recovery,
        loss_fn=lambda obs, pred: loss_weighted_sse(obs, pred, alpha=3.0),
        uses_recovery=True,
    ),
    ModelVariant(
        name="Filter (full)",
        param_names=[
            "pao2_0", "pvo2", "tau_washout", "gamma", "paco2_0", "k_co2", "r_offset", "tau_f",
        ],
        bounds_by_type=CO2BOHR_FILTER_BOUNDS,
        predict_fn=predict_filter_only_recovery,
        loss_fn=lambda obs, pred: loss_weighted_sse(obs, pred, alpha=3.0),
        uses_recovery=True,
    ),
    ModelVariant(
        name="D+F (full)",
        param_names=[
            "pao2_0", "pvo2", "tau_washout", "gamma", "paco2_0", "k_co2", "r_offset", "d", "tau_f",
        ],
        bounds_by_type=CO2BOHR_DF_BOUNDS,
        predict_fn=predict_delay_filter_recovery,
        loss_fn=lambda obs, pred: loss_weighted_sse(obs, pred, alpha=3.0),
        uses_recovery=True,
    ),
]


# ── Fitting ──────────────────────────────────────────────────────────────────


def fit_variant(variant: ModelVariant, hold: dict) -> dict:
    if variant.uses_recovery:
        t = hold["t"]
        spo2 = hold["spo2"]
        hr = hold["hr"]
    else:
        t = hold["t_apnea"]
        spo2 = hold["spo2_apnea"]
        hr = hold["hr"][:len(hold["t_apnea"])]

    t_end = hold["t_end"]
    hold_type = hold["type"]
    bounds = variant.bounds_by_type[hold_type]

    def objective(arr):
        pred = variant.predict_fn(t, hr, arr, t_end)
        return variant.loss_fn(spo2, pred)

    result = differential_evolution(
        objective,
        bounds,
        maxiter=3000,
        seed=42,
        tol=1e-10,
        polish=True,
        popsize=40,
        mutation=(0.5, 1.5),
        recombination=0.9,
    )

    # Compute metrics on full trace for comparable R²
    t_full = hold["t"]
    spo2_full = hold["spo2"]
    pred_full = variant.predict_fn(t_full, hold["hr"], result.x, t_end)
    r2_full = compute_r2(spo2_full, pred_full)
    rmse_full = compute_rmse(spo2_full, pred_full)

    # Apnea-only metrics
    t_apnea = hold["t_apnea"]
    spo2_apnea = hold["spo2_apnea"]
    pred_apnea = variant.predict_fn(t_apnea, hold["hr"][:len(t_apnea)], result.x, t_end)
    r2_apnea = compute_r2(spo2_apnea, pred_apnea)

    # Recovery-only metrics
    t_rec = hold["t_recovery"]
    spo2_rec = hold["spo2_recovery"]
    r2_rec = None
    rmse_rec = None
    if len(t_rec) > 3:
        pred_rec = variant.predict_fn(t_rec, hold["hr"][-len(t_rec):], result.x, t_end)
        r2_rec = compute_r2(spo2_rec, pred_rec)
        rmse_rec = compute_rmse(spo2_rec, pred_rec)

    at_bounds = check_bounds(result.x, bounds, variant.param_names)

    return {
        "variant": variant.name,
        "hold_id": hold["id"],
        "hold_type": hold_type,
        "r2": r2_full,
        "rmse": rmse_full,
        "r2_apnea": r2_apnea,
        "r2_recovery": r2_rec,
        "rmse_recovery": rmse_rec,
        "n_params": len(bounds),
        "at_bounds": at_bounds,
        "n_at_bounds": len(at_bounds),
        "params": dict(zip(variant.param_names, result.x, strict=True)),
        "converged": result.success,
        "pred_full": pred_full,
        "t_full": t_full,
        "spo2_full": spo2_full,
        "t_end": t_end,
    }


# ── Output ───────────────────────────────────────────────────────────────────


def print_results(all_results: list[dict]):
    by_hold = {}
    for r in all_results:
        key = f"{r['hold_type']} #{r['hold_id']}"
        by_hold.setdefault(key, []).append(r)

    variant_names = list(dict.fromkeys(r["variant"] for r in all_results))

    for hold_key, results in by_hold.items():
        print(f"\n{'='*110}")
        print(f"  {hold_key}")
        print(f"{'='*110}")

        header = f"  {'Metric':<22s}" + "".join(f" {n:>15s}" for n in variant_names)
        print(header)
        print(f"  {'-'*22}" + "".join(f" {'-'*15}" for _ in variant_names))

        lookup = {r["variant"]: r for r in results}

        for metric, fmt in [
            ("R² (full trace)", lambda r: f"{r['r2']:>15.6f}"),
            ("R² (apnea only)", lambda r: f"{r['r2_apnea']:>15.6f}"),
            ("R² (recovery)", lambda r: f"{r['r2_recovery']:>15.6f}" if r["r2_recovery"] is not None else f"{'N/A':>15s}"),
            ("RMSE (full)", lambda r: f"{r['rmse']:>15.4f}"),
            ("RMSE (recovery)", lambda r: f"{r['rmse_recovery']:>15.4f}" if r["rmse_recovery"] is not None else f"{'N/A':>15s}"),
            ("# params", lambda r: f"{r['n_params']:>15d}"),
            ("# at bounds", lambda r: f"{r['n_at_bounds']:>15d}"),
        ]:
            row = f"  {metric:<22s}"
            for v in variant_names:
                r = lookup.get(v)
                row += fmt(r) if r else f" {'N/A':>15s}"
            print(row)

        for v in variant_names:
            r = lookup.get(v)
            if r and r["at_bounds"]:
                print(f"    {v}: {r['at_bounds']}")

        variant_lookup = {v.name: v for v in VARIANTS}
        print(f"\n  Fitted parameters:")
        for v in variant_names:
            r = lookup.get(v)
            if not r:
                continue
            print(f"    {v}:")
            vdef = variant_lookup[v]
            bounds = dict(
                zip(vdef.param_names, vdef.bounds_by_type[r["hold_type"]], strict=True)
            )
            for pname, pval in r["params"].items():
                lo, hi = bounds[pname]
                marker = " <<<" if any(pname in ab for ab in r["at_bounds"]) else ""
                print(f"      {pname:>12s} = {pval:10.4f}  [{lo:>8.1f}, {hi:>8.1f}]{marker}")

    # Summary table
    print(f"\n{'='*110}")
    print("  KEY SENSOR PARAMS: d, tau_f, gamma across holds (recovery models)")
    print(f"{'='*110}")
    print(f"  {'Hold':<16s} {'g(base)':>8s} {'g(D+F)':>8s} {'d':>6s} {'tau_f':>6s} "
          f"{'R²full':>8s} {'R²apn':>8s} {'R²rec':>8s}")
    print(f"  {'-'*16} {'-'*8} {'-'*8} {'-'*6} {'-'*6} {'-'*8} {'-'*8} {'-'*8}")

    for hold_key, results in by_hold.items():
        lookup = {r["variant"]: r for r in results}
        g_base = lookup.get("CO2-Bohr (apnea)", {}).get("params", {}).get("gamma", float("nan"))
        r_df = lookup.get("D+F (full)", {})
        g_df = r_df.get("params", {}).get("gamma", float("nan"))
        d_df = r_df.get("params", {}).get("d", float("nan"))
        tf_df = r_df.get("params", {}).get("tau_f", float("nan"))
        r2_full = r_df.get("r2", float("nan"))
        r2_apn = r_df.get("r2_apnea", float("nan"))
        r2_rec = r_df.get("r2_recovery")
        r2_rec_s = f"{r2_rec:>8.4f}" if r2_rec is not None else f"{'N/A':>8s}"
        print(f"  {hold_key:<16s} {g_base:>8.4f} {g_df:>8.4f} {d_df:>6.1f} {tf_df:>6.1f} "
              f"{r2_full:>8.4f} {r2_apn:>8.4f} {r2_rec_s}")


def plot_results(all_results: list[dict], output_path: Path):
    by_hold = {}
    for r in all_results:
        key = f"{r['hold_type']} #{r['hold_id']}"
        by_hold.setdefault(key, []).append(r)

    n_holds = len(by_hold)
    fig, axes = plt.subplots(n_holds, 1, figsize=(16, 5 * n_holds), squeeze=False)

    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#d62728"]

    for idx, (hold_key, results) in enumerate(by_hold.items()):
        ax = axes[idx, 0]
        r0 = results[0]
        t_end = r0["t_end"]

        ax.plot(r0["t_full"], r0["spo2_full"], "k.", markersize=2, alpha=0.5, label="Observed")
        ax.axvline(x=t_end, color="red", linestyle="--", alpha=0.5, label="Apnea end")

        for i, r in enumerate(results):
            r2_str = f"R²={r['r2']:.4f}"
            if r["r2_recovery"] is not None:
                r2_str += f", rec={r['r2_recovery']:.4f}"
            label = f"{r['variant']} ({r2_str})"
            ax.plot(
                r["t_full"], r["pred_full"], color=colors[i % len(colors)],
                linewidth=1.5, alpha=0.8, label=label,
            )

        ax.set_title(hold_key, fontsize=14, fontweight="bold")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("SpO2 (%)")
        ax.set_ylim(30, 105)
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "v5 Exp 3: Recovery Phase — Full Trace Modeling",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    print("=" * 110)
    print("v5 EXPERIMENT 3: Recovery Phase — Full Trace Modeling")
    print("=" * 110)

    print("\nLoading holds with recovery data...")
    holds = load_holds_with_recovery()

    for h in holds:
        n_apnea = len(h["t_apnea"])
        n_rec = len(h["t_recovery"])
        rec_range = ""
        if n_rec > 0:
            rec_range = f", recovery SpO2 {h['spo2_recovery'].min():.0f}-{h['spo2_recovery'][-1]:.0f}%"
        print(
            f"  {h['type']} #{h['id']}: {n_apnea} apnea + {n_rec} recovery pts "
            f"(t_end={h['t_end']:.0f}s{rec_range})"
        )

    all_results = []

    for variant in VARIANTS:
        for hold in holds:
            print(f"\n{'─'*60}", flush=True)
            print(
                f"Fitting {variant.name} to {hold['type']} #{hold['id']} "
                f"({len(variant.param_names)} params, "
                f"{'full trace' if variant.uses_recovery else 'apnea only'})",
                flush=True,
            )
            print(f"{'─'*60}", flush=True)

            result = fit_variant(variant, hold)
            all_results.append(result)

            status = "OK" if result["converged"] else "WARN"
            rec_str = ""
            if result["r2_recovery"] is not None:
                rec_str = f", R²_rec={result['r2_recovery']:.4f}"
            print(
                f"  [{status}] R²={result['r2']:.6f}, RMSE={result['rmse']:.4f}, "
                f"at_bounds={result['n_at_bounds']}/{result['n_params']}"
                f"{rec_str}",
                flush=True,
            )

    print_results(all_results)

    output_dir = Path(__file__).resolve().parent
    plot_results(all_results, output_dir / "exp_v5_03_recovery.png")


if __name__ == "__main__":
    main()
