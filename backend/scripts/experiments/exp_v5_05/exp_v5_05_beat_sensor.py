"""
v5 Experiment 5: HR-Coupled Beat-Based Sensor.

Tests whether expressing delay and filter in heartbeats (rather than seconds)
improves fit, especially on holds with strong bradycardia.

During apnea, HR drops from ~60-100 to ~45-55 bpm. If transit time is B_delay
heartbeats, the delay in seconds stretches as HR drops:
    d(t) = B_delay * 60 / HR(t)

New params: B_delay (beats, 5-25) and B_avg (beats, 3-15) replace constant d
and tau_f. Per-sample IIR filter with time-varying alpha.

Model (9p): CO2-Bohr (7p) + B_delay + B_avg.

Baselines: CO2-Bohr (7p), CO2-Bohr+Delay (8p constant d), Richards (5p).

Usage:
    cd backend && uv run python -u scripts/exp_v5_05_beat_sensor.py
"""

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution

DB_PATH = Path(__file__).resolve().parents[4] / "data" / "spo2.db"

P50_BASE = 26.6


# ── Data loading ─────────────────────────────────────────────────────────────


def load_holds_by_type() -> dict[str, list[dict]]:
    conn = sqlite3.connect(DB_PATH)
    holds = conn.execute(
        "SELECT id, hold_type FROM holds WHERE hold_type != 'untagged' ORDER BY id"
    ).fetchall()

    result: dict[str, list[dict]] = {}
    for hold_id, hold_type in holds:
        rows = conn.execute(
            "SELECT elapsed_s, spo2, hr FROM hold_data WHERE hold_id = ? ORDER BY elapsed_s",
            (hold_id,),
        ).fetchall()
        if not rows:
            continue
        t = np.array([r[0] for r in rows], dtype=float)
        spo2 = np.array([r[1] for r in rows], dtype=float)
        hr = np.array([r[2] for r in rows], dtype=float)
        entry = {"id": hold_id, "t": t, "spo2": spo2, "hr": hr, "type": hold_type}
        result.setdefault(hold_type, []).append(entry)

    conn.close()
    return result


# ── Physiology functions ─────────────────────────────────────────────────────


def pao2_exponential(t, pao2_0, pvo2, tau):
    return pvo2 + (pao2_0 - pvo2) * np.exp(-t / max(tau, 0.01))


def p50_linear_co2(t, paco2_0, k_co2):
    paco2 = paco2_0 + k_co2 * t
    return P50_BASE + 0.48 * (paco2 - 40.0)


def odc_severinghaus(pao2, p50_eff, gamma):
    pao2_virtual = pao2 * (P50_BASE / np.maximum(p50_eff, 0.01))
    pao2_adj = P50_BASE * (np.maximum(pao2_virtual, 0.01) / P50_BASE) ** gamma
    x = np.maximum(pao2_adj, 0.01)
    return 100.0 / (1.0 + 23400.0 / (x**3 + 150.0 * x))


# ── Predict functions ────────────────────────────────────────────────────────


def predict_co2bohr(t, hr, params):
    """CO2-Bohr: 7 params."""
    pao2_0, pvo2, tau_washout, gamma, paco2_0, k_co2, r_offset = params
    pao2 = pao2_exponential(t, pao2_0, pvo2, tau_washout)
    p50 = p50_linear_co2(t, paco2_0, k_co2)
    sa = odc_severinghaus(pao2, p50, gamma)
    return np.clip(sa + r_offset, 0.0, 100.0)


def predict_co2bohr_delay(t, hr, params):
    """CO2-Bohr+Delay: 8 params, constant delay."""
    pao2_0, pvo2, tau_washout, gamma, paco2_0, k_co2, r_offset, d = params
    pao2 = pao2_exponential(t, pao2_0, pvo2, tau_washout)
    p50 = p50_linear_co2(t, paco2_0, k_co2)
    sa = odc_severinghaus(pao2, p50, gamma)
    sa_delayed = np.interp(t - d, t, sa, left=sa[0])
    return np.clip(sa_delayed + r_offset, 0.0, 100.0)


def predict_co2bohr_beat_sensor(t, hr, params):
    """CO2-Bohr+Beat sensor: 9 params, HR-dependent delay + filter.

    B_delay heartbeats of circulation delay: d(t) = B_delay * 60 / HR(t)
    B_avg heartbeats of sensor averaging: tau_f(t) = B_avg * 60 / HR(t)
    Per-sample IIR filter with time-varying alpha.
    """
    pao2_0, pvo2, tau_washout, gamma, paco2_0, k_co2, r_offset, b_delay, b_avg = params
    pao2 = pao2_exponential(t, pao2_0, pvo2, tau_washout)
    p50 = p50_linear_co2(t, paco2_0, k_co2)
    sa = odc_severinghaus(pao2, p50, gamma)

    # Time-varying delay: d(t) = B_delay * 60 / HR(t) seconds
    hr_safe = np.maximum(hr, 30.0)  # floor HR at 30 bpm for safety
    d_t = b_delay * 60.0 / hr_safe

    # Apply time-varying delay (per-sample interpolation)
    sa_delayed = np.empty_like(sa)
    for i in range(len(t)):
        t_lookup = t[i] - d_t[i]
        sa_delayed[i] = np.interp(t_lookup, t, sa, left=sa[0])

    # Time-varying IIR filter: tau_f(t) = B_avg * 60 / HR(t)
    dt = 1.0
    s_meas = np.empty_like(sa_delayed)
    s_meas[0] = sa_delayed[0]
    for i in range(1, len(sa_delayed)):
        tau_f_i = b_avg * 60.0 / hr_safe[i]
        alpha_i = dt / (tau_f_i + dt)
        s_meas[i] = (1.0 - alpha_i) * s_meas[i - 1] + alpha_i * sa_delayed[i]

    return np.clip(s_meas + r_offset, 0.0, 100.0)


def predict_co2bohr_const_delay_filter(t, hr, params):
    """CO2-Bohr+D+F: 9 params, constant delay + constant filter (for comparison)."""
    from scipy.signal import lfilter

    pao2_0, pvo2, tau_washout, gamma, paco2_0, k_co2, r_offset, d, tau_f = params
    pao2 = pao2_exponential(t, pao2_0, pvo2, tau_washout)
    p50 = p50_linear_co2(t, paco2_0, k_co2)
    sa = odc_severinghaus(pao2, p50, gamma)
    sa_delayed = np.interp(t - d, t, sa, left=sa[0])
    dt = 1.0
    alpha = dt / (max(tau_f, 0.01) + dt)
    s_meas = lfilter([alpha], [1.0, -(1.0 - alpha)], sa_delayed)
    return np.clip(s_meas + r_offset, 0.0, 100.0)


def predict_richards(t, hr, params):
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


def compute_drop_metrics(obs, pred, threshold=95.0):
    mask = obs < threshold
    if np.sum(mask) < 3:
        return None, None
    return compute_r2(obs[mask], pred[mask]), compute_rmse(obs[mask], pred[mask])


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
        (100, 250), (20, 50), (50, 250), (0.8, 1.5), (25, 45), (0.02, 0.15), (-3, 3), (3, 30),
    ],
    "FRC": [
        (80, 120), (20, 50), (20, 100), (0.8, 1.5), (30, 45), (0.02, 0.15), (-3, 3), (3, 30),
    ],
    "RV": [
        (70, 110), (20, 50), (10, 80), (0.8, 1.5), (35, 50), (0.02, 0.15), (-3, 3), (3, 30),
    ],
}

# Beat sensor: B_delay (5-25 beats), B_avg (3-15 beats)
BEAT_SENSOR_BOUNDS = {
    "FL": [
        (100, 250), (20, 50), (50, 250), (0.8, 2.0), (25, 45), (0.02, 0.15), (-3, 3),
        (5, 25), (3, 15),
    ],
    "FRC": [
        (80, 120), (20, 50), (20, 100), (0.8, 2.0), (30, 45), (0.02, 0.15), (-3, 3),
        (5, 25), (3, 15),
    ],
    "RV": [
        (70, 110), (20, 50), (10, 80), (0.8, 2.0), (35, 50), (0.02, 0.15), (-3, 3),
        (5, 25), (3, 15),
    ],
}

# Constant D+F for comparison
CONST_DF_BOUNDS = {
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


# ── Model variants ──────────────────────────────────────────────────────────


@dataclass
class ModelVariant:
    name: str
    param_names: list[str]
    bounds_by_type: dict[str, list[tuple[float, float]]]
    predict_fn: callable
    loss_fn: callable


VARIANTS = [
    ModelVariant(
        name="CO2-Bohr",
        param_names=["pao2_0", "pvo2", "tau_washout", "gamma", "paco2_0", "k_co2", "r_offset"],
        bounds_by_type=CO2BOHR_BOUNDS,
        predict_fn=predict_co2bohr,
        loss_fn=lambda obs, pred: loss_weighted_sse(obs, pred, alpha=3.0),
    ),
    ModelVariant(
        name="CO2-Bohr+Delay",
        param_names=[
            "pao2_0", "pvo2", "tau_washout", "gamma", "paco2_0", "k_co2", "r_offset", "d",
        ],
        bounds_by_type=CO2BOHR_DELAY_BOUNDS,
        predict_fn=predict_co2bohr_delay,
        loss_fn=lambda obs, pred: loss_weighted_sse(obs, pred, alpha=3.0),
    ),
    ModelVariant(
        name="Const-D+F",
        param_names=[
            "pao2_0", "pvo2", "tau_washout", "gamma", "paco2_0", "k_co2", "r_offset", "d", "tau_f",
        ],
        bounds_by_type=CONST_DF_BOUNDS,
        predict_fn=predict_co2bohr_const_delay_filter,
        loss_fn=lambda obs, pred: loss_weighted_sse(obs, pred, alpha=3.0),
    ),
    ModelVariant(
        name="Beat-Sensor",
        param_names=[
            "pao2_0", "pvo2", "tau_washout", "gamma", "paco2_0", "k_co2", "r_offset",
            "b_delay", "b_avg",
        ],
        bounds_by_type=BEAT_SENSOR_BOUNDS,
        predict_fn=predict_co2bohr_beat_sensor,
        loss_fn=lambda obs, pred: loss_weighted_sse(obs, pred, alpha=3.0),
    ),
    ModelVariant(
        name="Richards",
        param_names=["s_max", "s_min", "t50", "k", "nu"],
        bounds_by_type=RICHARDS_BOUNDS,
        predict_fn=predict_richards,
        loss_fn=lambda obs, pred: np.sum((obs - pred) ** 2),
    ),
]


# ── Fitting ──────────────────────────────────────────────────────────────────


def fit_variant(variant: ModelVariant, hold: dict) -> dict:
    t = hold["t"]
    spo2 = hold["spo2"]
    hr = hold["hr"]
    hold_type = hold["type"]
    bounds = variant.bounds_by_type[hold_type]

    def objective(arr):
        pred = variant.predict_fn(t, hr, arr)
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

    pred = variant.predict_fn(t, hr, result.x)
    r2 = compute_r2(spo2, pred)
    rmse = compute_rmse(spo2, pred)
    r2_drop, rmse_drop = compute_drop_metrics(spo2, pred)
    at_bounds = check_bounds(result.x, bounds, variant.param_names)

    return {
        "variant": variant.name,
        "hold_id": hold["id"],
        "hold_type": hold_type,
        "r2": r2,
        "rmse": rmse,
        "r2_drop": r2_drop,
        "rmse_drop": rmse_drop,
        "n_params": len(bounds),
        "at_bounds": at_bounds,
        "n_at_bounds": len(at_bounds),
        "params": dict(zip(variant.param_names, result.x, strict=True)),
        "converged": result.success,
        "pred": pred,
        "t": t,
        "spo2": spo2,
        "hr": hold["hr"],
    }


# ── Output ───────────────────────────────────────────────────────────────────


def print_results(all_results: list[dict]):
    by_hold = {}
    for r in all_results:
        key = f"{r['hold_type']} #{r['hold_id']}"
        by_hold.setdefault(key, []).append(r)

    variant_names = list(dict.fromkeys(r["variant"] for r in all_results))

    for hold_key, results in by_hold.items():
        print(f"\n{'='*100}")
        print(f"  {hold_key}")
        print(f"{'='*100}")

        # HR stats for this hold
        r0 = results[0]
        hr = r0["hr"]
        print(f"  HR range: {hr.min():.0f}-{hr.max():.0f} bpm, mean={hr.mean():.0f}")

        header = f"  {'Metric':<22s}" + "".join(f" {n:>14s}" for n in variant_names)
        print(header)
        print(f"  {'-'*22}" + "".join(f" {'-'*14}" for _ in variant_names))

        lookup = {r["variant"]: r for r in results}

        for metric, fmt in [
            ("R²", lambda r: f"{r['r2']:>14.6f}"),
            ("R² (drop <95%)", lambda r: f"{r['r2_drop']:>14.6f}" if r["r2_drop"] is not None else f"{'N/A':>14s}"),
            ("RMSE", lambda r: f"{r['rmse']:>14.4f}"),
            ("# params", lambda r: f"{r['n_params']:>14d}"),
            ("# at bounds", lambda r: f"{r['n_at_bounds']:>14d}"),
        ]:
            row = f"  {metric:<22s}"
            for v in variant_names:
                r = lookup.get(v)
                row += fmt(r) if r else f" {'N/A':>14s}"
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
                # For beat params, show equivalent delay in seconds
                extra = ""
                if pname == "b_delay":
                    avg_hr = r["hr"].mean()
                    extra = f"  (≈{pval * 60 / avg_hr:.1f}s at mean HR)"
                elif pname == "b_avg":
                    avg_hr = r["hr"].mean()
                    extra = f"  (≈{pval * 60 / avg_hr:.1f}s at mean HR)"
                print(f"      {pname:>12s} = {pval:10.4f}  [{lo:>8.1f}, {hi:>8.1f}]{marker}{extra}")

    # Summary
    print(f"\n{'='*100}")
    print("  BEAT SENSOR vs CONSTANT SENSOR SUMMARY")
    print(f"{'='*100}")
    print(f"  {'Hold':<16s} {'HR range':>10s} {'B_delay':>8s} {'B_avg':>8s} "
          f"{'d_equiv':>8s} {'tau_equiv':>9s} {'R²(beat)':>9s} {'R²(base)':>9s} {'R²(const)':>10s}")
    print(f"  {'-'*16} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*9} {'-'*9} {'-'*9} {'-'*10}")

    for hold_key, results in by_hold.items():
        lookup = {r["variant"]: r for r in results}
        r_beat = lookup.get("Beat-Sensor", {})
        r_base = lookup.get("CO2-Bohr", {})
        r_const = lookup.get("Const-D+F", {})

        hr = r_beat.get("hr", np.array([70.0]))
        b_delay = r_beat.get("params", {}).get("b_delay", float("nan"))
        b_avg = r_beat.get("params", {}).get("b_avg", float("nan"))
        d_equiv = b_delay * 60.0 / hr.mean()
        tau_equiv = b_avg * 60.0 / hr.mean()

        print(f"  {hold_key:<16s} {f'{hr.min():.0f}-{hr.max():.0f}':>10s} "
              f"{b_delay:>8.1f} {b_avg:>8.1f} "
              f"{d_equiv:>8.1f} {tau_equiv:>9.1f} "
              f"{r_beat.get('r2', float('nan')):>9.4f} "
              f"{r_base.get('r2', float('nan')):>9.4f} "
              f"{r_const.get('r2', float('nan')):>10.4f}")


def plot_results(all_results: list[dict], output_path: Path):
    by_hold = {}
    for r in all_results:
        key = f"{r['hold_type']} #{r['hold_id']}"
        by_hold.setdefault(key, []).append(r)

    n_holds = len(by_hold)
    fig, axes = plt.subplots(n_holds, 1, figsize=(14, 5 * n_holds), squeeze=False)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for idx, (hold_key, results) in enumerate(by_hold.items()):
        ax = axes[idx, 0]
        r0 = results[0]
        ax.plot(r0["t"], r0["spo2"], "k.", markersize=2, alpha=0.5, label="Observed")

        for i, r in enumerate(results):
            label = f"{r['variant']} (R²={r['r2']:.4f})"
            ax.plot(
                r["t"], r["pred"], color=colors[i % len(colors)], linewidth=1.5,
                alpha=0.8, label=label,
            )

        ax.set_title(f"{hold_key} (HR: {r0['hr'].min():.0f}-{r0['hr'].max():.0f} bpm)",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("SpO2 (%)")
        ax.set_ylim(30, 105)
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "v5 Exp 5: HR-Coupled Beat-Based Sensor",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    print("=" * 100)
    print("v5 EXPERIMENT 5: HR-Coupled Beat-Based Sensor")
    print("=" * 100)

    print("\nLoading hold data from DB...")
    holds_by_type = load_holds_by_type()

    for htype, holds in holds_by_type.items():
        for h in holds:
            hr = h["hr"]
            print(
                f"  {htype} #{h['id']}: {len(h['t'])}pts, SpO2 {h['spo2'].min():.0f}-{h['spo2'].max():.0f}%, "
                f"HR {hr.min():.0f}-{hr.max():.0f} bpm (mean={hr.mean():.0f})"
            )

    all_results = []

    for variant in VARIANTS:
        for htype, holds in holds_by_type.items():
            for hold in holds:
                print(f"\n{'─'*60}", flush=True)
                print(
                    f"Fitting {variant.name} to {hold['type']} #{hold['id']} "
                    f"({len(hold['t'])} pts, {len(variant.param_names)} params)",
                    flush=True,
                )
                print(f"{'─'*60}", flush=True)

                result = fit_variant(variant, hold)
                all_results.append(result)

                status = "OK" if result["converged"] else "WARN"
                print(
                    f"  [{status}] R²={result['r2']:.6f}, RMSE={result['rmse']:.4f}, "
                    f"at_bounds={result['n_at_bounds']}/{result['n_params']}",
                    flush=True,
                )

    print_results(all_results)

    output_dir = Path(__file__).resolve().parent
    plot_results(all_results, output_dir / "exp_v5_05_beat_sensor.png")


if __name__ == "__main__":
    main()
