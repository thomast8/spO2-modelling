"""
v4 model comparison: sensor delay + censored loss.

Tests 4 model variants across all hold types (FL, FRC, RV):
  1. CO2-Bohr       — 7 params, r_offset only, weighted SSE
  2. CO2-Bohr+Delay — 8 params, pure delay + clip, weighted SSE
  3. CO2-Bohr+Delay+Cens — 8 params, pure delay + clip, censored NLL
  4. Richards       — 5 params, descriptive sigmoid, unweighted SSE

v4 hypothesis: physiology params compensate for missing sensor delay.
Adding a pure time delay (circulation lag) should free gamma from pinning
at its upper bound and reduce params-at-bounds.

Usage:
    cd backend && uv run python -u scripts/compare_v4_model.py
"""

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import norm

DB_PATH = Path(__file__).resolve().parents[3] / "data" / "spo2.db"

P50_BASE = 26.6  # Baseline P50 (mmHg), fixed haemoglobin constant
SIGMA_FIXED = 1.5  # Fixed noise std for censored likelihood


# ── Data loading ─────────────────────────────────────────────────────────────


def load_holds_by_type() -> dict[str, list[dict]]:
    """Load all tagged holds from DB, grouped by hold type."""
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
    """Exponential PAO2 decay to pvo2 floor."""
    return pvo2 + (pao2_0 - pvo2) * np.exp(-t / max(tau, 0.01))


def p50_linear_co2(t, paco2_0, k_co2):
    """Linear CO2 rise -> linear P50 shift via Bohr effect.

    P50_eff = 26.6 + 0.48 * (PaCO2(t) - 40)
    where PaCO2(t) = paco2_0 + k_co2 * t
    beta = 0.48 mmHg P50 / mmHg CO2, fixed from literature.
    """
    paco2 = paco2_0 + k_co2 * t
    return P50_BASE + 0.48 * (paco2 - 40.0)


def odc_severinghaus(pao2, p50_eff, gamma):
    """Severinghaus (1979) with virtual PO2 Bohr + gamma steepness."""
    pao2_virtual = pao2 * (P50_BASE / np.maximum(p50_eff, 0.01))
    pao2_adj = P50_BASE * (np.maximum(pao2_virtual, 0.01) / P50_BASE) ** gamma
    x = np.maximum(pao2_adj, 0.01)
    return 100.0 / (1.0 + 23400.0 / (x**3 + 150.0 * x))


# ── Predict functions ────────────────────────────────────────────────────────


def predict_co2bohr(t, hr, params):
    """CO2-Bohr: 7 params, r_offset only."""
    pao2_0, pvo2, tau_washout, gamma, paco2_0, k_co2, r_offset = params
    pao2 = pao2_exponential(t, pao2_0, pvo2, tau_washout)
    p50 = p50_linear_co2(t, paco2_0, k_co2)
    sa = odc_severinghaus(pao2, p50, gamma)
    return np.clip(sa + r_offset, 0.0, 100.0)


def predict_co2bohr_delay(t, hr, params):
    """CO2-Bohr+Delay: 8 params, pure time delay + clip."""
    pao2_0, pvo2, tau_washout, gamma, paco2_0, k_co2, r_offset, d = params
    # Physiology (identical to CO2-Bohr)
    pao2 = pao2_exponential(t, pao2_0, pvo2, tau_washout)
    p50 = p50_linear_co2(t, paco2_0, k_co2)
    sa = odc_severinghaus(pao2, p50, gamma)
    # Observation: pure delay
    sa_delayed = np.interp(t - d, t, sa, left=sa[0])
    return np.clip(sa_delayed + r_offset, 0.0, 100.0)


def predict_richards(t, hr, params):
    """Richards (generalized logistic) sigmoid — descriptive fallback (5 params)."""
    s_max, s_min, t50, k, nu = params
    z = np.clip((t - t50) / max(k, 0.01), -500, 500)
    base = 1.0 + nu * np.exp(z)
    return np.clip(
        s_min + (s_max - s_min) / np.power(np.maximum(base, 1e-10), 1.0 / nu), 0.0, 100.0
    )


# ── Loss functions ───────────────────────────────────────────────────────────


def loss_weighted_sse(obs, pred, alpha=3.0):
    """Weighted SSE with higher weight on drop region (SpO2 < 95)."""
    weights = np.where(obs < 95, alpha, 1.0)
    return np.sum(weights * (obs - pred) ** 2)


def loss_censored_vec(obs, pred, sigma=SIGMA_FIXED):
    """Vectorized censored NLL — normal for observed, CDF for censored at 40%."""
    uncensored = obs > 40
    nll = 0.0
    if np.any(uncensored):
        nll -= np.sum(norm.logpdf(obs[uncensored], pred[uncensored], sigma))
    if np.any(~uncensored):
        cdf_vals = norm.cdf(40, pred[~uncensored], sigma)
        nll -= np.sum(np.log(np.maximum(cdf_vals, 1e-300)))
    return nll


# ── Bounds per hold type ─────────────────────────────────────────────────────

# CO2-Bohr: [pao2_0, pvo2, tau_washout, gamma, paco2_0, k_co2, r_offset]
CO2BOHR_BOUNDS = {
    "FL": [(100, 250), (20, 50), (50, 250), (0.8, 2.0), (25, 45), (0.02, 0.15), (-3, 3)],
    "FRC": [(80, 120), (20, 50), (20, 100), (0.8, 2.0), (30, 45), (0.02, 0.15), (-3, 3)],
    "RV": [(70, 110), (20, 50), (10, 80), (0.8, 2.0), (35, 50), (0.02, 0.15), (-3, 3)],
}

# CO2-Bohr+Delay: same + d; gamma narrowed to 0.8-1.5
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

RICHARDS_BOUNDS = {
    "FL": [(96, 101), (0, 96), (50, 500), (5, 80), (0.1, 10.0)],
    "FRC": [(96, 101), (0, 96), (20, 300), (3, 60), (0.1, 10.0)],
    "RV": [(96, 101), (0, 96), (10, 250), (3, 60), (0.1, 10.0)],
}


# ── Model variant definitions ────────────────────────────────────────────────


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
        name="CO2-Bohr+Delay+Cens",
        param_names=[
            "pao2_0", "pvo2", "tau_washout", "gamma", "paco2_0", "k_co2", "r_offset", "d",
        ],
        bounds_by_type=CO2BOHR_DELAY_BOUNDS,
        predict_fn=predict_co2bohr_delay,
        loss_fn=lambda obs, pred: loss_censored_vec(obs, pred),
    ),
    ModelVariant(
        name="Richards",
        param_names=["s_max", "s_min", "t50", "k", "nu"],
        bounds_by_type=RICHARDS_BOUNDS,
        predict_fn=predict_richards,
        loss_fn=lambda obs, pred: np.sum((obs - pred) ** 2),
    ),
]


# ── Metrics ──────────────────────────────────────────────────────────────────


def compute_r2(obs, pred):
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def compute_rmse(obs, pred):
    return float(np.sqrt(np.mean((obs - pred) ** 2)))


def compute_drop_metrics(obs, pred, threshold=95.0):
    """R² and RMSE for the drop region only (SpO2 < threshold)."""
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


# ── Fitting ──────────────────────────────────────────────────────────────────


def fit_variant(variant: ModelVariant, hold: dict) -> dict:
    """Fit a variant to a single hold. Returns metrics dict."""
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
    }


# ── Output ───────────────────────────────────────────────────────────────────


def print_results(all_results: list[dict]):
    """Print comparison tables."""
    by_hold = {}
    for r in all_results:
        key = f"{r['hold_type']} #{r['hold_id']}"
        by_hold.setdefault(key, []).append(r)

    variant_names = list(dict.fromkeys(r["variant"] for r in all_results))

    for hold_key, results in by_hold.items():
        print(f"\n{'='*80}")
        print(f"  {hold_key}")
        print(f"{'='*80}")

        header = f"  {'Metric':<22s}" + "".join(f" {n:>16s}" for n in variant_names)
        print(header)
        print(f"  {'-'*22}" + "".join(f" {'-'*16}" for _ in variant_names))

        lookup = {r["variant"]: r for r in results}

        # R²
        row = f"  {'R²':<22s}"
        for v in variant_names:
            r = lookup.get(v)
            row += f" {r['r2']:>16.6f}" if r else f" {'N/A':>16s}"
        print(row)

        # R² drop
        row = f"  {'R² (drop <95%)':<22s}"
        for v in variant_names:
            r = lookup.get(v)
            if r and r["r2_drop"] is not None:
                row += f" {r['r2_drop']:>16.6f}"
            else:
                row += f" {'N/A':>16s}"
        print(row)

        # RMSE
        row = f"  {'RMSE':<22s}"
        for v in variant_names:
            r = lookup.get(v)
            row += f" {r['rmse']:>16.4f}" if r else f" {'N/A':>16s}"
        print(row)

        # RMSE drop
        row = f"  {'RMSE (drop)':<22s}"
        for v in variant_names:
            r = lookup.get(v)
            if r and r["rmse_drop"] is not None:
                row += f" {r['rmse_drop']:>16.4f}"
            else:
                row += f" {'N/A':>16s}"
        print(row)

        # Params
        row = f"  {'# params':<22s}"
        for v in variant_names:
            r = lookup.get(v)
            row += f" {r['n_params']:>16d}" if r else f" {'N/A':>16s}"
        print(row)

        # At bounds
        row = f"  {'# at bounds':<22s}"
        for v in variant_names:
            r = lookup.get(v)
            row += f" {r['n_at_bounds']:>16d}" if r else f" {'N/A':>16s}"
        print(row)

        # At-bounds detail
        for v in variant_names:
            r = lookup.get(v)
            if r and r["at_bounds"]:
                print(f"    {v}: {r['at_bounds']}")

        # Fitted params
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


def plot_results(all_results: list[dict], output_path: Path):
    """Plot observed vs predicted for all variants, grouped by hold."""
    by_hold = {}
    for r in all_results:
        key = f"{r['hold_type']} #{r['hold_id']}"
        by_hold.setdefault(key, []).append(r)

    n_holds = len(by_hold)
    fig, axes = plt.subplots(n_holds, 1, figsize=(14, 5 * n_holds), squeeze=False)

    colors = ["#1f77b4", "#ff7f0e", "#d62728", "#2ca02c", "#9467bd"]

    for idx, (hold_key, results) in enumerate(by_hold.items()):
        ax = axes[idx, 0]
        r0 = results[0]
        ax.plot(r0["t"], r0["spo2"], "k.", markersize=2, alpha=0.5, label="Observed")

        for i, r in enumerate(results):
            label = f"{r['variant']} (R²={r['r2']:.4f})"
            if r["r2_drop"] is not None:
                label += f", drop={r['r2_drop']:.4f}"
            ax.plot(
                r["t"], r["pred"], color=colors[i % len(colors)], linewidth=1.5,
                alpha=0.8, label=label,
            )

        ax.set_title(hold_key, fontsize=14, fontweight="bold")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("SpO2 (%)")
        ax.set_ylim(30, 105)
        ax.axhline(y=95, color="gray", linestyle="--", alpha=0.3, label="95% threshold")
        ax.axhline(y=40, color="red", linestyle="--", alpha=0.3, label="40% censor")
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────


def flush_print(*args, **kwargs):
    print(*args, **kwargs, flush=True)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="v4 model comparison: sensor delay + censored loss")
    parser.add_argument("--only", nargs="+", help="Run only these variants (by name)")
    args = parser.parse_args()

    flush_print("Loading hold data from DB...")
    holds_by_type = load_holds_by_type()

    for htype, holds in holds_by_type.items():
        flush_print(
            f"  {htype}: {len(holds)} holds — "
            + ", ".join(
                f"#{h['id']} ({len(h['t'])}pts, {h['spo2'].min():.0f}-{h['spo2'].max():.0f}%)"
                for h in holds
            )
        )

    variants_to_run = VARIANTS
    if args.only:
        variants_to_run = [v for v in VARIANTS if v.name in args.only]
        flush_print(f"\nRunning only: {[v.name for v in variants_to_run]}")

    all_results = []

    for variant in variants_to_run:
        for htype, holds in holds_by_type.items():
            for hold in holds:
                flush_print(f"\n{'─'*60}")
                flush_print(
                    f"Fitting {variant.name} to {hold['type']} #{hold['id']} "
                    f"({len(hold['t'])} pts, {len(variant.param_names)} params)"
                )
                flush_print(f"{'─'*60}")

                result = fit_variant(variant, hold)
                all_results.append(result)

                status = "OK" if result["converged"] else "WARN"
                flush_print(
                    f"  [{status}] R²={result['r2']:.6f}, RMSE={result['rmse']:.4f}, "
                    f"at_bounds={result['n_at_bounds']}/{result['n_params']}"
                )
                if result["r2_drop"] is not None:
                    flush_print(
                        f"       Drop region: R²={result['r2_drop']:.6f}, "
                        f"RMSE={result['rmse_drop']:.4f}"
                    )

    print_results(all_results)

    output_dir = Path(__file__).resolve().parent
    plot_results(all_results, output_dir / "compare_v4_model.png")


if __name__ == "__main__":
    main()
