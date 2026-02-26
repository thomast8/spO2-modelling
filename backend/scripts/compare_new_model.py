"""
Compare model architectures for SpO2 desaturation fitting.

Tests 5 model variants across all hold types (FL, FRC, RV):
  1. Baseline — current Severinghaus+gamma (7 params)
  2. Proposal — Weibull PAO2 + saturating P50 + Hill ODC + delay/filter + censored loss (9 params)
  3. Hybrid  — Weibull PAO2 + linear CO2 Bohr + Sev+gamma + delay/filter + censored loss (9 params)
  4. Current+obs — exponential PAO2 + Sev+gamma + delay/filter + censored loss (9 params)
  5. Richards — descriptive sigmoid fallback (5 params)

Usage:
    cd backend && uv run python scripts/compare_new_model.py
"""

import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import norm

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "spo2.db"

P50_BASE = 26.6  # Baseline P50 (mmHg), fixed haemoglobin constant


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


# ── PAO2 decay functions ────────────────────────────────────────────────────


def pao2_exponential(t, pao2_0, pvo2, tau):
    """Current: exponential decay to pvo2 floor."""
    return pvo2 + (pao2_0 - pvo2) * np.exp(-t / max(tau, 0.01))


def pao2_weibull(t, pa0, tau, p):
    """Proposed: stretched exponential (Weibull) decay toward zero."""
    return pa0 * np.exp(-((t / max(tau, 0.01)) ** p))


# ── P50 / Bohr functions ───────────────────────────────────────────────────


def p50_linear_co2(t, paco2_0, k_co2):
    """Current: linear CO2 rise -> linear P50 shift.

    P50_eff = 26.6 + 0.48 * (PaCO2(t) - 40)
    where PaCO2(t) = paco2_0 + k_co2 * t
    beta = 0.48 mmHg P50 / mmHg CO2, fixed from literature.
    """
    paco2 = paco2_0 + k_co2 * t
    return P50_BASE + 0.48 * (paco2 - 40.0)


def p50_saturating(t, p50_start, delta_p50, tau_co2):
    """Proposed: saturating exponential P50 shift."""
    return p50_start + delta_p50 * (1.0 - np.exp(-t / max(tau_co2, 0.01)))


# ── ODC functions ───────────────────────────────────────────────────────────


def odc_severinghaus(pao2, p50_eff, gamma):
    """Current: Severinghaus (1979) with virtual PO2 Bohr + gamma steepness."""
    pao2_virtual = pao2 * (P50_BASE / np.maximum(p50_eff, 0.01))
    pao2_adj = P50_BASE * (np.maximum(pao2_virtual, 0.01) / P50_BASE) ** gamma
    x = np.maximum(pao2_adj, 0.01)
    return 100.0 / (1.0 + 23400.0 / (x**3 + 150.0 * x))


def odc_hill(pao2, p50_eff, n):
    """Proposed: Hill equation."""
    pao2 = np.maximum(pao2, 0.01)
    p50 = np.maximum(p50_eff, 0.01)
    return 100.0 * (pao2**n) / (pao2**n + p50**n)


# ── Observation model ──────────────────────────────────────────────────────


def obs_none(sa, t, r_offset):
    """Current: just offset + clip."""
    return np.clip(sa + r_offset, 0.0, 100.0)


def obs_delay_filter(sa, t, d, tau_f, r_offset):
    """Proposed: time delay + IIR low-pass filter + offset + clip."""
    from scipy.signal import lfilter

    # Delay: shift signal by d seconds (interpolated)
    sa_delayed = np.interp(t - d, t, sa, left=sa[0])

    # IIR low-pass filter via scipy.signal.lfilter (vectorized, no Python loop)
    # y[i] = (1-alpha)*y[i-1] + alpha*x[i]  =>  lfilter([alpha], [1, -(1-alpha)], x)
    dt = 1.0
    alpha = dt / (max(tau_f, 0.01) + dt)
    s_meas = lfilter([alpha], [1.0, -(1.0 - alpha)], sa_delayed)

    return np.clip(s_meas + r_offset, 0.0, 100.0)


# ── Loss functions ──────────────────────────────────────────────────────────

SIGMA_FIXED = 1.5  # Fixed noise std for censored likelihood


def loss_weighted_sse(obs, pred, alpha=3.0):
    """Current: weighted SSE with higher weight on drop region (SpO2 < 95)."""
    weights = np.where(obs < 95, alpha, 1.0)
    return np.sum(weights * (obs - pred) ** 2)


def loss_censored(obs, pred, sigma=SIGMA_FIXED):
    """Proposed: censored likelihood — normal for observed, CDF for censored at 40%."""
    nll = 0.0
    for y, yhat in zip(obs, pred):
        if y > 40:
            nll -= norm.logpdf(y, yhat, sigma)
        else:
            # Censored: we only know true value <= 40
            cdf_val = norm.cdf(40, yhat, sigma)
            nll -= np.log(max(cdf_val, 1e-300))
    return nll


def loss_censored_vec(obs, pred, sigma=SIGMA_FIXED):
    """Vectorized censored NLL for speed."""
    uncensored = obs > 40
    nll = 0.0
    if np.any(uncensored):
        nll -= np.sum(norm.logpdf(obs[uncensored], pred[uncensored], sigma))
    if np.any(~uncensored):
        cdf_vals = norm.cdf(40, pred[~uncensored], sigma)
        nll -= np.sum(np.log(np.maximum(cdf_vals, 1e-300)))
    return nll


# ── Model variant definitions ──────────────────────────────────────────────


@dataclass
class ModelVariant:
    name: str
    param_names: list[str]
    bounds_by_type: dict[str, list[tuple[float, float]]]
    predict_fn: callable
    loss_fn: callable


def predict_baseline(t, hr, params):
    """Variant 1: Current Severinghaus+gamma model (7 params)."""
    pao2_0, pvo2, tau_washout, gamma, paco2_0, k_co2, r_offset = params
    pao2 = pao2_exponential(t, pao2_0, pvo2, tau_washout)
    p50 = p50_linear_co2(t, paco2_0, k_co2)
    sa = odc_severinghaus(pao2, p50, gamma)
    return obs_none(sa, t, r_offset)


def predict_proposal(t, hr, params):
    """Variant 2: Weibull + saturating P50 + Hill + delay/filter (9 params)."""
    pa0, tau_w, p_w, p50_start, delta_p50, tau_co2, d, tau_f, r_offset = params
    pao2 = pao2_weibull(t, pa0, tau_w, p_w)
    p50 = p50_saturating(t, p50_start, delta_p50, tau_co2)
    sa = odc_hill(pao2, p50, 2.7)  # n fixed at 2.7
    return obs_delay_filter(sa, t, d, tau_f, r_offset)


def predict_hybrid(t, hr, params):
    """Variant 3: Weibull + linear CO2 Bohr + Sev+gamma + delay/filter (9 params)."""
    pa0, tau_w, p_w, gamma, paco2_0, k_co2, d, tau_f, r_offset = params
    pao2 = pao2_weibull(t, pa0, tau_w, p_w)
    p50 = p50_linear_co2(t, paco2_0, k_co2)
    sa = odc_severinghaus(pao2, p50, gamma)
    return obs_delay_filter(sa, t, d, tau_f, r_offset)


def predict_current_plus_obs(t, hr, params):
    """Variant 4: Current model + delay/filter observation model (9 params)."""
    pao2_0, pvo2, tau_washout, gamma, paco2_0, k_co2, d, tau_f, r_offset = params
    pao2 = pao2_exponential(t, pao2_0, pvo2, tau_washout)
    p50 = p50_linear_co2(t, paco2_0, k_co2)
    sa = odc_severinghaus(pao2, p50, gamma)
    return obs_delay_filter(sa, t, d, tau_f, r_offset)


def predict_richards(t, hr, params):
    """Variant 5: Richards (generalized logistic) sigmoid — descriptive fallback (5 params).

    S(t) = s_min + (s_max - s_min) / (1 + nu * exp((t - t50)/k))^(1/nu)

    Decreasing sigmoid: S → s_max as t → -∞, S → s_min as t → +∞.
    nu controls asymmetry: nu=1 is standard logistic, nu<1 sharper onset, nu>1 sharper nadir.
    """
    s_max, s_min, t50, k, nu = params
    z = np.clip((t - t50) / max(k, 0.01), -500, 500)
    base = 1.0 + nu * np.exp(z)
    return np.clip(s_min + (s_max - s_min) / np.power(np.maximum(base, 1e-10), 1.0 / nu), 0.0, 100.0)


# ── Bounds per hold type ───────────────────────────────────────────────────

BASELINE_BOUNDS = {
    "FL": [
        (100, 250),   # pao2_0
        (20, 50),     # pvo2
        (50, 250),    # tau_washout
        (0.8, 2.0),   # gamma
        (25, 45),     # paco2_0 (can be low after hyperventilation)
        (0.02, 0.15), # k_co2 (mmHg/s; ~1-9 mmHg/min)
        (-3.0, 3.0),  # r_offset
    ],
    "FRC": [
        (80, 120),
        (20, 50),
        (20, 100),
        (0.8, 2.0),
        (30, 45),
        (0.02, 0.15),
        (-3.0, 3.0),
    ],
    "RV": [
        (70, 110),
        (20, 50),
        (10, 80),
        (0.8, 2.0),
        (35, 50),
        (0.02, 0.15),
        (-3.0, 3.0),
    ],
}

PROPOSAL_BOUNDS = {
    "FL": [
        (90, 120),    # pa0 (Weibull initial)
        (150, 1200),  # tau_w (Weibull time constant)
        (1.0, 4.5),   # p_w (Weibull shape)
        (15, 30),     # p50_start
        (0, 20),      # delta_p50
        (30, 600),    # tau_co2
        (5, 30),      # d (delay)
        (0.5, 8),     # tau_f (filter)
        (-3.0, 3.0),  # r_offset
    ],
    "FRC": [
        (80, 110),
        (80, 700),
        (1.0, 4.0),
        (15, 30),
        (0, 20),
        (30, 600),
        (5, 30),
        (0.5, 8),
        (-3.0, 3.0),
    ],
    "RV": [
        (70, 100),
        (40, 500),
        (1.0, 3.5),
        (15, 30),
        (0, 20),
        (30, 600),
        (5, 30),
        (0.5, 8),
        (-3.0, 3.0),
    ],
}

HYBRID_BOUNDS = {
    "FL": [
        (90, 120),    # pa0
        (150, 1200),  # tau_w
        (1.0, 4.5),   # p_w
        (0.8, 2.0),   # gamma
        (25, 45),     # paco2_0
        (0.02, 0.15), # k_co2
        (5, 30),      # d
        (0.5, 8),     # tau_f
        (-3.0, 3.0),  # r_offset
    ],
    "FRC": [
        (80, 110),
        (80, 700),
        (1.0, 4.0),
        (0.8, 2.0),
        (30, 45),
        (0.02, 0.15),
        (5, 30),
        (0.5, 8),
        (-3.0, 3.0),
    ],
    "RV": [
        (70, 100),
        (40, 500),
        (1.0, 3.5),
        (0.8, 2.0),
        (35, 50),
        (0.02, 0.15),
        (5, 30),
        (0.5, 8),
        (-3.0, 3.0),
    ],
}

CURRENT_OBS_BOUNDS = {
    "FL": [
        (100, 250),   # pao2_0
        (20, 50),     # pvo2
        (50, 250),    # tau_washout
        (0.8, 2.0),   # gamma
        (25, 45),     # paco2_0
        (0.02, 0.15), # k_co2
        (5, 30),      # d
        (0.5, 8),     # tau_f
        (-3.0, 3.0),  # r_offset
    ],
    "FRC": [
        (80, 120),
        (20, 50),
        (20, 100),
        (0.8, 2.0),
        (30, 45),
        (0.02, 0.15),
        (5, 30),
        (0.5, 8),
        (-3.0, 3.0),
    ],
    "RV": [
        (70, 110),
        (20, 50),
        (10, 80),
        (0.8, 2.0),
        (35, 50),
        (0.02, 0.15),
        (5, 30),
        (0.5, 8),
        (-3.0, 3.0),
    ],
}

RICHARDS_BOUNDS = {
    "FL": [
        (96, 101),    # s_max (plateau SpO2, slightly above 100 for offset)
        (0, 96),      # s_min (nadir — wide range, FL#1 barely drops)
        (50, 500),    # t50 (midpoint time — FL has long plateaus)
        (5, 80),      # k (time scale of sigmoid transition)
        (0.1, 10.0),  # nu (asymmetry)
    ],
    "FRC": [
        (96, 101),
        (0, 96),
        (20, 300),
        (3, 60),
        (0.1, 10.0),
    ],
    "RV": [
        (96, 101),
        (0, 96),
        (10, 250),
        (3, 60),
        (0.1, 10.0),
    ],
}

VARIANTS = [
    ModelVariant(
        name="Baseline",
        param_names=["pao2_0", "pvo2", "tau_washout", "gamma", "paco2_0", "k_co2", "r_offset"],
        bounds_by_type=BASELINE_BOUNDS,
        predict_fn=predict_baseline,
        loss_fn=lambda obs, pred: loss_weighted_sse(obs, pred, alpha=3.0),
    ),
    ModelVariant(
        name="Proposal",
        param_names=["pa0", "tau_w", "p_w", "p50_start", "delta_p50", "tau_co2", "d", "tau_f", "r_offset"],
        bounds_by_type=PROPOSAL_BOUNDS,
        predict_fn=predict_proposal,
        loss_fn=lambda obs, pred: loss_censored_vec(obs, pred),
    ),
    ModelVariant(
        name="Hybrid",
        param_names=["pa0", "tau_w", "p_w", "gamma", "paco2_0", "k_co2", "d", "tau_f", "r_offset"],
        bounds_by_type=HYBRID_BOUNDS,
        predict_fn=predict_hybrid,
        loss_fn=lambda obs, pred: loss_censored_vec(obs, pred),
    ),
    ModelVariant(
        name="Current+obs",
        param_names=["pao2_0", "pvo2", "tau_washout", "gamma", "paco2_0", "k_co2", "d", "tau_f", "r_offset"],
        bounds_by_type=CURRENT_OBS_BOUNDS,
        predict_fn=predict_current_plus_obs,
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


# ── Metrics ────────────────────────────────────────────────────────────────


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


# ── Fitting ────────────────────────────────────────────────────────────────


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


# ── Output ─────────────────────────────────────────────────────────────────


def print_results(all_results: list[dict]):
    """Print comparison tables."""
    # Group by hold
    by_hold = {}
    for r in all_results:
        key = f"{r['hold_type']} #{r['hold_id']}"
        by_hold.setdefault(key, []).append(r)

    # Use only variants present in results
    variant_names = list(dict.fromkeys(r["variant"] for r in all_results))

    for hold_key, results in by_hold.items():
        print(f"\n{'='*80}")
        print(f"  {hold_key}")
        print(f"{'='*80}")

        # Header
        header = f"  {'Metric':<22s}" + "".join(f" {n:>12s}" for n in variant_names)
        print(header)
        print(f"  {'-'*22}" + "".join(f" {'-'*12}" for _ in variant_names))

        # Build lookup
        lookup = {r["variant"]: r for r in results}

        # R²
        row = f"  {'R²':<22s}"
        for v in variant_names:
            r = lookup.get(v)
            row += f" {r['r2']:>12.6f}" if r else f" {'N/A':>12s}"
        print(row)

        # R² drop
        row = f"  {'R² (drop <95%)':<22s}"
        for v in variant_names:
            r = lookup.get(v)
            if r and r["r2_drop"] is not None:
                row += f" {r['r2_drop']:>12.6f}"
            else:
                row += f" {'N/A':>12s}"
        print(row)

        # RMSE
        row = f"  {'RMSE':<22s}"
        for v in variant_names:
            r = lookup.get(v)
            row += f" {r['rmse']:>12.4f}" if r else f" {'N/A':>12s}"
        print(row)

        # RMSE drop
        row = f"  {'RMSE (drop)':<22s}"
        for v in variant_names:
            r = lookup.get(v)
            if r and r["rmse_drop"] is not None:
                row += f" {r['rmse_drop']:>12.4f}"
            else:
                row += f" {'N/A':>12s}"
        print(row)

        # Params
        row = f"  {'# params':<22s}"
        for v in variant_names:
            r = lookup.get(v)
            row += f" {r['n_params']:>12d}" if r else f" {'N/A':>12s}"
        print(row)

        # At bounds
        row = f"  {'# at bounds':<22s}"
        for v in variant_names:
            r = lookup.get(v)
            row += f" {r['n_at_bounds']:>12d}" if r else f" {'N/A':>12s}"
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
            bounds = dict(zip(vdef.param_names, vdef.bounds_by_type[r["hold_type"]], strict=True))
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

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for idx, (hold_key, results) in enumerate(by_hold.items()):
        ax = axes[idx, 0]
        # Plot observed data
        r0 = results[0]
        ax.plot(r0["t"], r0["spo2"], "k.", markersize=2, alpha=0.5, label="Observed")

        # Plot each variant
        for i, r in enumerate(results):
            label = f"{r['variant']} (R²={r['r2']:.4f})"
            if r["r2_drop"] is not None:
                label += f", drop={r['r2_drop']:.4f}"
            ax.plot(r["t"], r["pred"], color=colors[i % len(colors)], linewidth=1.5,
                    alpha=0.8, label=label)

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


# ── Main ───────────────────────────────────────────────────────────────────


def flush_print(*args, **kwargs):
    print(*args, **kwargs, flush=True)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compare model variants for SpO2 fitting")
    parser.add_argument("--only", nargs="+", help="Run only these variants (by name)")
    args = parser.parse_args()

    flush_print("Loading hold data from DB...")
    holds_by_type = load_holds_by_type()

    for htype, holds in holds_by_type.items():
        flush_print(f"  {htype}: {len(holds)} holds — "
                    + ", ".join(f"#{h['id']} ({len(h['t'])}pts, {h['spo2'].min():.0f}-{h['spo2'].max():.0f}%)"
                                for h in holds))

    variants_to_run = VARIANTS
    if args.only:
        variants_to_run = [v for v in VARIANTS if v.name in args.only]
        flush_print(f"\nRunning only: {[v.name for v in variants_to_run]}")

    all_results = []

    # Fit each variant to each hold
    for variant in variants_to_run:
        for htype, holds in holds_by_type.items():
            for hold in holds:
                flush_print(f"\n{'─'*60}")
                flush_print(f"Fitting {variant.name} to {hold['type']} #{hold['id']} "
                            f"({len(hold['t'])} pts, {len(variant.param_names)} params)")
                flush_print(f"{'─'*60}")

                result = fit_variant(variant, hold)
                all_results.append(result)

                status = "OK" if result["converged"] else "WARN"
                flush_print(f"  [{status}] R²={result['r2']:.6f}, RMSE={result['rmse']:.4f}, "
                            f"at_bounds={result['n_at_bounds']}/{result['n_params']}")
                if result["r2_drop"] is not None:
                    flush_print(f"       Drop region: R²={result['r2_drop']:.6f}, "
                                f"RMSE={result['rmse_drop']:.4f}")

    # Print comparison
    print_results(all_results)

    # Plot
    output_dir = Path(__file__).resolve().parent
    plot_results(all_results, output_dir / "compare_new_model.png")


if __name__ == "__main__":
    main()
