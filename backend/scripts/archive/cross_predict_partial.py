"""
Partial-transfer cross-prediction: fix shared physiology, re-fit initial conditions.

Tests whether mechanistic model parameters separate cleanly into:
  - Shared physiology (same person): pvo2, gamma, Bohr params, r_offset
  - Type-specific initial conditions: pao2_0, tau_washout (+ paco2_0 for CO2-Bohr)

If partial transfer works, the shared params carry real physiological meaning
and a "global fit" with shared physiology + per-type initial conditions is viable.

Usage:
    cd backend && uv run python -u scripts/cross_predict_partial.py
"""

import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution

DB_PATH = Path(__file__).resolve().parents[3] / "data" / "spo2.db"
P50_BASE = 26.6


# ── Data loading ────────────────────────────────────────────────────────────


def load_all_holds() -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    holds = conn.execute(
        "SELECT id, hold_type FROM holds WHERE hold_type != 'untagged' ORDER BY id"
    ).fetchall()
    result = []
    for hold_id, hold_type in holds:
        rows = conn.execute(
            "SELECT elapsed_s, spo2, hr FROM hold_data WHERE hold_id = ? ORDER BY elapsed_s",
            (hold_id,),
        ).fetchall()
        if not rows:
            continue
        result.append({
            "id": hold_id,
            "type": hold_type,
            "t": np.array([r[0] for r in rows], dtype=float),
            "spo2": np.array([r[1] for r in rows], dtype=float),
            "hr": np.array([r[2] for r in rows], dtype=float),
        })
    conn.close()
    return result


# ── Model functions ─────────────────────────────────────────────────────────


def predict_production(t, params):
    pao2_0, pvo2, tau_washout, gamma, bohr_max, tau_bohr, r_offset = params
    pao2 = pvo2 + (pao2_0 - pvo2) * np.exp(-t / max(tau_washout, 0.01))
    p50 = P50_BASE + bohr_max * (1.0 - np.exp(-t / max(tau_bohr, 0.01)))
    pao2_v = pao2 * (P50_BASE / np.maximum(p50, 0.01))
    pao2_adj = P50_BASE * (np.maximum(pao2_v, 0.01) / P50_BASE) ** gamma
    x = np.maximum(pao2_adj, 0.01)
    sa = 100.0 / (1.0 + 23400.0 / (x**3 + 150.0 * x))
    return np.clip(sa + r_offset, 0.0, 100.0)


def predict_co2bohr(t, params):
    pao2_0, pvo2, tau_washout, gamma, paco2_0, k_co2, r_offset = params
    pao2 = pvo2 + (pao2_0 - pvo2) * np.exp(-t / max(tau_washout, 0.01))
    paco2 = paco2_0 + k_co2 * t
    p50 = P50_BASE + 0.48 * (paco2 - 40.0)
    pao2_v = pao2 * (P50_BASE / np.maximum(p50, 0.01))
    pao2_adj = P50_BASE * (np.maximum(pao2_v, 0.01) / P50_BASE) ** gamma
    x = np.maximum(pao2_adj, 0.01)
    sa = 100.0 / (1.0 + 23400.0 / (x**3 + 150.0 * x))
    return np.clip(sa + r_offset, 0.0, 100.0)


# ── Bounds ──────────────────────────────────────────────────────────────────

# Full bounds (all params)
PRODUCTION_BOUNDS = {
    "FL": [(100, 250), (20, 50), (50, 250), (0.8, 2.0), (2, 15), (40, 250), (-3, 3)],
    "FRC": [(80, 120), (20, 50), (20, 100), (0.8, 2.0), (2, 15), (40, 250), (-3, 3)],
    "RV": [(70, 110), (20, 50), (10, 80), (0.8, 2.0), (2, 15), (40, 250), (-3, 3)],
}
CO2BOHR_BOUNDS = {
    "FL": [(100, 250), (20, 50), (50, 250), (0.8, 2.0), (25, 45), (0.02, 0.15), (-3, 3)],
    "FRC": [(80, 120), (20, 50), (20, 100), (0.8, 2.0), (30, 45), (0.02, 0.15), (-3, 3)],
    "RV": [(70, 110), (20, 50), (10, 80), (0.8, 2.0), (35, 50), (0.02, 0.15), (-3, 3)],
}

# Parameter names and which are type-specific vs shared
MODELS = {
    "Production": {
        "predict": predict_production,
        "bounds": PRODUCTION_BOUNDS,
        "names": ["pao2_0", "pvo2", "tau_washout", "gamma", "bohr_max", "tau_bohr", "r_offset"],
        # Indices of type-specific params (re-fit on target)
        "type_specific": [0, 2],       # pao2_0, tau_washout
        "shared": [1, 3, 4, 5, 6],     # pvo2, gamma, bohr_max, tau_bohr, r_offset
    },
    "CO2-Bohr": {
        "predict": predict_co2bohr,
        "bounds": CO2BOHR_BOUNDS,
        "names": ["pao2_0", "pvo2", "tau_washout", "gamma", "paco2_0", "k_co2", "r_offset"],
        # paco2_0 is type-specific (starting CO2 depends on hyperventilation prep)
        "type_specific": [0, 2, 4],    # pao2_0, tau_washout, paco2_0
        "shared": [1, 3, 5, 6],        # pvo2, gamma, k_co2, r_offset
    },
}


# ── Metrics ─────────────────────────────────────────────────────────────────


def compute_r2(obs, pred):
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def compute_rmse(obs, pred):
    return float(np.sqrt(np.mean((obs - pred) ** 2)))


# ── Fitting ─────────────────────────────────────────────────────────────────


def fit_full(model_name, hold):
    """Fit all params on a hold."""
    model = MODELS[model_name]
    bounds = model["bounds"][hold["type"]]
    predict = model["predict"]
    t, spo2 = hold["t"], hold["spo2"]

    def objective(arr):
        pred = predict(t, arr)
        w = np.where(spo2 < 95, 3.0, 1.0)
        return np.sum(w * (spo2 - pred) ** 2)

    result = differential_evolution(
        objective, bounds, maxiter=3000, seed=42, tol=1e-10,
        polish=True, popsize=40, mutation=(0.5, 1.5), recombination=0.9,
    )
    return result.x


def fit_partial(model_name, source_params, target_hold):
    """Fix shared params from source, re-fit only type-specific on target."""
    model = MODELS[model_name]
    predict = model["predict"]
    type_specific_idx = model["type_specific"]
    shared_idx = model["shared"]
    target_bounds = model["bounds"][target_hold["type"]]

    # Bounds for the free (type-specific) params only
    free_bounds = [target_bounds[i] for i in type_specific_idx]

    t, spo2 = target_hold["t"], target_hold["spo2"]

    def objective(free_arr):
        # Build full param vector: shared from source, free from optimizer
        full = np.copy(source_params)
        for j, idx in enumerate(type_specific_idx):
            full[idx] = free_arr[j]
        pred = predict(t, full)
        w = np.where(spo2 < 95, 3.0, 1.0)
        return np.sum(w * (spo2 - pred) ** 2)

    result = differential_evolution(
        objective, free_bounds, maxiter=1000, seed=42, tol=1e-10,
        polish=True, popsize=20, mutation=(0.5, 1.5), recombination=0.9,
    )

    # Reconstruct full param vector
    full = np.copy(source_params)
    for j, idx in enumerate(type_specific_idx):
        full[idx] = result.x[j]
    return full


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    holds = load_all_holds()
    hold_labels = {h["id"]: f"{h['type']}#{h['id']}" for h in holds}
    hold_by_id = {h["id"]: h for h in holds}

    print("Loaded holds:")
    for h in holds:
        print(f"  {hold_labels[h['id']]}: {len(h['t'])} pts, "
              f"SpO2 {h['spo2'].min():.0f}-{h['spo2'].max():.0f}%", flush=True)

    # ── Step 1: Full fits on all holds ──────────────────────────────────────
    fitted = {}
    for model_name in MODELS:
        for hold in holds:
            label = hold_labels[hold["id"]]
            print(f"\nFull fit {model_name} on {label}...", end="", flush=True)
            params = fit_full(model_name, hold)
            fitted[(model_name, hold["id"])] = params
            pred = MODELS[model_name]["predict"](hold["t"], params)
            r2 = compute_r2(hold["spo2"], pred)
            print(f" R²={r2:.4f}", flush=True)

    # ── Step 2: Partial-transfer cross-prediction ───────────────────────────
    for model_name in MODELS:
        model = MODELS[model_name]
        type_spec_names = [model["names"][i] for i in model["type_specific"]]
        shared_names = [model["names"][i] for i in model["shared"]]

        print(f"\n{'='*100}")
        print(f"PARTIAL TRANSFER: {model_name}")
        print(f"  Shared (fixed from source): {', '.join(shared_names)}")
        print(f"  Type-specific (re-fit on target): {', '.join(type_spec_names)}")
        print(f"{'='*100}")

        # Header
        print(f"\n  {'Source → Target':<22s}  {'Raw R²':>8s}  {'Partial R²':>10s}  {'Self-fit R²':>11s}  "
              f"{'Raw RMSE':>9s}  {'Part RMSE':>9s}  {'Self RMSE':>9s}  "
              f"{'Partial params':s}")
        print(f"  {'-'*22}  {'-'*8}  {'-'*10}  {'-'*11}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*40}")

        for source_hold in holds:
            for target_hold in holds:
                if source_hold["id"] == target_hold["id"]:
                    continue

                src_label = hold_labels[source_hold["id"]]
                tgt_label = hold_labels[target_hold["id"]]
                pair_label = f"{src_label} → {tgt_label}"

                source_params = fitted[(model_name, source_hold["id"])]
                target_params = fitted[(model_name, target_hold["id"])]

                # Raw transfer
                raw_pred = model["predict"](target_hold["t"], source_params)
                raw_r2 = compute_r2(target_hold["spo2"], raw_pred)
                raw_rmse = compute_rmse(target_hold["spo2"], raw_pred)

                # Self-fit
                self_pred = model["predict"](target_hold["t"], target_params)
                self_r2 = compute_r2(target_hold["spo2"], self_pred)
                self_rmse = compute_rmse(target_hold["spo2"], self_pred)

                # Partial transfer
                print(f"  Fitting {pair_label}...", end="", flush=True)
                partial_params = fit_partial(model_name, source_params, target_hold)
                partial_pred = model["predict"](target_hold["t"], partial_params)
                partial_r2 = compute_r2(target_hold["spo2"], partial_pred)
                partial_rmse = compute_rmse(target_hold["spo2"], partial_pred)

                # Show the re-fitted type-specific param values
                param_str = ", ".join(
                    f"{model['names'][i]}={partial_params[i]:.1f}"
                    for i in model["type_specific"]
                )

                print(f"\r  {pair_label:<22s}  {raw_r2:>8.4f}  {partial_r2:>10.4f}  {self_r2:>11.4f}  "
                      f"{raw_rmse:>9.2f}  {partial_rmse:>9.2f}  {self_rmse:>9.2f}  "
                      f"{param_str}", flush=True)

    # ── Step 3: Summary tables ──────────────────────────────────────────────

    # Cross-type pairs (the most interesting ones)
    cross_type_pairs = []
    for src in holds:
        for tgt in holds:
            if src["id"] != tgt["id"] and src["type"] != tgt["type"]:
                cross_type_pairs.append((src["id"], tgt["id"]))

    print(f"\n{'='*100}")
    print("CROSS-TYPE PARTIAL TRANSFER SUMMARY (R²)")
    print(f"{'='*100}")

    for model_name in MODELS:
        model = MODELS[model_name]
        print(f"\n--- {model_name} ---")
        print(f"  {'Source → Target':<22s}  {'Raw':>8s}  {'Partial':>8s}  {'Self-fit':>8s}  {'Recovery':>8s}")
        print(f"  {'-'*22}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

        for src_id, tgt_id in cross_type_pairs:
            src = hold_by_id[src_id]
            tgt = hold_by_id[tgt_id]
            src_label = hold_labels[src_id]
            tgt_label = hold_labels[tgt_id]

            source_params = fitted[(model_name, src_id)]
            target_params = fitted[(model_name, tgt_id)]

            raw_pred = model["predict"](tgt["t"], source_params)
            raw_r2 = compute_r2(tgt["spo2"], raw_pred)

            partial_params = fit_partial(model_name, source_params, tgt)
            partial_pred = model["predict"](tgt["t"], partial_params)
            partial_r2 = compute_r2(tgt["spo2"], partial_pred)

            self_pred = model["predict"](tgt["t"], target_params)
            self_r2 = compute_r2(tgt["spo2"], self_pred)

            # Recovery: what fraction of the gap from raw to self-fit does partial close?
            gap = self_r2 - raw_r2
            recovery = (partial_r2 - raw_r2) / gap * 100 if abs(gap) > 1e-6 else 0.0

            print(f"  {src_label+' → '+tgt_label:<22s}  {raw_r2:>8.4f}  {partial_r2:>8.4f}  "
                  f"{self_r2:>8.4f}  {recovery:>7.1f}%")

    # ── Step 4: Plot select cross-type partial transfers ────────────────────

    # Pick representative pairs: FL#6 → each other type, best of each
    plot_pairs = [
        (6, 2, "FL#6 → FRC#2"),
        (6, 5, "FL#6 → FRC#5"),
        (6, 3, "FL#6 → RV#3"),
        (6, 4, "FL#6 → RV#4"),
        (5, 6, "FRC#5 → FL#6"),
        (3, 6, "RV#3 → FL#6"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    colors = {"Production": "#1f77b4", "CO2-Bohr": "#ff7f0e"}

    for idx, (src_id, tgt_id, title) in enumerate(plot_pairs):
        ax = axes[idx // 2, idx % 2]
        tgt = hold_by_id[tgt_id]
        ax.plot(tgt["t"], tgt["spo2"], "k.", ms=2, alpha=0.5, label="Observed")

        for model_name in MODELS:
            model = MODELS[model_name]
            source_params = fitted[(model_name, src_id)]

            # Raw transfer (dashed)
            raw_pred = model["predict"](tgt["t"], source_params)
            raw_r2 = compute_r2(tgt["spo2"], raw_pred)

            # Partial transfer (solid)
            partial_params = fit_partial(model_name, source_params, tgt)
            partial_pred = model["predict"](tgt["t"], partial_params)
            partial_r2 = compute_r2(tgt["spo2"], partial_pred)

            ax.plot(tgt["t"], raw_pred, color=colors[model_name], lw=1, alpha=0.4,
                    ls="--", label=f"{model_name} raw (R²={raw_r2:.3f})")
            ax.plot(tgt["t"], partial_pred, color=colors[model_name], lw=1.5, alpha=0.8,
                    label=f"{model_name} partial (R²={partial_r2:.3f})")

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("SpO2 (%)")
        ax.set_ylim(30, 105)
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = Path(__file__).resolve().parent / "cross_predict_partial.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out}")


if __name__ == "__main__":
    main()
