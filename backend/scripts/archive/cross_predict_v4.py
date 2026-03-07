"""
v4 cross-prediction: fit on one hold, predict all others.

Tests whether model parameters generalise with the delay observation model.
Adds CO2-Bohr+Delay to the cross-prediction matrix from v3.

Models:
  - CO2-Bohr       — 7 params, r_offset only
  - CO2-Bohr+Delay — 8 params, pure time delay + clip
  - Richards       — 5 params, descriptive sigmoid

Usage:
    cd backend && uv run python -u scripts/cross_predict_v4.py
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


# ── Model functions ──────────────────────────────────────────────────────────


def predict_co2bohr(t, params):
    """CO2-Bohr: 7 params."""
    pao2_0, pvo2, tau_washout, gamma, paco2_0, k_co2, r_offset = params
    pao2 = pvo2 + (pao2_0 - pvo2) * np.exp(-t / max(tau_washout, 0.01))
    paco2 = paco2_0 + k_co2 * t
    p50 = P50_BASE + 0.48 * (paco2 - 40.0)
    pao2_v = pao2 * (P50_BASE / np.maximum(p50, 0.01))
    pao2_adj = P50_BASE * (np.maximum(pao2_v, 0.01) / P50_BASE) ** gamma
    x = np.maximum(pao2_adj, 0.01)
    sa = 100.0 / (1.0 + 23400.0 / (x**3 + 150.0 * x))
    return np.clip(sa + r_offset, 0.0, 100.0)


def predict_co2bohr_delay(t, params):
    """CO2-Bohr+Delay: 8 params, pure time delay."""
    pao2_0, pvo2, tau_washout, gamma, paco2_0, k_co2, r_offset, d = params
    pao2 = pvo2 + (pao2_0 - pvo2) * np.exp(-t / max(tau_washout, 0.01))
    paco2 = paco2_0 + k_co2 * t
    p50 = P50_BASE + 0.48 * (paco2 - 40.0)
    pao2_v = pao2 * (P50_BASE / np.maximum(p50, 0.01))
    pao2_adj = P50_BASE * (np.maximum(pao2_v, 0.01) / P50_BASE) ** gamma
    x = np.maximum(pao2_adj, 0.01)
    sa = 100.0 / (1.0 + 23400.0 / (x**3 + 150.0 * x))
    # Observation: pure delay
    sa_delayed = np.interp(t - d, t, sa, left=sa[0])
    return np.clip(sa_delayed + r_offset, 0.0, 100.0)


def predict_richards(t, params):
    """Richards sigmoid: 5 params."""
    s_max, s_min, t50, k, nu = params
    z = np.clip((t - t50) / max(k, 0.01), -500, 500)
    base = 1.0 + nu * np.exp(z)
    return np.clip(
        s_min + (s_max - s_min) / np.power(np.maximum(base, 1e-10), 1.0 / nu), 0.0, 100.0
    )


# ── Bounds ──────────────────────────────────────────────────────────────────

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
RICHARDS_BOUNDS = {
    "FL": [(96, 101), (0, 96), (50, 500), (5, 80), (0.1, 10)],
    "FRC": [(96, 101), (0, 96), (20, 300), (3, 60), (0.1, 10)],
    "RV": [(96, 101), (0, 96), (10, 250), (3, 60), (0.1, 10)],
}

MODELS = {
    "CO2-Bohr": {
        "predict": predict_co2bohr,
        "bounds": CO2BOHR_BOUNDS,
        "names": ["pao2_0", "pvo2", "tau_washout", "gamma", "paco2_0", "k_co2", "r_offset"],
    },
    "CO2-Bohr+Delay": {
        "predict": predict_co2bohr_delay,
        "bounds": CO2BOHR_DELAY_BOUNDS,
        "names": [
            "pao2_0", "pvo2", "tau_washout", "gamma", "paco2_0", "k_co2", "r_offset", "d",
        ],
    },
    "Richards": {
        "predict": predict_richards,
        "bounds": RICHARDS_BOUNDS,
        "names": ["s_max", "s_min", "t50", "k", "nu"],
    },
}


# ── Fitting + cross-prediction ──────────────────────────────────────────────


def compute_r2(obs, pred):
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def compute_rmse(obs, pred):
    return float(np.sqrt(np.mean((obs - pred) ** 2)))


def fit_model(model_name, hold):
    """Fit a model to a hold, return fitted params."""
    model = MODELS[model_name]
    bounds = model["bounds"][hold["type"]]
    predict = model["predict"]
    t, spo2 = hold["t"], hold["spo2"]

    if model_name == "Richards":
        def objective(arr):
            return np.sum((spo2 - predict(t, arr)) ** 2)
    else:
        def objective(arr):
            pred = predict(t, arr)
            w = np.where(spo2 < 95, 3.0, 1.0)
            return np.sum(w * (spo2 - pred) ** 2)

    result = differential_evolution(
        objective, bounds, maxiter=3000, seed=42, tol=1e-10,
        polish=True, popsize=40, mutation=(0.5, 1.5), recombination=0.9,
    )
    return result.x


def cross_predict(model_name, params, hold):
    """Predict a hold using pre-fitted params. Return R², RMSE."""
    predict = MODELS[model_name]["predict"]
    pred = predict(hold["t"], params)
    r2 = compute_r2(hold["spo2"], pred)
    rmse = compute_rmse(hold["spo2"], pred)
    return r2, rmse, pred


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    holds = load_all_holds()
    hold_labels = {h["id"]: f"{h['type']} #{h['id']}" for h in holds}

    print("Loaded holds:")
    for h in holds:
        print(f"  {hold_labels[h['id']]}: {len(h['t'])} pts, "
              f"SpO2 {h['spo2'].min():.0f}-{h['spo2'].max():.0f}%", flush=True)

    # Fit all models on all holds
    fitted = {}
    for model_name in MODELS:
        for hold in holds:
            label = hold_labels[hold["id"]]
            print(f"\nFitting {model_name} on {label}...", end="", flush=True)
            params = fit_model(model_name, hold)
            fitted[(model_name, hold["id"])] = params
            r2, rmse, _ = cross_predict(model_name, params, hold)
            print(f" R²={r2:.4f}, RMSE={rmse:.2f}", flush=True)

    # Cross-prediction matrix (R²)
    print(f"\n{'='*90}")
    print("CROSS-PREDICTION MATRIX (R²)")
    print(f"{'='*90}")

    for model_name in MODELS:
        print(f"\n--- {model_name} ---")
        col_labels = [hold_labels[h["id"]] for h in holds]
        header = f"  {'Train \\ Predict':<16s}" + "".join(f" {l:>10s}" for l in col_labels)
        print(header)
        print(f"  {'-'*16}" + "".join(f" {'-'*10}" for _ in holds))

        for train_hold in holds:
            params = fitted[(model_name, train_hold["id"])]
            row = f"  {hold_labels[train_hold['id']]:<16s}"
            for pred_hold in holds:
                r2, _, _ = cross_predict(model_name, params, pred_hold)
                if train_hold["id"] == pred_hold["id"]:
                    row += f" {'['+f'{r2:.4f}'+']':>10s}"
                else:
                    row += f" {r2:>10.4f}"
            print(row)

    # RMSE version
    print(f"\n{'='*90}")
    print("CROSS-PREDICTION MATRIX (RMSE)")
    print(f"{'='*90}")

    for model_name in MODELS:
        print(f"\n--- {model_name} ---")
        col_labels = [hold_labels[h["id"]] for h in holds]
        header = f"  {'Train \\ Predict':<16s}" + "".join(f" {l:>10s}" for l in col_labels)
        print(header)
        print(f"  {'-'*16}" + "".join(f" {'-'*10}" for _ in holds))

        for train_hold in holds:
            params = fitted[(model_name, train_hold["id"])]
            row = f"  {hold_labels[train_hold['id']]:<16s}"
            for pred_hold in holds:
                _, rmse, _ = cross_predict(model_name, params, pred_hold)
                if train_hold["id"] == pred_hold["id"]:
                    row += f" {'['+f'{rmse:.2f}'+']':>10s}"
                else:
                    row += f" {rmse:>10.2f}"
            print(row)

    # Within-type transfer summary
    pairs = [
        ("FL", 6, 1, "FL#6 -> FL#1"),
        ("FL", 1, 6, "FL#1 -> FL#6"),
        ("FRC", 5, 2, "FRC#5 -> FRC#2"),
        ("FRC", 2, 5, "FRC#2 -> FRC#5"),
        ("RV", 4, 3, "RV#4 -> RV#3"),
        ("RV", 3, 4, "RV#3 -> RV#4"),
    ]

    print(f"\n{'='*90}")
    print("WITHIN-TYPE TRANSFER SUMMARY")
    print(f"{'='*90}")
    header = f"  {'Pair':<18s}" + "".join(f" {m:>16s}" for m in MODELS)
    print(header)
    print(f"  {'-'*18}" + "".join(f" {'-'*16}" for _ in MODELS))

    hold_by_id = {h["id"]: h for h in holds}
    for _, train_id, pred_id, label in pairs:
        row = f"  {label:<18s}"
        for model_name in MODELS:
            params = fitted[(model_name, train_id)]
            r2, _, _ = cross_predict(model_name, params, hold_by_id[pred_id])
            row += f" {r2:>16.4f}"
        print(row)

    # Cross-type transfer (trained on FL#6)
    print(f"\n{'='*90}")
    print("CROSS-TYPE TRANSFER (trained on FL #6)")
    print(f"{'='*90}")
    header = f"  {'Predict':<16s}" + "".join(f" {m:>16s}" for m in MODELS)
    print(header)
    print(f"  {'-'*16}" + "".join(f" {'-'*16}" for _ in MODELS))

    for pred_hold in holds:
        row = f"  {hold_labels[pred_hold['id']]:<16s}"
        for model_name in MODELS:
            params = fitted[(model_name, 6)]
            r2, _, _ = cross_predict(model_name, params, pred_hold)
            marker = " *" if pred_hold["id"] == 6 else ""
            row += f" {r2:>16.4f}{marker}"
        print(row)

    # Plot: within-type cross-predictions
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    colors = {"CO2-Bohr": "#1f77b4", "CO2-Bohr+Delay": "#ff7f0e", "Richards": "#2ca02c"}

    for row_idx, (_, train_id, pred_id, label) in enumerate(pairs[:3]):
        pred_hold = hold_by_id[pred_id]
        train_hold = hold_by_id[train_id]

        # Left: prediction on unseen hold
        ax = axes[row_idx, 0]
        ax.plot(pred_hold["t"], pred_hold["spo2"], "k.", ms=2, alpha=0.5, label="Observed")
        for model_name in MODELS:
            params = fitted[(model_name, train_id)]
            r2, rmse, pred = cross_predict(model_name, params, pred_hold)
            ax.plot(
                pred_hold["t"], pred, color=colors[model_name], lw=1.5, alpha=0.8,
                label=f"{model_name} (R²={r2:.3f})",
            )
        ax.set_title(
            f"Predict {hold_labels[pred_id]} (trained on {hold_labels[train_id]})",
            fontsize=11, fontweight="bold",
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("SpO2 (%)")
        ax.set_ylim(30, 105)
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(True, alpha=0.3)

        # Right: self-fit on training hold for reference
        ax = axes[row_idx, 1]
        ax.plot(train_hold["t"], train_hold["spo2"], "k.", ms=2, alpha=0.5, label="Observed")
        for model_name in MODELS:
            params = fitted[(model_name, train_id)]
            r2, rmse, pred = cross_predict(model_name, params, train_hold)
            ax.plot(
                train_hold["t"], pred, color=colors[model_name], lw=1.5, alpha=0.8,
                label=f"{model_name} (R²={r2:.3f})",
            )
        ax.set_title(
            f"Self-fit on {hold_labels[train_id]}", fontsize=11, fontweight="bold",
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("SpO2 (%)")
        ax.set_ylim(30, 105)
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = Path(__file__).resolve().parent / "cross_predict_v4.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out}")


if __name__ == "__main__":
    main()
