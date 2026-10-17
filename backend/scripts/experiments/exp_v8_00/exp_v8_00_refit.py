"""v8 Experiment 00: refit v7.08 as-is on every session with SpO2 and record parameter drift.

Nothing about the model changes here. The v7.08 Stage A and Stage B fit functions are
imported from `exp_v7_08_ship.py` and run on hold dicts built by `nadir_export.py`. One fit
per group; a group is a session, or the pool of all Nadir sessions. The output is the drift
table a v8 has to beat, not a new model.

Usage:
    cd backend && uv run python -u scripts/experiments/exp_v8_00/exp_v8_00_refit.py \
        --group 2026-09-04 path/to/nadir-export.json "../Saturday, February 21, 2026.csv"
    --group all   runs every session, one after the other
    --group pool  fits every app-recorded hold under one set of globals (hours at v7's
                  optimiser settings with more than a handful of holds; not part of "all")
    --quick       caps DE at 5 generations to check the plumbing
    --plot        skips fitting and draws results/<group>.png from the saved results/<group>.json
    --summary     skips fitting and prints the drift table from every saved results/*.json
    --rescore     adds apnea-window RMSE to every saved results/*.json (needs the data files)
    --transfer G  freezes the four globals (tau_0, p, k_co2, gamma) at results/G.json's values
                  and fits only the per-hold parameters; writes results/transfer-<group>.json.
                  This is the spec's own "Priority 2" test: do one session's globals carry?
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from nadir_export import load, qc  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "ship", HERE.parent / "exp_v7_08" / "exp_v7_08_ship.py"
)
ship = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ship)


def fit_group(name, holds, globals_from=None):
    """One v7.08 fit. With `globals_from` (a saved result), tau_0, p, k_co2 and gamma are
    pinned to that result's values and only the per-hold parameters are fitted."""
    fit_holds = [h for h in holds if qc(h) is None]
    excluded = {h["id"]: qc(h) for h in holds if qc(h) is not None}
    print(
        f"\n{'=' * 100}\n{name}: {len(fit_holds)} holds fitted, "
        f"{len(excluded)} excluded {excluded}\n{'=' * 100}"
    )
    if not fit_holds:
        return None

    nadir_info = {h["id"]: ship.compute_nadir_info(h) for h in fit_holds}
    s_base = {h["id"]: float(np.median(h["spo2"][h["t"] <= 20])) for h in fit_holds}

    stage_b_kwargs = {}
    if globals_from:
        ga, gb = globals_from["stage_a"], globals_from["stage_b"]
        # run_stage_a reads these module constants at call time; a hair-width interval pins them.
        ship.TAU0_BOUNDS = (ga["tau_0"] - 1e-4, ga["tau_0"] + 1e-4)
        ship.P_BOUNDS = (ga["p"] - 1e-4, ga["p"] + 1e-4)
        stage_b_kwargs = {"k_co2_fixed": gb["k_co2"], "gamma_fixed": gb["gamma"]}
        print(
            f"  globals frozen from {globals_from['group']}: tau_0={ga['tau_0']:.2f} "
            f"p={ga['p']:.2f} k_co2={gb['k_co2']:.4f} gamma={gb['gamma']:.3f}"
        )

    flat_a, conv_a, loss_a = ship.run_stage_a(fit_holds, nadir_info, s_base, seed=42)
    eval_a, _ = ship.evaluate_stage_a(flat_a, fit_holds, nadir_info, s_base)
    frozen = ship.extract_frozen_sensor(flat_a, fit_holds)

    flat_b, conv_b, loss_b = ship.run_stage_b(
        fit_holds, nadir_info, frozen, s_base, **stage_b_kwargs
    )
    eval_b = ship.evaluate_stage_b(flat_b, fit_holds, nadir_info, frozen, s_base)

    n = len(fit_holds)
    out = {
        "group": name,
        "globals_from": globals_from["group"] if globals_from else None,
        "n_holds": n,
        "n_points": int(sum(len(h["t"]) for h in fit_holds)),
        "excluded": excluded,
        "stage_a": {
            "converged": bool(conv_a),
            "loss": float(loss_a),
            "tau_0": float(frozen["tau_0"]),
            "p": float(frozen["p"]),
            "tau_0_at_bound": bool(ship.is_at_bound(frozen["tau_0"], *ship.TAU0_BOUNDS)),
            "p_at_bound": bool(ship.is_at_bound(frozen["p"], *ship.P_BOUNDS)),
        },
        "stage_b": {
            "converged": bool(conv_b),
            "loss": float(loss_b),
            "k_co2": float(flat_b[0]),
            "gamma": float(flat_b[1]),
            "k_co2_at_bound": bool(ship.is_at_bound(flat_b[0], *ship.K_CO2_BOUNDS)),
            "gamma_at_bound": bool(ship.is_at_bound(flat_b[1], *ship.GAMMA_BOUNDS)),
        },
        "holds": [],
    }
    for i, h in enumerate(fit_holds):
        ea, eb = eval_a[i], eval_b[i]
        n_apnea = len(h["t_apnea"])  # h["t"] starts with the apnea grid
        lo_tw, hi_tw = ship.PERHOLD_BOUNDS[h["type"]][0]
        lo_pc, hi_pc = ship.PERHOLD_BOUNDS[h["type"]][1]
        s_min, v_up = flat_a[3 + n + 2 * i : 3 + n + 2 * i + 2]
        out["holds"].append(
            {
                "id": h["id"],
                "type": h["type"],
                "duration": h["meta"]["duration"],
                "nadir_obs": float(h["spo2"].min()),
                "s_base": s_base[h["id"]],
                "delta": float(ea["delta"]),
                "eff_lag": float(ea["effective_lag"]),
                "S_min": float(s_min),
                "v_up": float(v_up),
                "r2a_stage_a": ea["r2_apnea"],
                "r2n_stage_a": ea["r2_nadir"],
                "tau_washout": float(eb["tau_washout"]),
                "tau_washout_at_bound": bool(ship.is_at_bound(eb["tau_washout"], lo_tw, hi_tw)),
                "paco2_0": float(eb["paco2_0"]),
                "paco2_0_at_bound": bool(ship.is_at_bound(eb["paco2_0"], lo_pc, hi_pc)),
                "r2a_stage_b": eb["r2_apnea"],
                "r2n_stage_b": eb["r2_nadir"],
                "rmse_apnea_stage_b": float(
                    ship.compute_rmse(h["spo2_apnea"], eb["pred_full"][:n_apnea])
                ),
                "nadir_err_stage_b": float(eb["nadir_err"]),
                "nadir_spo2_err_stage_b": float(eb["nadir_spo2_err"]),
            }
        )
    return out


def rebuild(result, holds):
    """Re-evaluate a saved result's Stage A and Stage B predictions on the holds."""
    by_id = {h["id"]: h for h in holds}
    fit_holds = [by_id[r["id"]] for r in result["holds"]]
    a, b = result["stage_a"], result["stage_b"]
    flat_a = np.array(
        [a["tau_0"], 1.0, a["p"]]
        + [r["delta"] for r in result["holds"]]
        + [v for r in result["holds"] for v in (r["S_min"], r["v_up"])]
    )
    flat_b = np.array(
        [b["k_co2"], b["gamma"]]
        + [r["delta"] for r in result["holds"]]
        + [v for r in result["holds"] for v in (r["tau_washout"], r["paco2_0"])]
    )
    nadir_info = {h["id"]: ship.compute_nadir_info(h) for h in fit_holds}
    s_base = {r["id"]: r["s_base"] for r in result["holds"]}
    eval_a, _ = ship.evaluate_stage_a(flat_a, fit_holds, nadir_info, s_base)
    eval_b = ship.evaluate_stage_b(
        flat_b, fit_holds, nadir_info, ship.extract_frozen_sensor(flat_a, fit_holds), s_base
    )
    return fit_holds, eval_a, eval_b


def rescore(result, holds):
    """Add the apnea-window RMSE to a saved result's holds."""
    fit_holds, _, eval_b = rebuild(result, holds)
    for r, h, eb in zip(result["holds"], fit_holds, eval_b, strict=True):
        n_apnea = len(h["t_apnea"])
        r["rmse_apnea_stage_b"] = float(
            ship.compute_rmse(h["spo2_apnea"], eb["pred_full"][:n_apnea])
        )
    return result


def plot_group(result, holds, out_path):
    """Observed trace with the Stage A (sensor) and Stage B (physiology) predictions per hold."""
    fit_holds, eval_a, eval_b = rebuild(result, holds)
    n = len(fit_holds)
    a, b = result["stage_a"], result["stage_b"]
    fig, axes = plt.subplots(n, 1, figsize=(10, 3.2 * n), squeeze=False)
    for ax, h, ea, eb in zip(axes[:, 0], fit_holds, eval_a, eval_b, strict=False):
        ax.plot(h["t"], h["spo2"], "k.", ms=3, alpha=0.6, label="observed")
        ax.plot(
            h["t"],
            ea["pred_full"],
            color="#1f77b4",
            lw=1.5,
            label=f"Stage A  R2a={ea['r2_apnea']:.2f}  lag={ea['effective_lag']:.0f}s",
        )
        ax.plot(
            h["t"],
            eb["pred_full"],
            color="#2ca02c",
            lw=2,
            alpha=0.8,
            label=f"Stage B  R2a={eb['r2_apnea']:.2f}  tau_w={eb['tau_washout']:.0f}s",
        )
        ax.axvline(h["t_end"], color="red", ls="--", alpha=0.5)
        ax.set_ylabel(f"{h['id']}\nSpO2 (%)", fontsize=8)
        ax.set_ylim(min(55, h["spo2"].min() - 5), 102)
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel("Time since hold start (s)")
    fig.suptitle(
        f"v7.08 refit on {result['group']}"
        + (
            f" with globals frozen from {result['globals_from']}"
            if result.get("globals_from")
            else ""
        )
        + f": tau_0={a['tau_0']:.1f}s p={a['p']:.2f} k_co2={b['k_co2']:.4f} gamma={b['gamma']:.2f}",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  plot: {out_path}")


def summarise(results):
    print(
        f"\n{'group':<22}{'holds':>5} {'pts':>5} {'tau_0':>6} {'p':>5} {'k_co2':>7} "
        f"{'gamma':>6} {'R2a':>6} {'RMSEa':>6}  bound hits"
    )
    print(
        f"{'v7.08 (published)':<22}{4:>5} {800:>5} {17.68:>6.2f} {3.39:>5.2f} {0.1385:>7.4f} "
        f"{1.863:>6.3f} {0.972:>6.3f} {'-':>6}  tau_w FRC x2, paco2_0 RV"
    )
    for r in results:
        if r is None:
            continue
        a, b = r["stage_a"], r["stage_b"]
        hits = [k for k, v in {**a, **b}.items() if k.endswith("_at_bound") and v]
        hits += [f"{h['id'].split('-')[-1]}:tau_w" for h in r["holds"] if h["tau_washout_at_bound"]]
        hits += [f"{h['id'].split('-')[-1]}:paco2_0" for h in r["holds"] if h["paco2_0_at_bound"]]
        r2a = np.mean([h["r2a_stage_b"] for h in r["holds"]])
        rmse = [h["rmse_apnea_stage_b"] for h in r["holds"] if "rmse_apnea_stage_b" in h]
        rmse_str = f"{np.mean(rmse):>6.2f}" if rmse else f"{'-':>6}"
        if r.get("globals_from"):
            hits = [f"globals from {r['globals_from']}"] + [x for x in hits if ":" in x]
        print(
            f"{r['group']:<22}{r['n_holds']:>5} {r['n_points']:>5} {a['tau_0']:>6.2f} "
            f"{a['p']:>5.2f} "
            f"{b['k_co2']:>7.4f} {b['gamma']:>6.3f} {r2a:>6.3f} {rmse_str}  "
            f"{', '.join(hits) or '-'}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument(
        "--group", default="all", help="session date, 'pool' (all Nadir holds), or 'all'"
    )
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--transfer", metavar="GROUP", help="freeze globals at results/GROUP.json")
    ap.add_argument("--rescore", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    if args.summary:
        summarise([json.loads(p.read_text()) for p in sorted(out_dir.glob("*.json"))])
        return

    if args.quick:
        de = ship.differential_evolution
        ship.differential_evolution = lambda *a, **k: de(*a, **{**k, "maxiter": 5, "popsize": 5})

    holds = [h for f in args.files for h in load(f)]
    sessions = sorted({h["meta"]["session"] for h in holds})
    groups = {s: [h for h in holds if h["meta"]["session"] == s] for s in sessions}
    groups["pool-nadir"] = [h for h in holds if h["meta"]["source"] == "nadir"]
    wanted = (
        sessions if args.group == "all" else [args.group if args.group != "pool" else "pool-nadir"]
    )

    if args.rescore:
        for path in sorted(out_dir.glob("*.json")):
            result = json.loads(path.read_text())
            if result["group"] in groups:
                path.write_text(json.dumps(rescore(result, groups[result["group"]]), indent=1))
                print(f"  rescored {path.name}")
        return
    globals_from = (
        json.loads((out_dir / f"{args.transfer}.json").read_text()) if args.transfer else None
    )
    prefix = "transfer-" if args.transfer else ""
    results = []
    for g in wanted:
        if args.plot:
            path = out_dir / f"{prefix}{g}.json"
            if path.exists():
                plot_group(json.loads(path.read_text()), groups[g], out_dir / f"{prefix}{g}.png")
            continue
        r = fit_group(g, groups[g], globals_from)
        results.append(r)
        if r is not None:
            (out_dir / f"{prefix}{g}.json").write_text(json.dumps(r, indent=1))
            plot_group(r, groups[g], out_dir / f"{prefix}{g}.png")
    if not args.plot:
        summarise(results)


if __name__ == "__main__":
    main()
