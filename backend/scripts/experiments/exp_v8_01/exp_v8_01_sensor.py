"""v8 Experiment 01: what the two oximeters do to a hold, without ground truth.

Four questions, each answered with the smallest thing that can answer it:

  A. Trace statistics by device and step kind: integer runs, step sizes, and "freeze then
     jump" events (a flat run followed by a large step), with how often they coincide with
     contact loss or a sample gap.
  B. A freeze mask: the rule a later fit would use to treat frozen samples as missing, and
     how many hold-seconds it removes per device.
  C. Model-free lag measures per hold (nadir delay, onset of the recovery rise) by device,
     within RV where both devices have tables, with bootstrap intervals; the dependence of
     nadir delay on the end-of-hold slope, with each session left out in turn; heart rate
     as a covariate on the onset, reported and not modelled.
  D. One small post-hold sensor fit per device: the descent continues for a combined
     transit-plus-delay T, then rises exponentially (per-hold time constant); the sensor is
     a first-order low-pass tau_s. T and tau_s are shared per device, bootstrapped over
     holds, and tau_s is profiled because it trades against the latent rise.

Usage:
    cd backend && uv run python scripts/experiments/exp_v8_01/exp_v8_01_sensor.py \
        /path/to/nadir-export.json "../Saturday, February 21, 2026.csv"
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "exp_v8_00"))
from nadir_export import load, qc  # noqa: E402

OUT = HERE / "results"
RNG = np.random.default_rng(42)


def device(meta):
    return "pm100" if meta["source"] == "nadir" else "masimo"


# ---------------------------------------------------------------- A. trace statistics


def runs(t, s):
    """(start index, end index, step after the run) for every flat run; the last run has no step."""
    out, j = [], 0
    while j < len(s):
        k = j
        while k + 1 < len(s) and s[k + 1] == s[j]:
            k += 1
        out.append((j, k, s[k + 1] - s[k] if k + 1 < len(s) else None))
        j = k + 1
    return out


def freeze_events(t, s, min_run, min_jump):
    """Flat runs of at least `min_run` seconds followed by a step of at least `min_jump`."""
    return [
        (j, k, step)
        for j, k, step in runs(t, s)
        if step is not None and t[k] - t[j] + 1 >= min_run and abs(step) >= min_jump
    ]


def trace_statistics(export):
    rows = {}
    for session in export["sessions"]:
        dev = "masimo" if {"imported", "stamina"} <= set(session.get("tags", [])) else "pm100"
        for step in session["steps"]:
            vitals = step.get("vitals", [])
            good = [
                v for v in vitals if v.get("spo2") is not None and v.get("contactDetected", True)
            ]
            if len(good) < 30:
                continue
            t = np.array([round(v["at"]) for v in good], float)
            s = np.array([v["spo2"] for v in good], float)
            kind = step.get("kindName") if step.get("kindName") in ("hold", "activity") else "other"
            r = rows.setdefault(
                (dev, kind), {"samples": 0, "runs": [], "steps": [], "ev83": [], "ev52": []}
            )
            r["samples"] += len(s)
            rr = runs(t, s)
            r["runs"] += [t[k] - t[j] + 1 for j, k, _ in rr[:-1]]
            r["steps"] += [abs(st) for _, _, st in rr[:-1] if st]
            lost = {round(v["at"]) for v in vitals if not v.get("contactDetected", True)}
            present = set(t)
            for key, (mr, mj) in {"ev83": (8, 3), "ev52": (5, 2)}.items():
                for j, k, step_ in freeze_events(t, s, mr, mj):
                    window = range(int(t[j]), int(t[k]) + 4)
                    lost_near = any(x in lost for x in window)
                    gap_near = any(x not in present and x not in lost for x in window)
                    # Was the trace moving in the 30 s before the run? A freeze mid-descent
                    # is a different thing from a plateau on a flat signal.
                    w = (t >= t[j] - 30) & (t < t[j])
                    moving = w.sum() > 5 and abs(s[j] - s[w][0]) / (t[j] - t[w][0]) >= 0.1
                    r[key].append((t[k] - t[j] + 1, step_, lost_near or gap_near, moving))
    return rows


def print_trace_statistics(rows, out):
    out.append("A. TRACE STATISTICS BY DEVICE AND STEP KIND")
    out.append("")
    out.append(
        f"  {'device':<8}{'kind':<10}{'samples':>8}  {'run median':>10}{'run p90':>8}"
        f"{'runs>=10s':>10}  {'|step| median':>13}{'max':>5}  {'freeze 8s/3pt':>14}"
        f"{'per 1000s':>10}{'near loss':>10}{'moving':>8}"
        f"  {'freeze 5s/2pt':>14}{'per 1000s':>10}{'near loss':>10}{'moving':>8}"
    )
    for (dev, kind), r in sorted(rows.items()):
        run_len, step_size = np.array(r["runs"]), np.array(r["steps"])
        e83, e52 = r["ev83"], r["ev52"]
        out.append(
            f"  {dev:<8}{kind:<10}{r['samples']:>8}  {np.median(run_len):>10.0f}"
            f"{np.percentile(run_len, 90):>8.0f}"
            f"{np.mean(run_len >= 10):>10.2f}  {np.median(step_size):>13.0f}{step_size.max():>5.0f}"
            f"  {len(e83):>14}{1000 * len(e83) / r['samples']:>10.1f}"
            f"{sum(1 for e in e83 if e[2]):>10}{sum(1 for e in e83 if e[3]):>8}"
            f"  {len(e52):>14}{1000 * len(e52) / r['samples']:>10.1f}"
            f"{sum(1 for e in e52 if e[2]):>10}{sum(1 for e in e52 if e[3]):>8}"
        )
    out.append("")
    out.append(
        "  near loss = the run or the three seconds after it contain a no-contact sample or a"
        " missing second;\n  moving = the trace fell or rose 0.1 %/s or more over the 30 s before"
        " the run."
    )
    out.append("")


# ---------------------------------------------------------------- B. freeze mask


def freeze_mask(t, s, min_run=8, min_jump=3):
    """Samples a later fit should treat as missing: the flat run before a large jump."""
    mask = np.zeros(len(s), dtype=bool)
    for j, k, _ in freeze_events(t, s, min_run, min_jump):
        mask[j : k + 1] = True
    return mask


def freeze_summary(holds, out):
    out.append("B. FREEZE MASK: HOLD-SECONDS A FIT WOULD DROP")
    out.append("")
    for dev in ("masimo", "pm100"):
        hs = [h for h in holds if device(h["meta"]) == dev]
        if not hs:
            out.append(f"  {dev:<8} no QC-passing holds loaded")
            continue
        total = sum(len(h["t"]) for h in hs)
        dropped = sum(int(freeze_mask(h["t"], h["spo2"]).sum()) for h in hs)
        affected = sum(1 for h in hs if freeze_mask(h["t"], h["spo2"]).any())
        out.append(
            f"  {dev:<8} {len(hs):>3} QC-passing holds, {total:>6} s of trace, "
            f"{dropped:>4} s masked"
            f" ({100 * dropped / total:.1f} %), {affected} holds touched"
        )
    out.append("")
    out.append(
        "  Rule: a run of 8 s or more followed by a step of 3 points or more; the run is masked."
    )
    out.append("")


# ---------------------------------------------------------------- C. lag measures


def lag_measures(h):
    t, s, hr, te = h["t"], h["spo2"], h["hr"], h["t_end"]
    after, ta = s[t > te], t[t > te]
    running_min = np.minimum.accumulate(after)
    rise = np.where(after - running_min >= 2)[0]
    before = (t > te - 30) & (t <= te)
    return {
        "id": h["id"],
        "device": device(h["meta"]),
        "type": h["type"],
        "session": h["meta"]["session"],
        "duration": h["meta"]["duration"],
        "nadir": float(s.min()),
        "nadir_delay": float(t[np.argmin(s)] - te),
        "onset": float(ta[rise[0]] - te) if len(rise) else None,
        "end_slope": float((s[before][-1] - s[before][0]) / 30),
        "hr_end": float(np.nanmean(hr[(t > te - 10) & (t <= te)])),
        "recovery_s": float(len(h["t_recovery"])),
    }


def boot_median(x, n=2000):
    x = np.asarray(x, float)
    meds = [np.median(RNG.choice(x, len(x))) for _ in range(n)]
    return np.median(x), np.percentile(meds, 2.5), np.percentile(meds, 97.5)


def lag_section(measures, out):
    out.append("C. MODEL-FREE LAG MEASURES (holds with at least 15 s of trace after the end)")
    out.append("")
    out.append(
        f"  {'hold':<18}{'device':<8}{'type':<5}{'dur':>5}{'nadir':>6}{'nadir delay':>12}"
        f"{'onset':>7}"
        f"{'end slope':>11}{'HR end':>8}{'rec s':>6}"
    )
    for m in measures:
        onset = f"{m['onset']:7.0f}" if m["onset"] is not None else f"{'-':>7}"
        out.append(
            f"  {m['id']:<18}{m['device']:<8}{m['type']:<5}{m['duration']:>5.0f}{m['nadir']:>6.0f}"
            f"{m['nadir_delay']:>12.0f}{onset}{m['end_slope']:>11.2f}{m['hr_end']:>8.0f}"
            f"{m['recovery_s']:>6.0f}"
        )
    out.append("")
    out.append("  Medians with 95 % bootstrap intervals over holds:")
    out.append("")
    out.append(f"  {'group':<22}{'n':>3}  {'nadir delay (s)':<24}{'onset (s)':<24}")
    for label, sel in [
        ("masimo, all", lambda m: m["device"] == "masimo"),
        ("pm100, all", lambda m: m["device"] == "pm100"),
        ("masimo, FL", lambda m: m["device"] == "masimo" and m["type"] == "FL"),
        ("pm100, FL", lambda m: m["device"] == "pm100" and m["type"] == "FL"),
        ("masimo, RV", lambda m: m["device"] == "masimo" and m["type"] == "RV"),
        ("pm100, RV", lambda m: m["device"] == "pm100" and m["type"] == "RV"),
        (
            "pm100, RV, 90-140 s",
            lambda m: m["device"] == "pm100" and m["type"] == "RV" and 90 <= m["duration"] <= 140,
        ),
        (
            "masimo, RV, 90-140 s",
            lambda m: m["device"] == "masimo" and m["type"] == "RV" and 90 <= m["duration"] <= 140,
        ),
    ]:
        g = [m for m in measures if sel(m)]
        if len(g) < 3:
            continue
        nd = boot_median([m["nadir_delay"] for m in g])
        on = [m["onset"] for m in g if m["onset"] is not None]
        nd_s = f"{nd[0]:4.0f} [{nd[1]:3.0f}, {nd[2]:3.0f}]"
        if len(on) >= 3:
            om = boot_median(on)
            on_s = f"{om[0]:4.0f} [{om[1]:3.0f}, {om[2]:3.0f}] n={len(on)}"
        else:
            on_s = "-"
        out.append(f"  {label:<22}{len(g):>3}  {nd_s:<24}{on_s:<24}")
    censored = [m for m in measures if m["onset"] is None]
    if censored:
        out.append(
            "  Onset is censored where the trace ends before the rise: "
            + ", ".join(f"{m['id']} (>= {m['recovery_s']:.0f} s)" for m in censored)
            + ". Those holds are in the nadir-delay medians and out of the onset medians,"
            " which are therefore biased short."
        )
    out.append("")

    # Nadir delay against end-of-hold slope, leaving each session out in turn.
    out.append("  Nadir delay against the slope over the last 30 s of the hold (Pearson r):")
    for dev in ("masimo", "pm100"):
        g = [m for m in measures if m["device"] == dev]
        if len({m["session"] for m in g}) < 2:
            out.append(f"    {dev:<8} fewer than two sessions loaded; skipped")
            continue
        r_all = np.corrcoef([m["nadir_delay"] for m in g], [m["end_slope"] for m in g])[0, 1]
        loso = []
        for sess in sorted({m["session"] for m in g}):
            gg = [m for m in g if m["session"] != sess]
            loso.append(
                (
                    sess,
                    np.corrcoef([m["nadir_delay"] for m in gg], [m["end_slope"] for m in gg])[0, 1],
                )
            )
        worst = min(loso, key=lambda x: abs(x[1]))
        out.append(
            f"    {dev:<8} r = {r_all:+.2f} over {len(g)} holds; leaving one session out, r runs"
            f" {min(v for _, v in loso):+.2f} to {max(v for _, v in loso):+.2f}"
            f" (closest to zero without {worst[0]})"
        )
    out.append("")

    # Heart rate as a covariate on onset, after device and slope.
    g = [m for m in measures if m["onset"] is not None and np.isfinite(m["hr_end"])]
    design = np.column_stack(
        [np.ones(len(g)), [m["device"] == "pm100" for m in g], [m["end_slope"] for m in g]]
    ).astype(float)
    y = np.array([m["onset"] for m in g])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    hr = np.array([m["hr_end"] for m in g])
    out.append(
        f"  Onset = {beta[0]:.0f} {beta[1]:+.0f}·[PM100] {beta[2]:+.0f}·slope;"
        f" residual sd {resid.std():.0f} s. Correlation of the residual with end-of-hold"
        f" heart rate: r = {np.corrcoef(resid, hr)[0, 1]:+.2f}"
        f" over {len(g)} holds. Reported, not modelled."
    )
    out.append("")


# ---------------------------------------------------------------- D. post-hold sensor fit


def predict(t, te, s_end, slope, base, delay, plateau, tau_r, tau_s, y0):
    """Latent: the descent continues for T, sits on a floor for P, then rises exponentially to
    the baseline. Sensor: first-order low-pass started at the first observed value."""
    floor = s_end + slope * delay
    latent = np.where(t <= te + delay, s_end + slope * (t - te), floor)
    late = t > te + delay + plateau
    rise = floor + (base - floor) * (1 - np.exp(-(t - te - delay - plateau) / tau_r))
    latent = np.where(late, rise, latent)
    y = np.empty_like(latent)
    y[0] = y0
    a = 1.0 / max(tau_s, 1e-3)
    for i in range(1, len(t)):
        y[i] = y[i - 1] + (1 - np.exp(-a * (t[i] - t[i - 1]))) * (latent[i] - y[i - 1])
    return y


def fit_device(holds, tau_s_fixed=None):
    """Shared (T, P, tau_s) per device, per-hold tau_r. Window: last 30 s of apnea plus recovery."""
    segs = []
    for h in holds:
        t, s, te = h["t"], h["spo2"], h["t_end"]
        w = t >= te - 30
        before = (t > te - 30) & (t <= te)
        slope = (s[before][-1] - s[before][0]) / 30
        base = float(np.median(s[t <= 20]))
        segs.append((t[w], s[w], te, float(s[before][-1]), slope, base))

    def unpack(x):
        tau_s = tau_s_fixed if tau_s_fixed is not None else x[2]
        return x[0], x[1], tau_s, x[3:]

    def residuals(x):
        delay, plateau, tau_s, tau_rs = unpack(x)
        return np.concatenate(
            [
                predict(t, te, s_end, slope, base, delay, plateau, tau_r, tau_s, s[0]) - s
                for (t, s, te, s_end, slope, base), tau_r in zip(segs, tau_rs, strict=True)
            ]
        )

    x0 = np.array([15.0, 10.0, 4.0] + [8.0] * len(segs))
    lo = np.array([0.0, 0.0, 0.5] + [1.0] * len(segs))
    hi = np.array([60.0, 60.0, 60.0] + [120.0] * len(segs))
    res = least_squares(residuals, x0, bounds=(lo, hi))
    delay, plateau, tau_s, tau_rs = unpack(res.x)
    return {
        "T": float(delay),
        "P": float(plateau),
        "tau_s": float(tau_s),
        "tau_r": [float(v) for v in tau_rs],
        "rmse": float(np.sqrt(np.mean(res.fun**2))),
        "n": len(segs),
    }


def sensor_section(holds, out):
    out.append("D. POST-HOLD SENSOR FIT (holds with at least 45 s of trace after the end)")
    out.append("")
    out.append(
        "  Latent SaO2: the last-30-s descent continues for T seconds past the end of the hold,"
        " sits on\n  a floor for P seconds, then rises exponentially to the pre-hold baseline"
        " with a"
        " per-hold time\n  constant tau_r. Sensor: first-order low-pass with time constant tau_s,"
        " started at the first\n  observed value. T absorbs lung-to-finger transit and any pure"
        " device delay, and T + P is when\n  the rise reaches the finger; neither splits into"
        " physiology and device here. tau_s and tau_r are\n  both low-pass, so tau_s is profiled"
        " below; its interval is what the data allow, not what the\n  fit reports."
    )
    out.append("")
    results = {}
    for dev in ("masimo", "pm100"):
        hs = [h for h in holds if device(h["meta"]) == dev and len(h["t_recovery"]) >= 45]
        if len(hs) < 3:
            out.append(f"  {dev}: only {len(hs)} holds with 45 s of recovery; skipped")
            continue
        fit = fit_device(hs)
        boots = []
        for _ in range(200):
            idx = RNG.choice(len(hs), len(hs))
            b = fit_device([hs[i] for i in idx])
            boots.append((b["T"], b["P"], b["tau_s"]))
        boots = np.array(boots)
        profile = []
        for tau_s in (1, 2, 4, 8, 12, 16, 24, 32):
            profile.append((tau_s, fit_device(hs, tau_s_fixed=tau_s)["rmse"]))
        results[dev] = {
            "fit": fit,
            "boot": boots.tolist(),
            "profile": profile,
            "holds": [h["id"] for h in hs],
        }
        lo, hi = np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)
        out.append(
            f"  {dev:<8} n={len(hs):>2}  T = {fit['T']:4.1f} s [{lo[0]:4.1f}, {hi[0]:4.1f}]"
            f"  P = {fit['P']:4.1f} s [{lo[1]:4.1f}, {hi[1]:4.1f}]"
            f"  tau_s = {fit['tau_s']:4.1f} s [{lo[2]:4.1f}, {hi[2]:4.1f}]"
            f"  tau_r median {np.median(fit['tau_r']):4.1f} s  RMSE {fit['rmse']:.2f}"
        )
        out.append(
            "           tau_s profile (RMSE): " + "  ".join(f"{ts}s:{r:.2f}" for ts, r in profile)
        )
        plot_device(dev, hs, fit)
    out.append("")
    return results


def plot_device(dev, hs, fit):
    n = min(len(hs), 8)
    fig, axes = plt.subplots(n, 1, figsize=(9, 2.6 * n), squeeze=False)
    for ax, h, tau_r in zip(axes[:, 0], hs[:n], fit["tau_r"][:n], strict=False):
        t, s, te = h["t"], h["spo2"], h["t_end"]
        w = t >= te - 30
        before = (t > te - 30) & (t <= te)
        slope = (s[before][-1] - s[before][0]) / 30
        base = float(np.median(s[t <= 20]))
        pred = predict(
            t[w],
            te,
            float(s[before][-1]),
            slope,
            base,
            fit["T"],
            fit["P"],
            tau_r,
            fit["tau_s"],
            s[w][0],
        )
        ax.plot(t[w] - te, s[w], "k.", ms=3, label="observed")
        ax.plot(t[w] - te, pred, color="#d62728", lw=1.5, label=f"fit  tau_r={tau_r:.0f}s")
        ax.axvline(0, color="grey", ls="--", alpha=0.6)
        ax.set_ylabel(h["id"], fontsize=7)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel("Seconds after the end of the hold")
    fig.suptitle(
        f"{dev}: post-hold fit, T={fit['T']:.1f}s P={fit['P']:.1f}s tau_s={fit['tau_s']:.1f}s",
        fontsize=10,
    )
    plt.tight_layout()
    OUT.mkdir(exist_ok=True)
    plt.savefig(OUT / f"posthold-{dev}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- main


def main():
    export_path = next((p for p in sys.argv[1:] if p.endswith(".json")), None)
    holds = [h for p in sys.argv[1:] for h in load(p) if qc(h) is None]
    out = []

    if export_path:
        print_trace_statistics(trace_statistics(json.loads(Path(export_path).read_text())), out)
    else:
        out.append(
            "A. TRACE STATISTICS: skipped, no Nadir export given (needs every step's vitals)"
        )
        out.append("")
    freeze_summary(holds, out)
    measures = [lag_measures(h) for h in holds if len(h["t_recovery"]) >= 15]
    lag_section(measures, out)
    sensor = sensor_section(holds, out)

    OUT.mkdir(exist_ok=True)
    (OUT / "lag_measures.json").write_text(json.dumps(measures, indent=1))
    (OUT / "posthold_fit.json").write_text(json.dumps(sensor, indent=1))
    text = "\n".join(out)
    (OUT / "tables.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
