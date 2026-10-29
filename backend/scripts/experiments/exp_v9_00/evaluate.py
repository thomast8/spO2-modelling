"""v9.00: fit every learning-framework prior, run the forward chain and the gates, write the
results.

Usage (from backend/):
    uv run python scripts/experiments/exp_v9_00/evaluate.py \
        --export ~/Downloads/nadir-export-2026-10-29.json \
        --stamina "../Saturday, February 21, 2026.csv" --nhanes-cache <dir with *_E/F/G.xpt> \
        [--profile '{"sex": "male", "age": 35, "heightCm": 180, "weightKg": 75}']

`run()` returns everything `emit_swift.py` and `emit_fixtures.py` need.
"""

import argparse
import json
import math
from datetime import timedelta
from pathlib import Path

from archive import HARNESS_TIME_ZONE, load_all, started
from capacity import Context as CapacityContext
from capacity import Distance, capacity_rows
from capacity import report as capacity_report
from evidence import DEFAULT_FLOORS, evaluate
from features import (
    AXES,
    AXIS_UNIT,
    diver_axes,
    hold_markers,
    hold_model_rows,
    oximeter_key,
)
from fit_delay import KEYS
from fit_delay import report as delay_report
from fit_evidence import calibrate
from fit_hold_model import MODES, estimates, fit
from fit_hold_model import report as hold_model_report
from fit_hold_volume import report as hold_volume_report
from fit_lung_volume import fit_reference
from fit_oxygen_store import fit_constants, physiology_prior
from packing import NONE as PACKING_NONE
from packing import Packing, packed_gate
from packing import report as packing_report
from static_volumes import compare_routes, ratios_literal, static_volumes

HERE = Path(__file__).resolve().parent
DEFAULT_EXPORT = Path.home() / "Downloads" / "nadir-export-2026-10-29.json"
DEFAULT_STAMINA = HERE.parents[3] / "Saturday, February 21, 2026.csv"
# DiverProfile identifier → the key the harness profile uses (age stands in for dateOfBirth).
PROFILE_FIELDS = (
    ("sex", "sex"),
    ("dateOfBirth", "age"),
    ("heightCm", "heightCm"),
    ("weightKg", "weightKg"),
)


OPEN_CHOICES = [
    "7. CHOICES THE CONTRACT LEAVES TO THE HARNESS",
    "-" * 45,
    "",
    "  - Delay prior: data-informed (mu0 = mean of all session means, beta0 = pooled "
    "between-session\n    variance with alpha0 = 2, kappa0 = 2). LOSO re-chooses it without "
    "the held-out session.",
    "  - Time zone for §9 time of day: Europe/London (the day-only STAmina imports start at "
    "23:00Z in\n    August).",
    "  - Hold-volume standardisation: population SD over the labelled rows; band grid 0.05.",
    "  - Hold model σ² prior: inverse-gamma a0 = 3, b0 = 0.045 (prior mean σ = 0.15), judgement.",
    "  - Half-life: admitted like any input (must beat none on log score with coverage 70-90 %).",
    "  - Coefficients for a candidate input: ridge on uncensored rows within volumes, prior sd "
    "0.2 per input\n    sd; censored rows do not inform coefficients.",
    "  - level rowsToReach (the contract gives pb's only): smallest m with "
    "t90(2an+m)·sqrt(1/(1/s0² + (n+m)/σ²))\n    ≤ (1 − ρ)·1.2816·s0.",
    "  - A volume with no prior waits with 1 more max attempt at full lung (any outcome sets "
    "the placeholder).",
    "  - Point baselines (PB-fallback, comparable-history) are scored with the base model's "
    "spread;\n    comparable-history is openHoldEstimates for an open hold, the planned length "
    "for a timed one.",
    "  - Static volumes: a measured TLC is the TLC that the FRC and RV ratios multiply; the "
    "ratio log-SDs\n    are the larger sex's at 45 y.",
    "  - Oxygen store σ_MC uses the reference diver with measured TLC and route-2 FRC/RV.",
    "  - The delay calibration chain reuses the prior chosen on all sessions (mild leakage); it "
    "moves no floor.",
    "  - Profile-field units carry the DiverProfile identifiers (sex, dateOfBirth, heightCm, "
    "weightKg); the\n    sentence uses the display names. lungVolume.tlc is required only when "
    "every field is present and\n    no TLC can be estimated; its display name is TLC.",
    "  - Spirometry: unweighted (NHANES survey weights ignored); the 2011-12 cycle covers 84-89 % "
    "held out\n    while the earlier two cover 90-93 %, a cycle effect that one σ scale cannot "
    "remove.",
    "",
]


def profile_gates(profile, spiro, ratios):
    """spirometry and oxygenStore gates (§16): profile fields by identifier; lungVolume.tlc only
    when every field is present and no TLC can be measured or estimated."""
    profile = profile or {}
    fields = {
        ("profileField", ident): int(profile.get(k) is not None) for ident, k in PROFILE_FIELDS
    }
    spiro_fields = {u: c for u, c in fields.items() if u[1] != "weightKg"}
    g_spiro = evaluate("spirometry", DEFAULT_FLOORS["spirometry"], spiro_fields, None, False)
    counts = dict(fields)
    if all(fields.values()) and static_volumes(profile, spiro, ratios) is None:
        counts[("lungVolume", "tlc")] = 0
    g_o2 = evaluate("oxygenStore", DEFAULT_FLOORS["oxygenStore"], counts, None, False)
    return g_spiro, g_o2


def axes_section(sessions, t_now, h, out):
    out.append("5. HOLD MARKERS AND DIVER AXES (contract §7, §8)")
    out.append("-" * 47)
    out.append("")
    n_holds = sum(len(hold_markers(s)) for s in sessions)
    out.append(
        f"Markers are computed for all {n_holds} performed holds (measurements, always shown). "
        "Axes are medians\nover holds of 60 s or more before the as-of date, imported holds only "
        "when confirmed, pooled\nacross lung modes (the per-oximeter split below is the harness's "
        "only).\n"
        + "No axis scores how low SpO₂ went (nadir#703, #706): the low is a per-hold marker only, "
        "and the\nhypoxicTolerance axis is removed by decision. desaturationRate is the app's "
        "HoldOxygenDrop rate.\n"
        + (
            "Unweighted in this run: no half-life is in use."
            if h is None
            else f"Weighted by recency with a {h:g}-day half-life."
        )
    )
    out.append("")
    as_of = t_now + timedelta(seconds=1)
    ax = diver_axes(sessions, as_of, h)
    split = {}
    for key in KEYS:
        subset = [
            s for s in sessions if oximeter_key(s, [{"id": "pm100", "kind": "oximeter"}]) == key
        ]
        split[key] = diver_axes(subset, as_of, h)
    gates = {}
    out.append(
        f"  {'axis':<18}{'value':>9}{'holds':>7}{'effective':>10}  {'stamina':>14}  "
        f"{'pm100':>14}  gate"
    )
    for name, a in ax.items():
        unit = (AXIS_UNIT[name],)
        g = evaluate(f"axis.{name}", DEFAULT_FLOORS["axis"], {unit: a["count"]}, None, False)
        gates[name] = g
        val = "-" if a["value"] is None else f"{a['value']:.3g}"
        parts = [
            "-"
            if split[k][name]["value"] is None
            else f"{split[k][name]['value']:.3g} ({split[k][name]['count']})"
            for k in KEYS
        ]
        out.append(
            f"  {name:<18}{val:>9}{a['count']:>7}{a['effectiveCount']:>10.1f}  {parts[0]:>14}  "
            f"{parts[1]:>14}  {'enough' if g.enough else g.summary}"
        )
    out.append(
        "  Units: co2Tolerance first-contraction share of the hold, urgeOnset s, desaturationRate "
        "points per\n  minute falling (HoldOxygenDrop), diveReflex HR drop share, recovery s back "
        "to the "
        "pre-hold reading."
    )
    out.append("")
    return ax, gates


def run(
    export=DEFAULT_EXPORT, stamina=(DEFAULT_STAMINA,), nhanes_cache=None, profile=None, write=True
):
    sessions = load_all(export, [p for p in stamina if p])
    t_now = max(started(s) for s in sessions)
    out = [
        "v9.00 Results: the learning framework's priors, gates and forward chain on one diver",
        "=" * 84,
        "",
        "Date: 2026-10-29",
        "Script:  backend/scripts/experiments/exp_v9_00/evaluate.py (emit_swift.py, "
        "emit_fixtures.py)",
        f"Inputs:  Nadir full Export {Path(export).name} ({len(sessions) - len(stamina)} sessions "
        "after dedupe), the 21 February 2026\n         STAmina CSV rebuilt as StaminaImport would, "
        "NHANES 2007-2012 (SPX, DEMO, BMX, SMQ, MCQ).",
        "Contract: CONTRACT.md (v1 with §10 v2, §16, §17 packing, §18 capacity). Time zone for §9: "
        f"{HARNESS_TIME_ZONE}.",
        "Labels: the 6 August 2026 relabel (holds 1-8 RV, hold 9 full) is applied as "
        'styleSource "confirmed".',
        "",
        "Question: what does every learned quantity in the plan look like on Thomas's own "
        "recordings,\nwhich of them pass the evidence gate today, and what does each waiting one "
        "still need?",
        "",
        "",
    ]
    delay_prior, delay = delay_report(sessions, out)
    hold_volume = hold_volume_report(sessions, out)

    phys, spiro, ratios, o2 = None, None, None, None
    if nhanes_cache:
        out.append("3. LUNG VOLUMES AND OXYGEN STORES (contract §11, §12)")
        out.append("-" * 52)
        out.append("")
        spiro = fit_reference(nhanes_cache, out)
        ratios = ratios_literal()
        compare_routes(spiro, ratios, out)
        o2 = fit_constants(spiro, ratios, out)
        if profile:
            phys = physiology_prior(o2, profile, spiro, ratios)
        out.append(
            "  Thomas's physiology prior: "
            + (
                "from --profile, "
                + ", ".join(f"{v} {math.exp(p[0]):.0f} s (σ {p[1]:.2f})" for v, p in phys.items())
                if phys
                else "unavailable, the export carries no profile (sex, date of birth, height, "
                "weight), so the\n  hold model below uses its placeholder prior."
            )
        )
        out.append("")
    pk = Packing(profile, o2, spiro, ratios) if o2 is not None else PACKING_NONE
    hm = hold_model_report(sessions, out, physiology=phys, packing=pk)
    packing_report(sessions, out, pk, hm, phys, HARNESS_TIME_ZONE)
    ctx = CapacityContext(o2, profile, spiro, ratios, pk)
    cap = capacity_report(sessions, out, hm["config"], ctx, HARNESS_TIME_ZONE)
    rule = "v5" if cap["compare"]["admitted"] else "v2"
    hm["config"]["rowRule"] = rule
    ax, axis_gates = axes_section(sessions, t_now, hm["half_life"], out)

    out.append("6. EVIDENCE GATES (contract §13, §16) AND CALIBRATION")
    out.append("-" * 52)
    out.append("")
    cal = calibrate(sessions, hm["config"], delay_prior, phys, out, packing=pk, rule=rule, ctx=ctx)
    if rule == "v5":
        dist = Distance(sessions)
        _, own = ctx.physiology(sessions)
        rows = capacity_rows(sessions, dist)
        model = fit(sessions, hm["config"], own, rows=rows, packing=pk)
    else:
        model = fit(sessions, hm["config"], phys, packing=pk)
    est = estimates(model, sessions, floors=cal["floors"], rule=rule)
    out.append(f"  Shipped row rule: {rule} (§18 chain above); the gates below are its gates.")
    gates = [delay[k]["gate"] for k in KEYS]
    gates += [hold_volume["gate"]]
    gates += [axis_gates[n] for n in AXES]
    for v in MODES:
        gates += [est[v]["pb"]["gate"], est[v]["level"]["gate"]]
    gates.append(packed_gate(est["full"]["pb"]["gate"], pk))
    if spiro:
        gates += list(profile_gates(profile, spiro, ratios))
    out.append(f"  {'key':<24}{'state':<11}{'narrowing':>10}  counts / sentence")
    for g in gates:
        nar = "-" if g.narrowing is None else f"{g.narrowing:.2f}"
        out.append(
            f"  {g.key:<24}{'enough' if g.enough else 'waiting':<11}{nar:>10}  "
            f"{json.dumps(g.counts)}"
        )
        if not g.enough:
            out.append(f"  {'':<45}{g.summary}")
    out.append("")
    out.append(f"  Priors.evidence header: {cal['header']}")
    out.append("")
    out += OPEN_CHOICES

    def clock(t):
        return f"{int(t) // 60}:{round(t) % 60:02d}"

    notes = []
    v2_full = hm["estimates"]["full"]["level"]["estimate"]
    rows = [r for s in sessions for r in hold_model_rows(s) if r["mode"] == "full"]
    if v2_full and rows:
        longest = max(r["actual"] for r in rows)
        unc = sum(1 for r in rows if not r["censored"])
        notes.append(
            f"  - §10 v2: the full-lung level ({clock(v2_full[1])}) sits "
            f"{'above' if v2_full[1] > longest else 'below'} the longest full-slot row "
            f"({clock(longest)}); {unc} of the {len(rows)} full-slot\n    rows are diver-ended, "
            "the rest are lower bounds that lift the level."
        )
    v5_full = cap["estimates"]["full"]["level"]["estimate"]
    limit = [r for r in cap["rows"] if r["mode"] == "full" and not r["censored"]]
    if v5_full and limit:
        short = sum(1 for r in limit if r["actual"] < 180)
        notes.append(
            f"  - §18 v5: the full-lung level ({clock(v5_full[1])}) rests on {len(limit)} holds "
            f"at the limit (d ≥ 0.8); {short} of them are\n    under 3:00, table holds whose "
            "struggle share or fall reached the diver's 90th percentile. No rest-before\n    "
            "covariate is admitted to tell a table limit from a rested one, so they pull the level "
            "down."
        )
    notes += [
        "  - §18 d: the fall is spo2Baseline − spo2Low (the post-hold trough), both 90th "
        "percentiles pooled\n    across lung volumes; VO2_d reads only holds of known volume and "
        "skips holds with no fall.",
        "  - §18 chain: both rules are scored on the holds v5's own d calls the limit, so the test "
        "set is v5's\n    choice, not a neutral one.",
    ]
    out[len(out) - 1 : len(out) - 1] = notes
    text = "\n".join(out) + "\n"
    if write:
        (HERE / "exp_v9_00_results.txt").write_text(text)
    return {
        "sessions": sessions,
        "delay_prior": delay_prior,
        "delay": delay,
        "hold_volume": hold_volume,
        "spirometry": spiro,
        "ratios": ratios,
        "oxygen_store": o2,
        "physiology": phys,
        "hold_model": hm,
        "capacity": cap,
        "calibration": cal,
        "gates": gates,
        "axes": ax,
        "text": text,
    }


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--export", default=str(DEFAULT_EXPORT))
    p.add_argument("--stamina", nargs="*", default=[str(DEFAULT_STAMINA)])
    p.add_argument("--nhanes-cache")
    p.add_argument(
        "--profile",
        help="JSON: sex, age, heightCm, weightKg, optional measured {vc, fvc, tlc, frc, rv}",
    )
    return p


if __name__ == "__main__":
    a = cli().parse_args()
    r = run(a.export, a.stamina, a.nhanes_cache, json.loads(a.profile) if a.profile else None)
    print(r["text"])
