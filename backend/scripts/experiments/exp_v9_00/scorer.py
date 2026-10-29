"""Contract §19 bullet 3: the neutral scorer for the §10 v2 and §18/§19 v5 row rules.

Neither rule chooses the exam. The definitions below were fixed before any score was computed and
are printed verbatim in scorer_results.txt.

Usage (from backend/):
    uv run python scripts/experiments/exp_v9_00/scorer.py \
        --export ~/Downloads/nadir-export-2026-10-29.json \
        --stamina "../Saturday, February 21, 2026.csv" --nhanes-cache <dir with *_E/F/G.xpt> \
        [--profile '{"sex": ..., "age": ..., "heightCm": ..., "weightKg": ...}'] [--boot 2000]
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from archive import HARNESS_TIME_ZONE, finished, load_all, started
from capacity import NO_DIVER, target_context, v5_fit
from capacity import Context as DiverContext
from evaluate import DEFAULT_EXPORT, DEFAULT_STAMINA
from features import eligible_for_model, lung_mode_of
from fit_hold_model import MODES, base_config, fit
from fit_hold_model import predictive as model_predictive
from fit_lung_volume import fit_reference
from fit_oxygen_store import fit_constants
from packing import Packing
from scipy import stats
from static_volumes import ratios_literal

HERE = Path(__file__).resolve().parent
RULES = ("v2", "v5")
UPPER = 0.90  # the one-sided upper bound of (2) and the upper end of (3)'s central 80 %
TARGET_SHARE = 0.10
BOOT, SEED = 2000, 19

DEFINITIONS = f"""Definitions (fixed in scorer.py before any score was computed):
  Chain: finished sessions in (startedAt, id) order. For each held-out session both rules fit
    once on the finished sessions that started strictly before it, the same set for both, with
    the hold-model config base_config() (no half-life, no admitted inputs; v5 adds its own
    holdIndex and logRestBefore). A config chosen on all data would leak the held-out sessions.
  Targets: every performed hold of the held-out session with a trusted lung volume
    (features.eligible_for_model: a known style, and not an estimated or imported label unless
    confirmed), grouped by lung volume (packed counts as full). Nothing else selects them: not
    either rule's d, the stop button (endTrigger, outcome), maxAttempt, plannedDuration or
    symptoms. A session-volume is scored only when both rules return a predictive there, so the
    chain at a volume starts with the first session where both have a posterior.
  Predictive: capacity.predictive's (mu, sd, df) for ln capacity at the hold's own context
    (capacity.target_context), ln T = mu + sd * t_df, with one fit per held-out session.
  (1) Lower-bound log score: each target hold is a lower bound on capacity at its context,
    scored log P(ln T > ln actual) = log S_t((ln actual - mu) / sd; df). Reported as the mean
    per hold; higher is better.
  (2) Upper-bound calibration: per scored session-volume its best hold (the longest target,
    first of equals) is above the bound when actual > exp(mu + t_df({UPPER:.2f}) * sd) at the best
    hold's context. Reported as the share of scored session-volumes; at most
    {TARGET_SHARE:.0%} is the target.
  (3) Sharpness: the median over target holds of the 80 % predictive width in seconds,
    exp(mu + q * sd) - exp(mu - q * sd) with q = t_df({UPPER:.2f}); lower is sharper.
  Quantiles are scipy's Student-t, not the rules' own t90 table.
  Decision per volume (§19): rule A is preferred over B iff share_A(2) <= share_B(2) and
    mean_A(1) > mean_B(1) and median_A(3) <= median_B(3); when neither is preferred the result
    is undecided. "No worse on (2)" is read as "no higher a share"; each share is also marked
    against the {TARGET_SHARE:.0%} target so a reader can apply the threshold reading instead.
  Bootstrap: resamples of the scored session-volumes with replacement (seed {SEED}), giving the
    5-95 % interval of the mean per-hold difference in (1), v5 minus v2, the share of resamples
    where it is positive, and how often each verdict comes out. The predictives stay fixed."""

NOTES = """Notes on the comparison (from the code, not from these numbers):
  - capacity.predictive refits on every call. The scorer fits each rule once per held-out
    session and repeats predictive's lines after the fit (tests/test_scorer.py holds the two
    equal), so every hold of a session is scored against the same fit.
  - With a profile, v5 has a physiology prior before it has a single row, and capacity.predictive
    then raises a ValueError (fit_hold_model.predictive zips v5's two rest inputs against an
    empty centre). The scorer counts such a session-volume as v5 having no predictive and skips
    it; it is listed under "skipped" like any other.
  - Only lower bounds are observed. (1) and (2) both improve as a rule's level rises, so neither
    can punish a level that is too high; only (3), in seconds, pushes back, because the width
    scales with the level as well as with the log spread.
  - Under base_config() v2's predictive does not depend on the hold's context: every hold of a
    session gets the same one. v5's moves with holdIndex and logRestBefore, so at table holds
    it claims a lower, fatigued capacity. The two rules make different claims there: (1) can
    only punish v5's lower claim and (3) rewards it.
  - Before these definitions were written, one orientation probe printed both rules' predictive
    centres next to the session bests, to check the interface and where the chain starts. No
    metric was computed then."""


# ---------------------------------------------------------------- predictives


def predictor(rule, train, diver=NO_DIVER, tz=HARNESS_TIME_ZONE):
    """capacity.predictive with its fit done once: (volume, context) -> (mu, sd, df) or None.
    The lines after the fit repeat predictive's for base_config(), which admits no axes."""
    cfg = base_config()
    fin = finished(train)
    if not fin:
        return lambda volume, context: None
    if rule == "v5":
        model = v5_fit(fin, cfg, diver, tz)[0]
    else:
        model = fit(fin, cfg, diver.population(), tz, packing=diver.packing)
    covariates = model["cfg"]["covariates"]

    def at(volume, context):
        if len(model["xbar"]) != len(covariates):
            return None  # a prior with no rows: capacity.predictive raises here (NOTES)
        p = model_predictive(model, volume, [context.get(k) for k in covariates], [])
        if p is None:
            return None
        off = diver.packing.offset(
            {"style": context.get("style"), "packCount": context.get("packCount")}
        )
        return p[0] + off, p[1], p[2]

    return at


def lower_log_score(actual, p):
    mu, sd, df = p
    return float(stats.t.logsf((math.log(actual) - mu) / sd, df))


def upper_bound(p):
    mu, sd, df = p
    return math.exp(mu + float(stats.t.ppf(UPPER, df)) * sd)


def width80(p):
    mu, sd, df = p
    q = float(stats.t.ppf(UPPER, df))
    return math.exp(mu + q * sd) - math.exp(mu - q * sd)


# ---------------------------------------------------------------- the chain


def targets(session):
    """{volume: step indices} of the session's performed holds with a trusted lung volume."""
    out = {}
    for i, st in enumerate(session["steps"]):
        if eligible_for_model(session, st):
            out.setdefault(lung_mode_of(st), []).append(i)
    return out


def chain(sessions, diver=NO_DIVER, tz=HARNESS_TIME_ZONE):
    """({volume: records}, {volume: skipped dates}): one record per scored held-out
    session-volume with the per-hold scores; skipped where a rule had no predictive."""
    fin = finished(sessions)
    records, skipped = {v: [] for v in MODES}, {v: [] for v in MODES}
    for s in fin:
        by_volume = targets(s)
        if not by_volume:
            continue
        train = [t for t in fin if started(t) < started(s)]
        rules = {r: predictor(r, train, diver, tz) for r in RULES}
        for v, idx in by_volume.items():
            contexts = [target_context(s, i, train, tz) for i in idx]
            preds = [{r: f(v, c) for r, f in rules.items()} for c in contexts]
            if any(p is None for ps in preds for p in ps.values()):
                skipped[v].append(s["startedAt"][:10])
                continue
            actual = [s["steps"][i]["actualDuration"] for i in idx]
            best = preds[actual.index(max(actual))]
            records[v].append(
                {
                    "date": s["startedAt"][:16].replace("T", " "),
                    "actual": actual,
                    "ls": {
                        r: [lower_log_score(a, ps[r]) for a, ps in zip(actual, preds, strict=True)]
                        for r in RULES
                    },
                    "width": {r: [width80(ps[r]) for ps in preds] for r in RULES},
                    "centre": {r: math.exp(best[r][0]) for r in RULES},
                    "upper": {r: upper_bound(best[r]) for r in RULES},
                }
            )
    return records, skipped


# ---------------------------------------------------------------- metrics and the decision


def summarise(recs, rule):
    ls = [x for r in recs for x in r["ls"][rule]]
    exceeded = sum(max(r["actual"]) > r["upper"][rule] for r in recs)
    return {
        "ls": float(np.mean(ls)),
        "exceeded": exceeded,
        "share": exceeded / len(recs),
        "width": float(np.median([x for r in recs for x in r["width"][rule]])),
        "holds": len(ls),
    }


def prefers(a, b):
    """§19: no worse on (2), better on (1), at equal or lower (3)."""
    return a["share"] <= b["share"] and a["ls"] > b["ls"] and a["width"] <= b["width"]


def verdict(m):
    for a, b in (("v2", "v5"), ("v5", "v2")):
        if prefers(m[a], m[b]):
            return a
    return "undecided"


def bootstrap(recs, n=BOOT, seed=SEED):
    """(5 %, 95 %) of the (1) difference v5 - v2, the share of resamples where it is positive,
    and the verdict tally, over session-volumes resampled with replacement."""
    rng = np.random.default_rng(seed)
    diffs, tally = [], Counter()
    for _ in range(n):
        pick = [recs[j] for j in rng.integers(0, len(recs), len(recs))]
        m = {r: summarise(pick, r) for r in RULES}
        diffs.append(m["v5"]["ls"] - m["v2"]["ls"])
        tally[verdict(m)] += 1
    lo, hi = np.percentile(diffs, [5, 95])
    return float(lo), float(hi), float(np.mean(np.array(diffs) > 0)), tally


# ---------------------------------------------------------------- report


def clock(t):
    t = round(t)
    return f"{t // 60}:{t % 60:02d}"


def leader(m, key, higher, word):
    a, b = m["v2"][key], m["v5"][key]
    return "tie" if a == b else ("v2" if (a > b) == higher else "v5") + " " + word


def diver_for(nhanes_cache, profile):
    """The diver-side inputs, built as evaluate.run builds them."""
    if not nhanes_cache:
        return NO_DIVER
    spiro, ratios = fit_reference(nhanes_cache), ratios_literal()
    o2 = fit_constants(spiro, ratios)
    return DiverContext(o2, profile, spiro, ratios, Packing(profile, o2, spiro, ratios))


def run(export, stamina, nhanes_cache=None, profile=None, boot=BOOT, tz=HARNESS_TIME_ZONE):
    stamina = [p for p in stamina if p]
    sessions = load_all(export, stamina)
    records, skipped = chain(sessions, diver_for(nhanes_cache, profile), tz)
    held_out = [s for s in finished(sessions) if targets(s)]
    out = [
        "§19 neutral scorer: the §10 v2 and §18/§19 v5 row rules on one exam neither chooses",
        "=" * 84,
        "",
        "Date: 2026-10-29",
        "Script:  backend/scripts/experiments/exp_v9_00/scorer.py",
        f"Inputs:  Nadir full Export {Path(export).name}, "
        + (", ".join(Path(p).name for p in stamina) or "no STAmina CSV")
        + f", NHANES cache {'used' if nhanes_cache else 'not given'}.",
        "Diver:   "
        + (
            "profile supplied (physiology priors, VO2_d and packing on); its values are not "
            "printed."
            if profile
            else "no profile, so both rules use the placeholder prior and no packing offset."
        ),
        "",
        DEFINITIONS,
        "",
        f"Chain: {len(finished(sessions))} finished sessions, {len(held_out)} held-out sessions "
        f"with target holds ({sum(len(i) for s in held_out for i in targets(s).values())} "
        "holds).",
        f"  {'volume':<8}{'scored sessions':>16}{'holds':>7}  skipped (a rule had no predictive)",
    ]
    for v in MODES:
        recs = records[v]
        out.append(
            f"  {v:<8}{len(recs):>16}{sum(len(r['actual']) for r in recs):>7}  "
            + (", ".join(skipped[v]) or "none")
        )
    out += [
        "",
        "Summary per volume",
        f"  {'volume':<8}{'rule':<6}{'(1) mean log S':>15}{'(2) bests above upper':>26}"
        f"{'(3) median width':>18}",
    ]
    verdicts = []
    for v in MODES:
        recs = records[v]
        if not recs:
            out.append(f"  {v:<8}no scored session-volume (no target holds at this volume)")
            continue
        m = {r: summarise(recs, r) for r in RULES}
        for r in RULES:
            x = m[r]
            share = f"{x['exceeded']}/{len(recs)} ({x['share']:.0%}) " + (
                "pass" if x["share"] <= TARGET_SHARE else "FAIL"
            )
            out.append(
                f"  {v if r == 'v2' else '':<8}{r:<6}{x['ls']:>15.3f}{share:>26}"
                f"{x['width']:>16.0f} s"
            )
        lo, hi, pos, tally = bootstrap(recs, boot)
        w = verdict(m)
        verdicts.append(
            [
                f"  {v}: {w if w == 'undecided' else w.upper()}"
                f" ((1) {leader(m, 'ls', True, 'higher')}, "
                f"(2) {leader(m, 'share', False, 'lower')}, "
                f"(3) {leader(m, 'width', False, 'narrower')})",
                f"    (1) v5 - v2 = {m['v5']['ls'] - m['v2']['ls']:+.3f} per hold; bootstrap over "
                f"{len(recs)} sessions ({boot} resamples): 5-95 % [{lo:+.3f}, {hi:+.3f}], "
                f"positive in {pos:.0%}",
                "    verdict across resamples: "
                + ", ".join(f"{k} {tally[k] / boot:.0%}" for k in (*RULES, "undecided")),
            ]
        )
    out += ["", "Verdict per volume (preferred rule in capitals)"]
    for lines in verdicts:
        out += lines
    out += [
        "",
        "Per-session tables (centre and 90 % upper bound at the best hold's context; * marks a "
        "best above it;",
        "(1) is the session's mean log S over its target holds)",
    ]
    for v in MODES:
        if not records[v]:
            continue
        out += [
            f"  {v}:",
            f"    {'session':<18}{'holds':>5}{'best':>7}{'v2 centre':>11}{'upper':>8}"
            f"{'v5 centre':>11}{'upper':>8}{'(1) v2':>9}{'(1) v5':>9}",
        ]
        for r in records[v]:
            best = max(r["actual"])
            cells = "".join(
                f"{clock(r['centre'][k]):>11}"
                f"{clock(r['upper'][k]) + ('*' if best > r['upper'][k] else ' '):>8}"
                for k in RULES
            )
            out.append(
                f"    {r['date']:<18}{len(r['actual']):>5}{clock(best):>7}{cells}"
                + "".join(f"{np.mean(r['ls'][k]):>9.3f}" for k in RULES)
            )
    out += ["", NOTES, ""]
    return "\n".join(out)


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--export", default=str(DEFAULT_EXPORT))
    p.add_argument("--stamina", nargs="*", default=[str(DEFAULT_STAMINA)])
    p.add_argument("--nhanes-cache")
    p.add_argument("--profile", help="JSON as evaluate.py takes it; its values are never printed")
    p.add_argument("--boot", type=int, default=BOOT)
    return p


if __name__ == "__main__":
    a = cli().parse_args()
    text = run(
        a.export,
        a.stamina,
        Path(a.nhanes_cache).expanduser() if a.nhanes_cache else None,
        json.loads(a.profile) if a.profile else None,
        a.boot,
    )
    (HERE / "scorer_results.txt").write_text(text)
    print(text)
