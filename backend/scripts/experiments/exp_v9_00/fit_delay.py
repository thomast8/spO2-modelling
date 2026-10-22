"""Contract §4: post-hold delay per oximeter (recovery onset, Normal-Inverse-Gamma on session
means).

Hyperparameters (harness choice, data-informed): kappa0 = 2; mu0 = the mean of every session's
mean onset across both oximeters; the between-session variance s² is pooled within oximeter
(Σ (x − x̄_key)² / (N − K)); alpha0 = 2 so the prior mean of σ² is beta0 = s²·(alpha0 − 1) = s².
"""

import math

from archive import HARNESS_ROSTER, finished, is_performed_hold
from evidence import DEFAULT_FLOORS, evaluate
from features import is_censored_trace, oximeter_key, stamped_trace
from numerics import (
    nig_mean_interval,
    nig_predictive_interval,
    nig_rows_to_reach,
    nig_update,
)

KAPPA0, ALPHA0 = 2.0, 2.0
KEYS = ("stamina", "pm100")


def session_rows(sessions, key, roster=HARNESS_ROSTER):
    """[(session, mean onset | None, holds with onset, censored holds)] for sessions with `key`."""
    out = []
    for s in finished(sessions):
        if oximeter_key(s, roster) != key:
            continue
        onsets, censored = [], 0
        for i, st in enumerate(s["steps"]):
            if not is_performed_hold(st):
                continue
            trace = stamped_trace(i, s)
            if trace and trace["onsetSeconds"] is not None:
                onsets.append(float(trace["onsetSeconds"]))
            elif is_censored_trace(trace):
                censored += 1
        out.append((s, sum(onsets) / len(onsets) if onsets else None, len(onsets), censored))
    return out


def choose_prior(sessions, roster=HARNESS_ROSTER, exclude_id=None):
    groups = []
    for key in KEYS:
        xs = [
            x
            for s, x, _, _ in session_rows(sessions, key, roster)
            if x is not None and s["id"] != exclude_id
        ]
        if xs:
            groups.append(xs)
    allx = [x for g in groups for x in g]
    n, k = len(allx), len(groups)
    ss = sum((x - sum(g) / len(g)) ** 2 for g in groups for x in g)
    s2 = ss / (n - k)
    return sum(allx) / n, KAPPA0, ALPHA0, s2 * (ALPHA0 - 1)


def post_hold_delay(
    sessions, key, prior, roster=HARNESS_ROSTER, previously_shown=False, floors=None
):
    """`Analytics.postHoldDelay(oximeter:in:roster:previouslyShown:)` → dict with estimate and
    gate."""
    rows = session_rows(sessions, key, roster)
    xs = [x for _, x, _, _ in rows if x is not None]
    post = nig_update(prior, xs)
    lo, mid, hi = nig_mean_interval(post)
    plo, _, phi_ = nig_mean_interval(prior)
    w_prior = phi_ - plo
    narrowing = 1 - (hi - lo) / w_prior
    f = (floors or DEFAULT_FLOORS)["delay"]
    gate = evaluate(
        f"delay.{key}",
        f,
        {("sessionsWithOximeter", key): len(xs)},
        (narrowing, nig_rows_to_reach(post, w_prior, f["rhoEnough"])),
        previously_shown,
    )
    return {
        "key": key,
        "sessions": len(xs),
        "holds": sum(h for _, x, h, _ in rows if x is not None),
        "censored": sum(c for *_, c in rows),
        "posterior": post,
        "estimate": {"lower": lo, "value": mid, "upper": hi, "confidence": 0.8},
        "prior_interval": (plo, phi_),
        "narrowing": narrowing,
        "gate": gate,
        "session_means": [(s["startedAt"][:10], x, h, c) for s, x, h, c in rows],
    }


def loso(sessions, key, roster=HARNESS_ROSTER):
    """Leave one session out: is its mean onset inside the 80 % predictive interval of the rest?
    The prior is re-chosen without the held-out session too."""
    rows = [(s, x) for s, x, _, _ in session_rows(sessions, key, roster) if x is not None]
    out = []
    for s, x in rows:
        prior = choose_prior(sessions, roster, exclude_id=s["id"])
        rest = [y for t, y in rows if t["id"] != s["id"]]
        lo, mid, hi = nig_predictive_interval(nig_update(prior, rest))
        out.append((s["startedAt"][:10], x, lo, mid, hi, lo <= x <= hi))
    return out


def report(sessions, out):
    prior = choose_prior(sessions)
    mu0, k0, a0, b0 = prior
    out.append("1. POST-HOLD DELAY PER OXIMETER (contract §3, §4)")
    out.append("-" * 50)
    out.append("")
    out.append(
        "Recovery onset per hold: the first trusted SpO₂ reading 2 points above the post-hold "
        "minimum,\ncounted from after the last time that minimum is reached, walking every step "
        "up to the next\nperformed hold. One row per session: its mean onset."
    )
    out.append("")
    out.append(
        f"Prior (data-informed, kappa0 = 2): mu0 = {mu0:.1f} s (mean of all session means), "
        f"between-session sd\n{math.sqrt(b0 / (a0 - 1)):.1f} s pooled within oximeter, "
        f"alpha0 = {a0:g}, beta0 = {b0:.1f}. Prior 80 % interval on the mean:\n"
        f"{nig_mean_interval(prior)[0]:.1f} to {nig_mean_interval(prior)[2]:.1f} s."
    )
    out.append("")
    results = {}
    for key in KEYS:
        r = post_hold_delay(sessions, key, prior)
        results[key] = r
        out.append(f"  {key}: sessions with SpO₂ and this key")
        out.append(f"    {'session':<12}{'mean onset':>11}{'holds':>7}{'censored':>9}")
        shown = [row for row in r["session_means"] if row[1] is not None or row[3]]
        for day, x, h, c in shown:
            xs = "    -" if x is None else f"{x:7.1f} s"
            out.append(f"    {day:<12}{xs:>11}{h:>7}{c:>9}")
        other = len(r["session_means"]) - len(shown)
        if other:
            out.append(f"    ({other} more sessions with this key have no post-hold SpO₂)")
        e = r["estimate"]
        mun, kn, an, bn = r["posterior"]
        out.append(
            f"    posterior on the mean: {e['value']:.1f} s, 80 % {e['lower']:.1f} to "
            f"{e['upper']:.1f} s (df {2 * an:g}), n = {r['sessions']} sessions, "
            f"{r['holds']} holds, {r['censored']} censored"
        )
        out.append(f"    narrowing against the prior: {r['narrowing']:.2f}")
        cov = loso(sessions, key)
        hits = sum(1 for *_, ok in cov if ok)
        out.append(
            f"    leave-one-session-out 80 % predictive coverage: {hits}/{len(cov)}"
            + (f" ({hits / len(cov):.0%})" if cov else "")
        )
        for day, x, lo, _mid, hi, ok in cov:
            out.append(
                f"      {day}  {x:5.1f} s in [{lo:5.1f}, {hi:5.1f}]  {'yes' if ok else 'NO'}"
            )
        out.append("")
    out.append(
        "  Compensation rule (frozen): considered only when the per-hold predictive 80 % width "
        "on an\n  oximeter is ≤ 15 s over ≥ 10 sessions and a two-oximeter recording has "
        "separated device\n  delay from circulation transit. The export cannot hold a "
        "two-oximeter recording\n  (one SpO₂ source per session), so that condition is false."
    )
    for key in KEYS:
        r = results[key]
        lo, _, hi = nig_predictive_interval(r["posterior"])
        met = hi - lo <= 15 and r["sessions"] >= 10 and False
        out.append(
            f"  {key}: {r['sessions']} sessions, predictive 80 % width for a new session's "
            f"mean {hi - lo:.1f} s: {'MET' if met else 'NOT MET'}"
        )
    out.append("")
    return prior, results
