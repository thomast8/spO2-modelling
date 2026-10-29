"""Contract §18 (v5): capacity from every hold. The limit is read from the trace, not the flag.

- Per-diver consumption VO2_d from every performed hold of known volume with ≥ 60 s of trusted
  SpO₂: the usable store at its volume times the share of the usable desaturation its fall reached
  (spo2Baseline → spo2End against the population end saturation), over its duration. Log-normal
  prior N(ln VO2_pop, 0.25²); weighted mean of log consumption (weights = trusted seconds),
  Student-t interval over the between-hold spread.
- Distance to the limit d ∈ [0, 1]: the larger of the fall (spo2Baseline − spo2Low) over the
  diver's 90th-percentile fall and the struggle share over its 90th percentile; 0 when neither is
  measurable. Descriptive, never shown.
- Capacity rows: every eligible performed hold with a trace, weight d², a lower bound unless
  d ≥ 0.8.
"""

import math

import numpy as np
from archive import finished, is_performed_hold, started
from features import (
    cached_markers,
    eligible_for_model,
    lung_mode_of,
    oxygen_readings,
    past_limit_hold,
)
from fit_hold_model import MODES, fit, inputs, predictive
from fit_oxygen_store import physiology_prior, stores, volumes_for
from numerics import Z90, t90
from scipy import stats

TRACE_SECONDS = 60
LIMIT = 0.8
VO2_PRIOR_SD = 0.25


def trace_seconds(step):
    """Seconds with a trusted SpO₂ reading inside the hold."""
    return len(oxygen_readings(step))


def has_trace(step):
    return is_performed_hold(step) and trace_seconds(step) >= TRACE_SECONDS


# ---------------------------------------------------------------- VO2


def hold_consumption(store_ml, baseline, end, end_population, duration_s):
    """mL/min the hold's fall implies: store × share of the usable desaturation reached / time.
    None when there is no fall to read."""
    if baseline is None or end is None or duration_s <= 0 or baseline <= end_population:
        return None
    share = min(max((baseline - end) / (baseline - end_population), 0.0), 1.0)
    if share <= 0:
        return None
    return store_ml * share * 60 / duration_s


def combine_vo2(prior_ml_min, values, weights):
    """(VO2, 80 % low, 80 % high, effective holds): the log-normal prior with the weighted mean
    of log consumption, its standard error from the weighted between-hold spread."""
    lp, prior_var = math.log(prior_ml_min), VO2_PRIOR_SD**2
    if len(values) < 2:
        half = Z90 * VO2_PRIOR_SD
        return prior_ml_min, math.exp(lp - half), math.exp(lp + half), float(len(values))
    logs, w = np.log(np.array(values)), np.array(weights, float)
    total = w.sum()
    m = float((w * logs).sum() / total)
    n_eff = float(total**2 / (w**2).sum())
    spread = float((w * (logs - m) ** 2).sum() / total) * n_eff / (n_eff - 1)
    se2 = max(spread / n_eff, 1e-12)
    precision = 1 / prior_var + 1 / se2
    mu = (lp / prior_var + m / se2) / precision
    half = t90(n_eff - 1) * math.sqrt(1 / precision)
    return math.exp(mu), math.exp(mu - half), math.exp(mu + half), n_eff


def hold_store(step, constants, profile, spiro, ratios, packing):
    """Usable O₂ store (mL) at the hold's own lung volume, packing included."""
    vols = volumes_for(profile, spiro, ratios)
    mode = lung_mode_of(step)
    if vols is None or mode is None:
        return None
    volume = vols[mode][0]
    if step.get("style") == "packed" and packing is not None:
        volume += packing.delta_v(step.get("packCount"))
    return stores(constants, profile, {mode: volume})["usable"][mode]


def population_vo2(constants, profile):
    return profile["weightKg"] * constants["restingVO2PerKg"] * constants["diveReflexFactor"]


def fit_vo2(sessions, constants, profile, spiro, ratios, packing=None):
    """VO2_d with its interval and the holds behind it, or None without a profile."""
    if (
        not profile
        or profile.get("weightKg") is None
        or volumes_for(profile, spiro, ratios) is None
    ):
        return None
    end_pop = 100 * constants["arterialSaturationEnd"]
    values, weights = [], []
    for s in finished(sessions):
        markers = cached_markers(s)
        for i, st in enumerate(s["steps"]):
            if not (has_trace(st) and eligible_for_model(s, st)):
                continue
            store = hold_store(st, constants, profile, spiro, ratios, packing)
            m = markers[i]
            c = hold_consumption(
                store, m["spo2Baseline"], m["spo2End"], end_pop, st["actualDuration"]
            )
            if c is not None:
                values.append(c)
                weights.append(trace_seconds(st))
    vo2, lo, hi, n_eff = combine_vo2(population_vo2(constants, profile), values, weights)
    return {"vo2": vo2, "low": lo, "high": hi, "holds": len(values), "effective": n_eff}


# ---------------------------------------------------------------- distance to the limit


class Distance:
    """The diver's own 90th-percentile fall and struggle share, from the given sessions."""

    def __init__(self, sessions):
        falls, shares = [], []
        for s in finished(sessions):
            for i, st in enumerate(s["steps"]):
                if not is_performed_hold(st):
                    continue
                fall, share = self.measures(s, i)
                if fall is not None:
                    falls.append(fall)
                if share is not None:
                    shares.append(share)
        self.fall90 = float(np.percentile(falls, 90)) if falls else None
        self.share90 = float(np.percentile(shares, 90)) if shares else None

    @staticmethod
    def measures(session, index):
        """(SpO₂ fall, struggle share), each None when not measurable."""
        st = session["steps"][index]
        m = cached_markers(session)[index]
        fall = None
        if has_trace(st) and m["spo2Baseline"] is not None and m["spo2Low"] is not None:
            fall = max(m["spo2Baseline"] - m["spo2Low"], 0.0)
        share = None
        if m["firstContractionSeconds"] is not None and st["actualDuration"] > 0:
            share = (
                max(st["actualDuration"] - m["firstContractionSeconds"], 0) / st["actualDuration"]
            )
        return fall, share

    def d(self, session, index):
        if index == past_limit_hold(session):
            return 1.0  # a reported blackout/LMC/samba: that hold went past the limit
        fall, share = self.measures(session, index)
        parts = []
        if fall is not None and self.fall90:
            parts.append(min(fall / self.fall90, 1.0))
        if share is not None and self.share90:
            parts.append(min(share / self.share90, 1.0))
        return max(parts) if parts else 0.0


def capacity_rows(sessions, distance):
    """§18 rows: every eligible performed hold with a trace, weight d², censored unless d ≥ 0.8.
    No symptom excludes a hold."""
    rows = []
    for s in finished(sessions):
        for i, st in enumerate(s["steps"]):
            if not (has_trace(st) and eligible_for_model(s, st)):
                continue
            d = distance.d(s, i)
            rows.append(
                {
                    "session": s,
                    "stepIndex": i,
                    "mode": lung_mode_of(st),
                    "actual": st["actualDuration"],
                    "y": math.log(st["actualDuration"]),
                    "censored": d < LIMIT,
                    "weight": d * d,
                    "d": d,
                }
            )
    return rows


# ---------------------------------------------------------------- v2 against v5


def compare_rules(sessions, cfg, ctx, tz, burn_in=5):
    """Forward chain: at each session after the burn-in, both rules are fitted on the sessions
    before it and scored on its holds with d ≥ 0.8 (d from the earlier sessions' percentiles),
    as observations. Returns per-rule summaries over the holds both rules could predict."""
    fin = finished(sessions)
    with_trace = [s for s in fin if any(has_trace(st) for st in s["steps"])]
    scored = {"v2": [], "v5": []}
    for s in with_trace[burn_in:]:
        train = [t for t in fin if started(t) < started(s)]
        dist = Distance(train)
        tests = [r for r in capacity_rows([s], dist) if not r["censored"]]
        if not tests:
            continue
        phys2, phys5 = ctx.physiology(train)
        m2 = fit(train, cfg, phys2, tz, packing=ctx.packing)
        m5 = fit(train, cfg, phys5, tz, rows=capacity_rows(train, dist), packing=ctx.packing)
        for r in tests:
            step = r["session"]["steps"][r["stepIndex"]]
            off = ctx.packing.offset(step) if ctx.packing else 0.0
            x, a = inputs(r, train + [s], cfg, tz)
            p2, p5 = predictive(m2, r["mode"], x, a), predictive(m5, r["mode"], x, a)
            if p2 is None or p5 is None:
                continue
            for rule, (loc, scale, df) in (("v2", p2), ("v5", p5)):
                z = (r["y"] - loc - off) / scale
                scored[rule].append(
                    {
                        "logscore": float(stats.t.logpdf(z, df) - math.log(scale)),
                        "covered": abs(z) <= t90(df),
                    }
                )
    out = {}
    for rule, xs in scored.items():
        out[rule] = {
            "scored": len(xs),
            "logscore": sum(x["logscore"] for x in xs),
            "coverage": sum(x["covered"] for x in xs) / len(xs) if xs else None,
        }
    n = out["v5"]["scored"]
    out["decidable"] = any(0.7 <= k / n <= 0.9 for k in range(n + 1)) if n else False
    v2, v5 = out["v2"], out["v5"]
    out["admitted"] = bool(
        n
        and v5["logscore"] > v2["logscore"]
        and v5["coverage"] is not None
        and 0.7 <= v5["coverage"] <= 0.9
    )
    return out


class Context:
    """The profile-dependent pieces both rules need: population and fitted-VO2 physiology."""

    def __init__(self, constants, profile, spiro, ratios, packing):
        self.constants, self.profile, self.spiro, self.ratios = constants, profile, spiro, ratios
        self.packing = packing

    def physiology(self, sessions):
        """(population prior for v2, prior at VO2_d for v5); (None, None) without a profile."""
        if not self.profile or self.constants is None:
            return None, None
        pop = physiology_prior(self.constants, self.profile, self.spiro, self.ratios)
        vo2 = fit_vo2(sessions, self.constants, self.profile, self.spiro, self.ratios, self.packing)
        own = None
        if vo2 is not None:
            own = physiology_prior(
                self.constants, self.profile, self.spiro, self.ratios, vo2_ml_min=vo2["vo2"]
            )
        return pop, own


def _clock(t):
    return f"{int(t) // 60}:{round(t) % 60:02d}"


def report(sessions, out, cfg, ctx, tz, floors=None):
    """Section 4c: VO2, d per hold, v2 against v5 in the chain, the v5 model on all data."""
    from fit_hold_model import estimates

    fin = finished(sessions)
    out.append("4c. CAPACITY FROM EVERY HOLD (contract §18, row rule v5)")
    out.append("-" * 55)
    out.append("")
    out.append(
        "Every hold with 60 s or more of trusted SpO₂ is a row, weighted by d² (how close the "
        "trace came to\nthe diver's own limit) and a lower bound unless d ≥ 0.8, whoever ended it. "
        "Consumption is fitted\nfrom the same holds when a profile is given."
    )
    out.append("")
    vo2 = None
    if ctx.profile and ctx.constants is not None:
        vo2 = fit_vo2(fin, ctx.constants, ctx.profile, ctx.spiro, ctx.ratios, ctx.packing)
    if vo2 is None:
        out.append(
            "  VO2_d: unavailable without a profile (no store to read the falls against); the v5 "
            "prior is the\n  placeholder."
        )
    else:
        pop = population_vo2(ctx.constants, ctx.profile)
        out.append(
            f"  VO2_d {vo2['vo2']:.0f} mL/min, 80 % {vo2['low']:.0f}-{vo2['high']:.0f}, from "
            f"{vo2['holds']} holds (effective {vo2['effective']:.1f}); population {pop:.0f} "
            "mL/min.\n  The finger reading lags the hold, so spo2End understates the fall and "
            "VO2_d leans low."
        )
    dist = Distance(fin)
    rows = capacity_rows(fin, dist)
    out.append(
        f"  Own 90th percentiles: fall {dist.fall90:.0f} points, struggle share {dist.share90:.2f}."
        if dist.fall90 is not None and dist.share90 is not None
        else "  Own 90th percentiles: not measurable."
    )
    out.append(f"  {'session':<12}{'volume':<8}{'style':<8}{'held':>6}{'d':>6}{'weight':>8}  row")
    for r in rows:
        st = r["session"]["steps"][r["stepIndex"]]
        out.append(
            f"  {r['session']['startedAt'][:10]:<12}{r['mode']:<8}{st.get('style', ''):<8}"
            f"{_clock(r['actual']):>6}{r['d']:>6.2f}{r['weight']:>8.2f}  "
            f"{'LIMIT' if not r['censored'] else 'lower bound'}"
        )
    out.append("")
    cmp = compare_rules(fin, cfg, ctx, tz)
    out.append("  Forward chain, scored on each next session's holds with d ≥ 0.8:")
    for rule in ("v2", "v5"):
        c = cmp[rule]
        cov = "-" if c["coverage"] is None else f"{c['coverage']:.0%}"
        out.append(
            f"    {'§10 v2' if rule == 'v2' else '§18 v5':<8} scored {c['scored']:>3}  log score "
            f"{c['logscore']:>8.2f}  80% coverage {cov}"
        )
    n = cmp["v5"]["scored"]
    if cmp["admitted"]:
        verdict = "v5 beats v2 on log score with coverage within 70-90 %: §18 ships."
    elif not cmp["decidable"]:
        verdict = (
            f"not decidable with {n} scored holds (coverage 70-90 % cannot show): §10 v2 stays "
            "the shipped rule."
        )
    else:
        verdict = "v5 does not win: §10 v2 stays the shipped rule."
    out.append(f"    {verdict}")
    out.append("")
    _, own = ctx.physiology(fin)
    model = fit(fin, cfg, own, tz, rows=rows, packing=ctx.packing)
    est = estimates(model, fin, floors=floors, tz=tz, rule="v5")
    out.append(
        f"  v5 model on all data (σ = {math.sqrt(model['sigma2']):.3f} log units, df "
        f"{2 * model['an']:.1f}):"
    )
    for v in MODES:
        e = est[v]
        c = e["counts"]
        prior = model["priors"].get(v)
        pr = (
            "none"
            if prior is None
            else f"{prior[2]} N(ln {_clock(math.exp(prior[0]))}, {prior[1]:.2f}²)"
        )
        out.append(
            f"    {v:<5} prior {pr}; {c['rows']} rows, {c['uncensored']} at the limit, effective "
            f"{c['effective']:.1f}"
        )
        for kind in ("level", "pb"):
            r = e[kind]
            if r["estimate"] is None:
                out.append(f"      {kind:<6}no posterior; {r['gate'].summary}")
                continue
            lo, mid, hi = r["estimate"]
            g = r["gate"]
            out.append(
                f"      {kind:<6}{_clock(lo)}-{_clock(hi)} (centre {_clock(mid)}), narrowing "
                f"{r['narrowing']:.2f}, "
                + ("gate ENOUGH" if g.enough else f"gate waiting: {g.summary}")
            )
    out.append("")
    return {"compare": cmp, "model": model, "estimates": est, "vo2": vo2, "rows": rows}
