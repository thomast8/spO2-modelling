"""Contract §18 and §19 (v5): capacity from every hold. The limit is read from the trace, not
the flag.

- End saturation (§19): the post-hold minimum (spo2Low) when the recovery trace turned or ran
  60 s or more, else spo2End, flagged lag-uncorrected. `lag=False` reads spo2End throughout.
- Per-diver consumption VO2_d from every performed hold of known volume with ≥ 60 s of trusted
  SpO₂: the usable store at its volume times the share of the usable desaturation its fall reached
  (spo2Baseline → end saturation against the population end saturation), over its duration.
  Log-normal prior N(ln VO2_pop, 0.25²); weighted mean of log consumption (weights = trusted
  seconds), Student-t interval over the between-hold spread.
- Distance to the limit d ∈ [0, 1]: the larger of the fall (spo2Baseline − end saturation) over
  the diver's 90th-percentile fall and the struggle share over its 90th percentile, both per lung
  volume (pooled when a volume has fewer than 5 holds with a trace); 0 when neither is measurable.
  Descriptive, never shown.
- Capacity rows: every eligible performed hold with a trace, weight d², a lower bound unless
  d ≥ 0.8. §19: holdIndex and logRestBefore always ride as covariates, ridge-fitted on the limit
  rows; level and PB are read at the rested context.
- `predictive(rule, ...)` is the seam for §19's neutral scorer; no rule comparison lives here.
"""

import math
from collections import Counter
from datetime import timedelta

import numpy as np
from archive import HARNESS_TIME_ZONE, finished, is_performed_hold
from features import (
    cached_markers,
    eligible_for_model,
    hold_context,
    lung_mode_of,
    oxygen_readings,
    past_limit_hold,
    stamped_trace,
)
from fit_hold_model import (
    MODES,
    base_config,
    fit,
    gated_axes,
    pb_inputs,
    with_coefficients,
)
from fit_hold_model import predictive as model_predictive
from fit_oxygen_store import physiology_prior, stores, volumes_for
from numerics import Z90, t90
from packing import NONE as PACKING_NONE

TRACE_SECONDS = 60
LIMIT = 0.8
VO2_PRIOR_SD = 0.25
VOLUME_HOLDS = 5  # holds with a trace a volume needs for its own percentiles
REST_INPUTS = ("holdIndex", "logRestBefore")


def trace_seconds(step):
    """Seconds with a trusted SpO₂ reading inside the hold."""
    return len(oxygen_readings(step))


def has_trace(step):
    return is_performed_hold(step) and trace_seconds(step) >= TRACE_SECONDS


def end_saturation(session, index, lag=True):
    """(end SpO₂, lag-corrected) for a performed hold. §19: the finger shows the low 25 to 40 s
    late, so the post-hold minimum (spo2Low) stands in for the end when the recovery trace turned
    (an onset exists) or ran 60 s or more; otherwise spo2End, flagged lag-uncorrected."""
    m = cached_markers(session)[index]
    if lag and m["spo2Low"] is not None:
        trace = stamped_trace(index, session)
        if trace is not None and (
            trace["onsetSeconds"] is not None or trace["traceSeconds"] >= TRACE_SECONDS
        ):
            return m["spo2Low"], True
    return m["spo2End"], False


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


def fit_vo2(sessions, constants, profile, spiro, ratios, packing=None, lag=True):
    """VO2_d with its interval and the holds behind it (how many lag-uncorrected), or None
    without a profile."""
    if (
        not profile
        or profile.get("weightKg") is None
        or volumes_for(profile, spiro, ratios) is None
    ):
        return None
    end_pop = 100 * constants["arterialSaturationEnd"]
    values, weights, uncorrected = [], [], 0
    for s in finished(sessions):
        markers = cached_markers(s)
        for i, st in enumerate(s["steps"]):
            if not (has_trace(st) and eligible_for_model(s, st)):
                continue
            store = hold_store(st, constants, profile, spiro, ratios, packing)
            end, corrected = end_saturation(s, i, lag)
            c = hold_consumption(
                store, markers[i]["spo2Baseline"], end, end_pop, st["actualDuration"]
            )
            if c is not None:
                values.append(c)
                weights.append(trace_seconds(st))
                uncorrected += not corrected
    vo2, lo, hi, n_eff = combine_vo2(population_vo2(constants, profile), values, weights)
    return {
        "vo2": vo2,
        "low": lo,
        "high": hi,
        "holds": len(values),
        "effective": n_eff,
        "uncorrected": uncorrected,
    }


# ---------------------------------------------------------------- distance to the limit


def _p90(xs):
    return float(np.percentile(xs, 90)) if xs else None


class Distance:
    """The diver's own 90th-percentile fall and struggle share per lung volume, from the given
    sessions; a volume with fewer than 5 holds with a trace uses the pooled pair (`pooled_for`)."""

    def __init__(self, sessions, lag=True):
        self.lag = lag
        falls, shares, traced = {}, {}, Counter()
        for s in finished(sessions):
            for i, st in enumerate(s["steps"]):
                if not is_performed_hold(st):
                    continue
                mode = lung_mode_of(st)
                traced[mode] += has_trace(st)
                fall, share = self.measures(s, i)
                for key in (None, mode) if mode else (None,):
                    if fall is not None:
                        falls.setdefault(key, []).append(fall)
                    if share is not None:
                        shares.setdefault(key, []).append(share)
        self.pooled = (_p90(falls.get(None)), _p90(shares.get(None)))
        self.traced = {v: traced[v] for v in MODES}
        self.pooled_for = [v for v in MODES if traced[v] < VOLUME_HOLDS]
        self.scales = {
            v: self.pooled if v in self.pooled_for else (_p90(falls.get(v)), _p90(shares.get(v)))
            for v in MODES
        }

    def measures(self, session, index):
        """(SpO₂ fall to the end saturation, struggle share), each None when not measurable."""
        st = session["steps"][index]
        m = cached_markers(session)[index]
        fall = None
        if has_trace(st) and m["spo2Baseline"] is not None:
            end, _ = end_saturation(session, index, self.lag)
            if end is not None:
                fall = max(m["spo2Baseline"] - end, 0.0)
        share = None
        if m["firstContractionSeconds"] is not None and st["actualDuration"] > 0:
            share = (
                max(st["actualDuration"] - m["firstContractionSeconds"], 0) / st["actualDuration"]
            )
        return fall, share

    def d(self, session, index):
        if index == past_limit_hold(session):
            return 1.0  # a reported blackout/LMC/samba: that hold went past the limit
        fall90, share90 = self.scales.get(lung_mode_of(session["steps"][index]), self.pooled)
        fall, share = self.measures(session, index)
        parts = []
        if fall is not None and fall90:
            parts.append(min(fall / fall90, 1.0))
        if share is not None and share90:
            parts.append(min(share / share90, 1.0))
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


# ---------------------------------------------------------------- the v5 model


class Context:
    """The diver-side inputs both rules need: constants, profile, spirometry, ratios, packing.
    Without a profile (or constants) there is no physiology prior and no VO2_d."""

    def __init__(self, constants, profile, spiro, ratios, packing=PACKING_NONE):
        self.constants, self.profile, self.spiro, self.ratios = constants, profile, spiro, ratios
        self.packing = packing

    @property
    def has_profile(self):
        return bool(self.profile) and self.constants is not None

    def population(self):
        """§12 physiology prior at the population VO2 (v2's prior)."""
        if not self.has_profile:
            return None
        return physiology_prior(self.constants, self.profile, self.spiro, self.ratios)

    def vo2(self, sessions, lag=True):
        if not self.has_profile:
            return None
        return fit_vo2(
            sessions, self.constants, self.profile, self.spiro, self.ratios, self.packing, lag
        )

    def prior_at(self, vo2):
        """§18 physiology prior at a fitted VO2_d (v5's prior)."""
        if vo2 is None:
            return None
        return physiology_prior(
            self.constants, self.profile, self.spiro, self.ratios, vo2_ml_min=vo2["vo2"]
        )


NO_DIVER = Context(None, None, None, None)


def v5_fit(sessions, cfg, ctx=NO_DIVER, tz=HARNESS_TIME_ZONE, lag=True, rest=True, drop=None):
    """The v5 hold model on `sessions`: (model, rows, distance, vo2); model is None with no
    finished session. `lag` reads the §19 lag-adjusted end saturation for d and VO2_d; `rest`
    adds holdIndex and logRestBefore as covariates, ridge-fitted on the limit rows (prior sd 0.2
    per input sd) with their covariance in the predictive; `drop(row)` leaves rows out (the
    report's diagnostics only)."""
    fin = finished(sessions)
    dist = Distance(fin, lag)
    rows = [r for r in capacity_rows(fin, dist) if not (drop and drop(r))]
    vo2 = ctx.vo2(fin, lag)
    if not fin:
        return None, rows, dist, vo2
    c = dict(cfg, rowRule="v5")
    if rest:
        c["covariates"] = [*c["covariates"], *(x for x in REST_INPUTS if x not in c["covariates"])]
        c = with_coefficients(fin, c, tz, rows=rows, packing=ctx.packing)
    model = fit(fin, c, ctx.prior_at(vo2), tz, rows=rows, packing=ctx.packing)
    return model, rows, dist, vo2


def target_context(session, index, sessions_before, tz=HARNESS_TIME_ZONE):
    """`predictive`'s context for a real hold: its §9 context plus its style and pack count."""
    step = session["steps"][index]
    return {
        **hold_context(session, index, sessions_before, tz),
        "style": step.get("style"),
        "packCount": step.get("packCount"),
    }


def predictive(rule, sessions_before, volume, context, diver=NO_DIVER, cfg=None, tz=None):
    """Predictive of ln(capacity) for one new hold, for §19's neutral scorer.

    rule: "v2" (§10 rows: per-slot longest holds and max attempts, diver-ended ones uncensored,
        population physiology prior) or "v5" (§18 capacity rows with §19's lag-adjusted fall,
        per-volume percentiles and rest-before term, physiology prior at VO2_d).
    sessions_before: what the rule may learn from, all started before the target hold's session
        (the caller fixes the split; unfinished sessions are ignored).
    volume: the target hold's lung volume, "full", "frc" or "rv".
    context: the target hold as a dict. `holdIndex` and `logRestBefore` (§9, as hold_context
        gives them) feed v5's rest term, and any other §9 field a covariate admitted in `cfg`
        needs; `style` and `packCount` move a packed hold by its §17 offset. A missing field sits
        at the training centre. target_context(session, index, sessions_before) builds it.
    diver: Context (constants, profile, spirometry, ratios, packing). NO_DIVER, as in the public
        run, means both rules use the placeholder prior and no packing offset.
    cfg: the hold-model config (half-life, inputs admitted for v2); base_config() when None.
    tz: §9's time zone, the harness's when None.

    Returns (mu, sd, df) with ln T = mu + sd · t_df, or None when the rule has no posterior at
    that volume. It fits from scratch on every call."""
    tz = tz or HARNESS_TIME_ZONE
    cfg = cfg or base_config()
    fin = finished(sessions_before)
    if not fin:
        return None
    if rule == "v5":
        model = v5_fit(fin, cfg, diver, tz)[0]
    elif rule == "v2":
        model = fit(fin, cfg, diver.population(), tz, packing=diver.packing)
    else:
        raise ValueError(f"unknown row rule {rule!r}")
    c = model["cfg"]
    x = [context.get(k) for k in c["covariates"]]
    a = []
    if c["axes"]:
        ax = gated_axes(fin, model["t_now"] + timedelta(seconds=1), c["halfLifeDays"])
        a = [ax[k] for k in c["axes"]]
    p = model_predictive(model, volume, x, a)
    if p is None:
        return None
    mu, sd, df = p
    off = diver.packing.offset(
        {"style": context.get("style"), "packCount": context.get("packCount")}
    )
    return mu + off, sd, df


# ---------------------------------------------------------------- report


def _clock(t):
    return f"{int(t) // 60}:{round(t) % 60:02d}"


def _input(name, value):
    if value is None:
        return f"{name} -"
    if name == "logRestBefore":
        return f"rest before {_clock(math.exp(value) - 1)}"
    return f"{name} {value:g}"


def _vo2_line(vo2, label):
    return (
        f"    {label:<26}{vo2['vo2']:.0f} mL/min, 80 % {vo2['low']:.0f}-{vo2['high']:.0f}, from "
        f"{vo2['holds']} holds (effective {vo2['effective']:.1f}, {vo2['uncorrected']} at spo2End)"
    )


def _brief(model, est, label, out):
    """One line per volume: level and PB centres with their intervals."""
    out.append(f"    {label}:")
    for v in MODES:
        e = est[v]
        if e["level"]["estimate"] is None:
            out.append(f"      {v:<5} no posterior")
            continue
        lo, mid, hi = e["level"]["estimate"]
        plo, pmid, phi = e["pb"]["estimate"]
        out.append(
            f"      {v:<5} level {_clock(mid)} ({_clock(lo)}-{_clock(hi)})  pb {_clock(pmid)} "
            f"({_clock(plo)}-{_clock(phi)})  {model['counts'][v]['uncensored']} at the limit"
        )


def _level_centre(model, sessions, tz):
    from fit_hold_model import estimates

    if model is None or "full" not in model["post"]:
        return None
    return estimates(model, sessions, tz=tz, rule="v5")["full"]["level"]["estimate"][1]


CO2_TABLE_DAY = "2026-09-08"  # the CO₂ table whose 2:30 holds §19's rest term is aimed at


def report(sessions, out, cfg, ctx, tz, floors=None):
    """Section 4c: end saturation, VO2, d per hold, the v5 model with and without §19's lag
    adjustment and rest term, the rested context, and the CO₂-table check."""
    from fit_hold_model import estimates

    fin = finished(sessions)
    out.append("4c. CAPACITY FROM EVERY HOLD (contract §18 rows with §19's fall and rest term, v5)")
    out.append("-" * 82)
    out.append("")
    out.append(
        "Every hold with 60 s or more of trusted SpO₂ is a row, weighted by d² (how close the "
        "trace came to\nthe diver's own limit at that lung volume) and a lower bound unless "
        "d ≥ 0.8, whoever ended it. The\nfall runs to the post-hold minimum when the recovery "
        "trace turned or ran 60 s (the finger lags the\nhold), else to spo2End. holdIndex and "
        "logRestBefore ride on every row, fitted on the limit rows, and\nthe level and PB are read "
        "at the rested context."
    )
    out.append("")
    model, rows, dist, vo2 = v5_fit(fin, cfg, ctx, tz)
    plain_dist = Distance(fin, lag=False)
    flags = [end_saturation(r["session"], r["stepIndex"])[1] for r in rows]
    out.append(
        f"  End saturation: {sum(flags)} of {len(rows)} rows read at the post-hold minimum, "
        f"{len(rows) - sum(flags)} lag-uncorrected at spo2End\n  (their recovery trace neither "
        "turned nor ran 60 s)."
    )
    if vo2 is None:
        out.append(
            "  VO2_d: unavailable without a profile (no store to read the falls against); the v5 "
            "prior is the\n  placeholder."
        )
    else:
        out.append(f"  VO2_d (population {population_vo2(ctx.constants, ctx.profile):.0f} mL/min):")
        out.append(_vo2_line(vo2, "lag-adjusted (§19)"))
        out.append(_vo2_line(ctx.vo2(fin, lag=False), "at spo2End (§18)"))
    out.append("  Own 90th percentiles (fall in points, struggle share):")
    for v in MODES:
        f90, s90 = dist.scales[v]
        scale = (
            "not measurable"
            if f90 is None and s90 is None
            else f"fall {'-' if f90 is None else f'{f90:.0f}'}, share "
            f"{'-' if s90 is None else f'{s90:.2f}'}"
        )
        pooled = f", fewer than {VOLUME_HOLDS} so POOLED" if v in dist.pooled_for else ""
        out.append(f"    {v:<5} {scale} from {dist.traced[v]} holds with a trace{pooled}")
    out.append(
        f"  {'session':<12}{'volume':<8}{'style':<8}{'held':>6}{'d':>6}{'d@end':>7}{'weight':>8}"
        "  row"
    )
    for r, corrected in zip(rows, flags, strict=True):
        st = r["session"]["steps"][r["stepIndex"]]
        out.append(
            f"  {r['session']['startedAt'][:10]:<12}{r['mode']:<8}{st.get('style', ''):<8}"
            f"{_clock(r['actual']):>6}{r['d']:>6.2f}"
            f"{plain_dist.d(r['session'], r['stepIndex']):>7.2f}{r['weight']:>8.2f}  "
            f"{'LIMIT' if not r['censored'] else 'lower bound'}"
            + ("" if corrected else "  (lag-uncorrected)")
        )
    out.append(
        "  d@end is d with the fall to spo2End and the same per-volume percentiles rule, for "
        "comparison."
    )
    out.append("")
    est = estimates(model, fin, floors=floors, tz=tz, rule="v5")
    out.append(
        f"  v5 model on all data (σ = {math.sqrt(model['sigma2']):.3f} log units, df "
        f"{2 * model['an']:.1f}):"
    )
    c = model["cfg"]
    for j, name in enumerate(c["covariates"]):
        out.append(
            f"    {name:<16}coefficient {c['beta'][j]:+.3f} per unit (sd "
            f"{math.sqrt(c['betaCov'][j][j]):.3f}), centre {model['xbar'][j]:.2f}"
        )
    for v in MODES:
        e = est[v]
        cn = e["counts"]
        prior = model["priors"].get(v)
        pr = (
            "none"
            if prior is None
            else f"{prior[2]} N(ln {_clock(math.exp(prior[0]))}, {prior[1]:.2f}²)"
        )
        out.append(
            f"    {v:<5} prior {pr}; {cn['rows']} rows, {cn['uncensored']} at the limit, effective "
            f"{cn['effective']:.1f}"
        )
        if cn["uncensored"]:
            x, _ = pb_inputs(model, fin, v, tz)
            rested = ", ".join(_input(k, xv) for k, xv in zip(c["covariates"], x, strict=True))
            out.append(f"      rested context (3 longest limit rows): {rested or 'no inputs'}")
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
    out.append("  The same v5 fit with one §19 piece taken out:")
    for label, kw in (
        ("fall to spo2End (no lag adjustment), rest term on", {"lag": False}),
        ("lag-adjusted fall, no rest term", {"rest": False}),
    ):
        m = v5_fit(fin, cfg, ctx, tz, **kw)[0]
        _brief(m, estimates(m, fin, tz=tz, rule="v5"), label, out)
    table = [r for r in rows if r["session"]["startedAt"].startswith(CO2_TABLE_DAY)]
    if any(r["mode"] == "full" and not r["censored"] for r in table):
        n = sum(1 for r in table if r["mode"] == "full" and not r["censored"])
        out.append(
            f"  The {CO2_TABLE_DAY} CO₂ table ({n} full-lung holds at the limit) on the full-lung "
            "level centre:"
        )
        for label, rest in (("no rest term", False), ("rest term", True)):
            with_ = _level_centre(v5_fit(fin, cfg, ctx, tz, rest=rest)[0], fin, tz)
            without = _level_centre(
                v5_fit(
                    fin,
                    cfg,
                    ctx,
                    tz,
                    rest=rest,
                    drop=lambda r: r["session"]["startedAt"].startswith(CO2_TABLE_DAY),
                )[0],
                fin,
                tz,
            )
            if with_ is None or without is None:
                out.append(f"    {label:<14}no full-lung posterior")
                continue
            out.append(
                f"    {label:<14}with it {_clock(with_)}, without it {_clock(without)} "
                f"({with_ - without:+.0f} s)"
            )
    out.append(
        "  Which row rule ships is left to §19's neutral scorer (capacity.predictive); "
        "nothing here\n  compares v2 with v5."
    )
    out.append("")
    return {"model": model, "estimates": est, "vo2": vo2, "rows": rows, "distance": dist}
