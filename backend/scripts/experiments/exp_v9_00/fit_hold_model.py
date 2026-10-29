"""Contract §10 (v2): the hold model, its gates, and the forward chain that chooses its inputs.

On device only μ_v, V_v and σ² are learned; the harness chooses the half-life, admits covariates
(§9) and axes (§8), and fixes their coefficients. Everything here follows the contract except the
points marked HARNESS, which the contract leaves open.
"""

import math
from datetime import timedelta

import numpy as np
from archive import (
    HARNESS_TIME_ZONE,
    is_imported_from_stamina,
    is_performed_hold,
    looks_like_max_attempt,
    started,
    symptoms,
)
from archive import (
    finished as finished_sessions,
)
from evidence import DEFAULT_FLOORS, evaluate
from features import (
    AXES,
    AXIS_UNIT,
    CONTEXT_FIELDS,
    diver_axes,
    hold_context,
    hold_model_rows,
    lung_mode_of,
)
from numerics import Z90, log_survival, median, mills, newton1d, t90, weighted_median
from scipy import stats

MODES = ("full", "frc", "rv")
# HARNESS: σ² prior, inverse-gamma with mean 0.15² (a max attempt varies by about ±15 % from
# day to day). Judgement, not data: the export has two uncensored rows to learn it from.
SIGMA_A0, SIGMA_B0 = 3.0, 2 * 0.15**2
PLACEHOLDER_LOG_SD = 0.5
PLACEHOLDER_RATIOS = {"full": 1.0, "frc": 0.5, "rv": 1 / 3}
HALF_LIVES = (None, 180.0, 365.0)
BURN_IN = 5


def base_config(half_life=None):
    return {
        "halfLifeDays": half_life,
        "covariates": [],
        "beta": [],
        "betaCov": [],
        "axes": [],
        "gamma": [],
        "gammaCov": [],
        "sigmaA0": SIGMA_A0,
        "sigmaB0": SIGMA_B0,
        "placeholderLogSD": PLACEHOLDER_LOG_SD,
        "placeholderRatios": PLACEHOLDER_RATIOS,
        "personalisePhysiology": False,
        "rowRule": "v2",
    }


def model_rows(sessions):
    return [r for s in sessions for r in hold_model_rows(s)]


def weight(t_now, t, h):
    return 1.0 if h is None else 2 ** (-(t_now - t).total_seconds() / 86400 / h)


def gated_axes(sessions, as_of, h, floors=None):
    """Axis values whose gate is enough (previouslyShown false); a waiting axis is missing."""
    f = (floors or DEFAULT_FLOORS)["axis"]["enough"]
    return {
        k: v["value"] if v["count"] >= f[AXIS_UNIT[k]] else None
        for k, v in diver_axes(sessions, as_of, h).items()
    }


def inputs(row, sessions, cfg, tz):
    """(x, a) for a row: admitted covariates and admitted axes, None where missing."""
    x = []
    if cfg["covariates"]:
        ctx = hold_context(row["session"], row["stepIndex"], sessions, tz)
        x = [ctx[c] for c in cfg["covariates"]]
    a = []
    if cfg["axes"]:
        ax = gated_axes(sessions, started(row["session"]), cfg["halfLifeDays"])
        a = [ax[k] for k in cfg["axes"]]
    return x, a


def centres(values, weights):
    """Weighted median per column over rows where the value is present."""
    out = []
    for j in range(len(values[0]) if values else 0):
        pairs = [(v[j], w) for v, w in zip(values, weights, strict=True) if v[j] is not None]
        out.append(weighted_median([p[0] for p in pairs], [p[1] for p in pairs]) if pairs else 0.0)
    return out


def deviation(v, c):
    return [0.0 if x is None else x - m for x, m in zip(v, c, strict=True)]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


def quad(d, cov):
    return float(np.array(d) @ np.array(cov) @ np.array(d)) if d else 0.0


def fit(
    sessions,
    cfg,
    physiology=None,
    tz=HARNESS_TIME_ZONE,
    rows=None,
    t_now=None,
    priors=None,
    packing=None,
):
    """`Analytics.holdModel` before gating: per-volume posterior and the pooled σ².

    `priors` overrides the priors on μ_v (diagnostics that must hold the prior fixed);
    `packing` (packing.Packing) removes what a packed hold's extra air should buy (§17)."""
    fixed_priors = priors
    finished = finished_sessions(sessions)
    rows = model_rows(finished) if rows is None else rows
    t_now = max(started(s) for s in finished) if t_now is None else t_now
    h = cfg["halfLifeDays"]
    # A row may carry its own weight (§18: d²); recency multiplies it.
    w = [weight(t_now, started(r["session"]), h) * r.get("weight", 1.0) for r in rows]
    xa = [inputs(r, finished, cfg, tz) for r in rows]
    xbar = centres([x for x, _ in xa], w)
    abar = centres([a for _, a in xa], w)
    yp = [
        r["y"]
        - dot(cfg["beta"], deviation(x, xbar))
        - dot(cfg["gamma"], deviation(a, abar))
        - (packing.offset(r["session"]["steps"][r["stepIndex"]]) if packing else 0.0)
        for r, (x, a) in zip(rows, xa, strict=True)
    ]
    a0, b0 = cfg["sigmaA0"], cfg["sigmaB0"]
    full_rows = [r["actual"] for r in rows if r["mode"] == "full"]
    priors = {}
    for v in MODES:
        if fixed_priors is not None:
            if v in fixed_priors:
                priors[v] = fixed_priors[v]
        elif physiology and v in physiology:
            priors[v] = (*physiology[v], "physiology")
        elif full_rows:
            m0 = math.log(max(full_rows) * cfg["placeholderRatios"][v])
            priors[v] = (m0, cfg["placeholderLogSD"], "placeholder")
    by_mode = {
        v: [(yp[i], w[i], rows[i]["censored"]) for i in range(len(rows)) if rows[i]["mode"] == v]
        for v in MODES
    }
    sigma2, an, failed, post = b0 / (a0 - 1), a0, set(), {}
    for _ in range(6):
        for v, (m0, s0, _) in priors.items():
            if v in failed:
                continue
            data = by_mode[v]
            sd = math.sqrt(sigma2)

            def fgh(mu, data=data, m0=m0, s0=s0, sd=sd, sigma2=sigma2):
                f = -((mu - m0) ** 2) / (2 * s0 * s0)
                g = -(mu - m0) / (s0 * s0)
                h = -1 / (s0 * s0)
                for y, wi, cens in data:
                    if cens:
                        z = (y - mu) / sd
                        lam = mills(z)
                        f += wi * log_survival(z)
                        g += wi * lam / sd
                        h -= wi * lam * (lam - z) / sigma2
                    else:
                        f -= wi * (y - mu) ** 2 / (2 * sigma2)
                        g += wi * (y - mu) / sigma2
                        h -= wi / sigma2
                return f, g, h

            res = newton1d(fgh, m0)
            if res is None:
                failed.add(v)
                post.pop(v, None)
            else:
                post[v] = (res[0], -1 / res[1])
        su = sum(wi for v in post for _, wi, c in by_mode[v] if not c)
        ss = sum(wi * (y - post[v][0]) ** 2 for v in post for y, wi, c in by_mode[v] if not c)
        an, bn = a0 + su / 2, b0 + ss / 2
        sigma2 = bn / (an - 1)
    return {
        "rows": rows,
        "weights": w,
        "xbar": xbar,
        "abar": abar,
        "priors": priors,
        "post": post,
        "failed": failed,
        "sigma2": sigma2,
        "an": an,
        "t_now": t_now,
        "cfg": cfg,
        "counts": {
            v: {
                "rows": sum(1 for r in rows if r["mode"] == v),
                "uncensored": sum(1 for r in rows if r["mode"] == v and not r["censored"]),
                "censored": sum(1 for r in rows if r["mode"] == v and r["censored"]),
                "effective": sum(wi for r, wi in zip(rows, w, strict=True) if r["mode"] == v),
            }
            for v in MODES
        },
        "xa": xa,
    }


def predictive(model, v, x, a):
    """(loc, scale, df) of ln T for one new hold at inputs (x, a), or None without a posterior."""
    if v not in model["post"]:
        return None
    mu, var = model["post"][v]
    cfg = model["cfg"]
    dx, da = deviation(x, model["xbar"]), deviation(a, model["abar"])
    loc = mu + dot(cfg["beta"], dx) + dot(cfg["gamma"], da)
    total = var + model["sigma2"] + quad(dx, cfg["betaCov"]) + quad(da, cfg["gammaCov"])
    return loc, math.sqrt(total), 2 * model["an"]


def pb_inputs(model, sessions, v, tz=HARNESS_TIME_ZONE):
    """x_PB: componentwise median over the volume's 3 longest uncensored rows; a_PB as of
    tNow+1s."""
    cfg = model["cfg"]
    idx = [i for i, r in enumerate(model["rows"]) if r["mode"] == v and not r["censored"]]
    idx = sorted(idx, key=lambda i: -model["rows"][i]["actual"])[:3]
    x = []
    for j in range(len(cfg["covariates"])):
        vals = [model["xa"][i][0][j] for i in idx if model["xa"][i][0][j] is not None]
        x.append(median(vals) if vals else None)
    a = []
    if cfg["axes"]:
        ax = gated_axes(sessions, model["t_now"] + timedelta(seconds=1), cfg["halfLifeDays"])
        a = [ax[k] for k in cfg["axes"]]
    return x, a


def rows_to_reach(model, v, kind, rho):
    m0, s0, _ = model["priors"][v]
    a0, b0 = model["cfg"]["sigmaA0"], model["cfg"]["sigmaB0"]
    an, sigma2 = model["an"], model["sigma2"]
    n = model["counts"][v]["uncensored"]
    prior_pb = 2 * t90(2 * a0) * math.sqrt(s0 * s0 + b0 / (a0 - 1))
    for m in range(201):
        post_var = 1 / (1 / (s0 * s0) + (n + m) / sigma2)
        if kind == "pb":
            ok = 2 * t90(2 * an + m) * math.sqrt(post_var + sigma2) <= (1 - rho) * prior_pb
        else:  # HARNESS: the contract gives rowsToReach for pb only; level uses its own width
            ok = t90(2 * an + m) * math.sqrt(post_var) <= (1 - rho) * Z90 * s0
        if ok:
            return m
    return None


def estimates(
    model, sessions, previously_shown=frozenset(), floors=None, tz=HARNESS_TIME_ZONE, rule="v2"
):
    """Per volume: level and pb estimates with their gates (`HoldModelEstimate`). `rule` "v5"
    gates on holds with a trace and their effective weight (§18) instead of max attempts."""
    floors = floors or DEFAULT_FLOORS
    a0, b0 = model["cfg"]["sigmaA0"], model["cfg"]["sigmaB0"]
    out = {}
    full = model["counts"]["full"]
    full_have = full["uncensored"] if rule == "v2" else full["rows"]
    full_unit = ("maxAttempts", "full") if rule == "v2" else ("holdsWithSpO2", "full")
    for v in MODES:
        c = model["counts"][v]
        res = {"counts": c, "prior": model["priors"].get(v)}
        for kind in ("pb", "level"):
            key = f"{kind}.{v}"
            shown = key in previously_shown
            if rule == "v2":
                tf, counts = floors[kind], {("maxAttempts", v): c["uncensored"]}
            else:
                tf = floors[f"{kind}V5"]
                counts = {("holdsWithSpO2", v): c["rows"], ("effectiveHolds", v): c["effective"]}
            if v not in model["post"]:
                gate = evaluate(key, tf, counts, None, shown)
                if v not in model["priors"]:
                    gate.missing.append((full_unit, full_have, full_have + 1, "floor"))
                else:
                    gate.missing.append((("narrowing",), 0, 0, "narrowing"))
                gate.enough = False
                gate.counts["censored"] = c["censored"]
                res[kind] = {"estimate": None, "gate": gate}
                continue
            mu, var = model["post"][v]
            m0, s0, _ = model["priors"][v]
            an = model["an"]
            if kind == "level":
                half = t90(2 * an) * math.sqrt(var)
                est = (math.exp(mu - half), math.exp(mu), math.exp(mu + half))
                narrowing = 1 - half / (Z90 * s0)
            else:
                x, a = pb_inputs(model, sessions, v, tz)
                loc, scale, df = predictive(model, v, x, a)
                half = t90(df) * scale
                est = (math.exp(loc - half), math.exp(loc), math.exp(loc + half))
                prior_w = 2 * t90(2 * a0) * math.sqrt(s0 * s0 + b0 / (a0 - 1))
                narrowing = 1 - 2 * half / prior_w
            rows = rows_to_reach(model, v, kind, tf["rhoEnough"])
            gate = evaluate(key, tf, counts, (narrowing, rows), shown)
            gate.counts["censored"] = c["censored"]
            res[kind] = {"estimate": est, "gate": gate, "narrowing": narrowing}
        out[v] = res
    return out


# ---------------------------------------------------------------- baselines


def running_bests(sessions):
    """The best so far per slot from every earlier performed hold (today's bests, imports as
    full)."""
    best = {}
    for s in sessions:
        for st in s["steps"]:
            mode = lung_mode_of(st)
            if is_performed_hold(st) and mode:
                best[mode] = max(best.get(mode, 0), st["actualDuration"])
    return best


def open_hold_estimate(step, session, earlier, maxholds, runs=5):
    """`Analytics.openHoldEstimates` for one open max hold (falls back to the best)."""
    held = []
    for s in sorted(earlier, key=started, reverse=True):
        if s.get("planID") is None or s.get("planID") != session.get("planID"):
            continue
        for st in s["steps"]:
            if (
                st["id"] == step["id"]
                and st.get("style") == step.get("style")
                and st["kindName"] == "hold"
                and st.get("plannedDuration") is None
                and is_performed_hold(st)
                and st["outcome"] != {"type": "endedEarly", "reason": "aborted"}
            ):
                held.append(st["actualDuration"])
                break
        if len(held) >= runs:
            break
    if held:
        s_ = sorted(held)
        mid = len(s_) // 2
        return s_[mid] if len(s_) % 2 else (s_[mid - 1] + s_[mid]) // 2
    mode = lung_mode_of(step)
    return maxholds.get(mode) or maxholds.get("full")


def baseline_points(row, earlier):
    """{name: predicted seconds} for the point baselines."""
    step = row["session"]["steps"][row["stepIndex"]]
    best = running_bests(earlier)
    out = {}
    pb = best.get(row["mode"]) or best.get("full")
    if pb:
        out["PB-fallback"] = pb
    comp = (
        open_hold_estimate(step, row["session"], earlier, best)
        if step.get("plannedDuration") is None
        else step["plannedDuration"]
    )
    if comp:
        out["comparable-history"] = comp
    return out


# ---------------------------------------------------------------- forward chain


def score(loc, scale, df, row):
    z = (row["y"] - loc) / scale
    if row["censored"]:
        return float(stats.t.logsf(z, df))
    return float(stats.t.logpdf(z, df) - math.log(scale))


def chain(
    sessions,
    cfg,
    physiology=None,
    tz=HARNESS_TIME_ZONE,
    burn_in=BURN_IN,
    beta_fn=None,
    packing=None,
):
    """Forward chain by session date. Each session with rows, after the first `burn_in` of them, is
    predicted from every finished session before it. `beta_fn(train_model_inputs)` re-estimates
    coefficients per step for a candidate input. Returns scored predictions."""
    fin = finished_sessions(sessions)
    with_rows = [s for s in fin if hold_model_rows(s)]
    preds = []
    for s in with_rows[burn_in:]:
        train = [t for t in fin if started(t) < started(s)]
        c = dict(cfg)
        if beta_fn:
            c = beta_fn(train, c)
        model = fit(train, c, physiology, tz, packing=packing)
        for row in hold_model_rows(s):
            x, a = inputs(row, train + [s], c, tz)
            p = predictive(model, row["mode"], x, a)
            if p is None:
                preds.append({"row": row, "pred": None})
                continue
            loc, scale, df = p
            half = t90(df) * scale
            preds.append(
                {
                    "row": row,
                    "pred": p,
                    "logscore": score(loc, scale, df, row),
                    "covered": None if row["censored"] else abs(row["y"] - loc) <= half,
                    "abs_err": None if row["censored"] else abs(math.exp(loc) - row["actual"]),
                    "baselines": baseline_points(row, train),
                }
            )
    return preds


def summarise(preds, name="model"):
    scored = [p for p in preds if p["pred"] is not None]
    unc = [p for p in scored if not p["row"]["censored"]]
    cov = [p["covered"] for p in unc]
    return {
        "name": name,
        "scored": len(scored),
        "uncensored": len(unc),
        "unpredictable": len(preds) - len(scored),
        "logscore": sum(p["logscore"] for p in scored),
        "coverage": sum(cov) / len(cov) if cov else None,
        "mae": float(np.median([p["abs_err"] for p in unc])) if unc else None,
    }


def baseline_summaries(preds, physiology=None):
    """Point baselines scored with the base model's spread at the same step; physiology-only
    with its own N(ln T_phys, σ_phys²)."""
    out = {}
    names = ("PB-fallback", "comparable-history") + (("physiology-only",) if physiology else ())
    for name in names:
        ls, cov, err, n = 0.0, [], [], 0
        for p in preds:
            if name == "physiology-only":
                if p["pred"] is None or p["row"]["mode"] not in physiology:
                    continue
                loc, scale = physiology[p["row"]["mode"]]
                df = 1e9
                p = dict(p, baselines={name: math.exp(loc)})
            elif p["pred"] is None or name not in p["baselines"]:
                continue
            else:
                _, scale, df = p["pred"]
            loc = math.log(p["baselines"][name])
            n += 1
            ls += score(loc, scale, df, p["row"])
            if not p["row"]["censored"]:
                cov.append(abs(p["row"]["y"] - loc) <= t90(df) * scale)
                err.append(abs(p["baselines"][name] - p["row"]["actual"]))
        out[name] = {
            "name": name,
            "scored": n,
            "uncensored": len(err),
            "logscore": ls,
            "coverage": sum(cov) / len(cov) if cov else None,
            "mae": float(np.median(err)) if err else None,
        }
    return out


def single_input_beta(name, kind):
    """beta_fn for one candidate: its coefficient re-estimated on each step's training rows."""

    def fn(train, cfg, name=name, kind=kind):
        c = dict(cfg)
        c["covariates" if kind == "covariate" else "axes"] = [name]
        return with_coefficients(train, c)

    return fn


def with_coefficients(train, cfg, tz=HARNESS_TIME_ZONE):
    """cfg with beta/betaCov/gamma/gammaCov estimated jointly for its covariates and axes."""
    c = dict(cfg)
    coef, cov = estimate_coefficients(train, c, tz)
    k = len(c["covariates"])
    c["beta"], c["gamma"] = coef[:k], coef[k:]
    c["betaCov"] = [row[:k] for row in cov[:k]]
    c["gammaCov"] = [row[k:] for row in cov[k:]]
    return c


def estimate_coefficients(train, cfg, tz=HARNESS_TIME_ZONE):
    """HARNESS: joint ridge of y on the inputs within volumes, uncensored rows only, weighted by
    recency, inputs measured from the same weighted-median centres fit() uses (missing = centre).
    Prior sd 0.2 per input sd; σ² from the within-volume spread of y. An input with no spread,
    or fewer than 3 usable rows, gets coefficient 0 and variance 0 (it contributes nothing).
    Returns (coefficients, covariance) in covariates-then-axes order."""
    k = len(cfg["covariates"]) + len(cfg["axes"])
    zero = ([0.0] * k, [[0.0] * k for _ in range(k)])
    fin = finished_sessions(train)
    rows = model_rows(fin)
    if k == 0 or not rows:
        return zero
    t_now = max(started(s) for s in fin)
    w = [weight(t_now, started(r["session"]), cfg["halfLifeDays"]) for r in rows]
    xa = [inputs(r, fin, cfg, tz) for r in rows]
    centre = centres([x + a for x, a in xa], w)
    pts = [
        (r["mode"], deviation(x + a, centre), r["y"], wi)
        for r, (x, a), wi in zip(rows, xa, w, strict=True)
        if not r["censored"]
    ]
    if len(pts) < 3:
        return zero
    d, dy, ws, groups = [], [], [], {m for m, *_ in pts}
    for m in groups:
        grp = [p for p in pts if p[0] == m]
        sw = sum(p[3] for p in grp)
        mx = [sum(p[1][j] * p[3] for p in grp) / sw for j in range(k)]
        my = sum(p[2] * p[3] for p in grp) / sw
        d += [[p[1][j] - mx[j] for j in range(k)] for p in grp]
        dy += [p[2] - my for p in grp]
        ws += [p[3] for p in grp]
    dmat, yv, wv = np.array(d), np.array(dy), np.array(ws)
    sxx = (dmat**2 * wv[:, None]).sum(axis=0)
    live = [j for j in range(k) if sxx[j] > 0]
    if not live:
        return zero
    sd = np.sqrt(sxx[live] / wv.sum())
    s2 = max(float((wv * yv**2).sum()) / max(len(yv) - len(groups), 1), 1e-6)
    dl = dmat[:, live]
    prec = dl.T @ (dl * wv[:, None]) / s2 + np.diag((sd / 0.2) ** 2)
    cov_l = np.linalg.inv(prec)
    mean_l = cov_l @ (dl.T @ (wv * yv)) / s2
    coef, cov = zero
    for a_, ja in enumerate(live):
        coef[ja] = float(mean_l[a_])
        for b_, jb in enumerate(live):
            cov[ja][jb] = float(cov_l[a_, b_])
    return coef, cov


# ---------------------------------------------------------------- v1 rows (diagnostic)


def v1_rows(session):
    """Contract v1's rule, for the diagnostic only."""
    if session.get("endedAt") is None or "hypoxia" in symptoms(session):
        return []
    out = []
    for st in session["steps"]:
        if not (is_performed_hold(st) and looks_like_max_attempt(st)) or lung_mode_of(st) is None:
            continue
        imp = is_imported_from_stamina(session)
        if (imp and st.get("styleSource") != "confirmed") or (
            not imp and st.get("styleSource") in ("estimated", "imported")
        ):
            continue
        o = st["outcome"]
        planned = st.get("plannedDuration")
        cens = o["type"] == "endedEarly" or (
            o["type"] == "skipped" and planned is not None and st["actualDuration"] < planned - 1
        )
        out.append((lung_mode_of(st), cens))
    return out


# ---------------------------------------------------------------- report


def _fmt(t):
    return f"{int(t) // 60}:{int(round(t)) % 60:02d}"


def report(sessions, out, physiology=None, tz=HARNESS_TIME_ZONE, packing=None):
    fin = finished_sessions(sessions)
    rows = model_rows(fin)
    out.append("4. HOLD MODEL: WHERE THE PB MIGHT LAND (contract §10 v2)")
    out.append("-" * 56)
    out.append("")
    out.append(
        "Rows: per session and lung-mode slot, the longest eligible hold (BestDetection's "
        "candidate),\nplus any hold marked as a max attempt. Uncensored only when the diver "
        "ended it outside a\nseries; a timer end, the floor, an abort or a series hold is a "
        "lower bound."
    )
    out.append("")
    out.append(f"  {'session':<12}{'slot':<6}{'style':<8}{'held':>6}{'planned':>9}  {'row':<11}")
    for r in rows:
        st = r["session"]["steps"][r["stepIndex"]]
        planned = st.get("plannedDuration")
        out.append(
            f"  {r['session']['startedAt'][:10]:<12}{r['mode']:<6}{st.get('style', ''):<8}"
            f"{_fmt(r['actual']):>6}{_fmt(planned) if planned else 'open':>9}  "
            f"{'censored' if r['censored'] else 'UNCENSORED'}"
        )
    out.append("")

    # diagnostics
    out.append("  Diagnostics (harness only):")
    v1 = [x for s in fin for x in v1_rows(s)]
    for v in MODES:
        a = [c for m, c in v1 if m == v]
        b = [r["censored"] for r in rows if r["mode"] == v]
        out.append(
            f"    {v:<5} rows v1 {len(a):>2} ({a.count(False)} uncensored)   "
            f"v2 {len(b):>2} ({b.count(False)} uncensored)"
        )
    marked = sum(1 for s in fin if "hypoxia" in symptoms(s))
    out.append(
        f"    sessions with a reported hypoxia feeling: {marked} (a marker on the session's last "
        "hold; no\n    symptom excludes a hold, §18). No blackout, LMC or samba is recordable "
        "yet."
    )
    oct4 = next((s for s in fin if s["startedAt"].startswith("2026-10-04")), None)
    if oct4:
        short = [
            (i, st)
            for i, st in enumerate(oct4["steps"])
            if is_performed_hold(st) and st.get("style") == "full" and st["actualDuration"] < 60
        ]
        for i, st in short:
            win = [r for r in hold_model_rows(oct4) if r["mode"] == "full"]
            winner = oct4["steps"][win[0]["stepIndex"]] if win else None
            out.append(
                f"    4 Oct: the open-ended full hold of {st['actualDuration']} s (step {i}) is "
                "not a "
                f"row; the {winner.get('style')} {winner['actualDuration']} s hold takes the "
                "full slot."
                if winner
                else f"    4 Oct: the {st['actualDuration']} s open-ended full hold is not a row."
            )
    out.append("")

    # half-life choice on the base model
    out.append("  Forward chain (sessions with rows by date, first 5 as burn-in):")
    chains = {h: chain(fin, base_config(h), physiology, tz, packing=packing) for h in HALF_LIVES}
    sums = {h: summarise(p, f"base h={h or 'none'}") for h, p in chains.items()}

    # A half-life is an input like any other: it must beat no weighting on log score with
    # coverage inside 70-90 %, else none.
    def admits(sm, base):
        return (
            sm["logscore"] > base["logscore"]
            and sm["coverage"] is not None
            and 0.7 <= sm["coverage"] <= 0.9
        )

    better = [h for h in HALF_LIVES[1:] if admits(sums[h], sums[None])]
    best_h = max(better, key=lambda h: sums[h]["logscore"]) if better else None
    base = chains[best_h]
    out.append(
        f"    {'model':<22}{'scored':>7}{'uncens':>7}{'log score':>11}{'80% cov':>9}{'MAE s':>8}"
    )

    def line(sm):
        cov = "-" if sm["coverage"] is None else f"{sm['coverage']:.0%}"
        mae = "-" if sm["mae"] is None else f"{sm['mae']:.0f}"
        out.append(
            f"    {sm['name']:<22}{sm['scored']:>7}{sm['uncensored']:>7}{sm['logscore']:>11.2f}"
            f"{cov:>9}{mae:>8}"
        )

    for h in HALF_LIVES:
        line(sums[h])
    for sm in baseline_summaries(base, physiology).values():
        line(sm)
    if not physiology:
        out.append("    physiology-only       unavailable: the export carries no profile")
    n_unc = sums[None]["uncensored"]
    if not any(0.7 <= j / n_unc <= 0.9 for j in range(n_unc + 1)) if n_unc else True:
        out.append(
            f"    half-life: not decidable with {n_unc} uncensored rows, none used (a half-life "
            "must beat none on\n    log score with coverage within 70-90 %, which "
            f"{n_unc} rows cannot show)"
        )
    else:
        out.append(
            f"    half-life chosen: {best_h or 'none'} "
            "(a half-life must beat none on log score with coverage within 70-90 %)"
        )
    out.append(
        "    Point baselines are scored with the base model's spread at the same step, so they "
        "differ\n    only in where they centre the prediction."
    )
    out.append("")

    # admission
    cfg = base_config(best_h)
    admitted_cov, admitted_ax, decide = [], [], []
    base_sum = sums[best_h]
    all_rows_unc = sum(1 for r in rows if not r["censored"])
    out.append(
        "  Inputs, one at a time against the base (admit on log-score gain with coverage 70-90 %):"
    )
    for kind, names in (("covariate", CONTEXT_FIELDS), ("axis", tuple(AXES))):
        for name in names:
            preds = chain(
                fin, cfg, physiology, tz, beta_fn=single_input_beta(name, kind), packing=packing
            )
            sm = summarise(preds, name)
            gain = sm["logscore"] - base_sum["logscore"]
            ok = admits(sm, base_sum)
            if ok:
                (admitted_cov if kind == "covariate" else admitted_ax).append(name)
            c = dict(cfg)
            c["covariates" if kind == "covariate" else "axes"] = [name]
            coef, cov = estimate_coefficients(fin, c)
            b, var = coef[0], cov[0][0]
            if var == 0:
                need = f"not estimable ({all_rows_unc} uncensored rows)"
            elif b == 0:
                need = "no effect seen"
            else:
                n_need = all_rows_unc * (Z90 * math.sqrt(var) / abs(b)) ** 2
                more = max(0, math.ceil(n_need - all_rows_unc))
                need = f"about {more} more max attempt{'' if more == 1 else 's'} to decide"
            decide.append((name, need))
            out.append(
                f"    {name:<26}{'admitted' if ok else 'not admitted':<14}gain {gain:+.2f}   {need}"
            )
    out.append(
        f"    admitted covariates: {admitted_cov or 'none'}; admitted axes: {admitted_ax or 'none'}"
    )
    out.append(
        "    per-step estimate (former 4b): NOT ADOPTED"
        + ("" if admitted_cov else " (no context input admitted, so it would equal the level)")
    )
    out.append("")

    # the shipped model on all data
    cfg["covariates"], cfg["axes"] = admitted_cov, admitted_ax
    cfg = with_coefficients(fin, cfg)
    model = fit(fin, cfg, physiology, tz, packing=packing)
    est = estimates(model, fin)
    out.append(
        f"  Model on all data (half-life {best_h or 'none'}, σ = {math.sqrt(model['sigma2']):.3f} "
        f"log units, df {2 * model['an']:.1f}):"
    )
    for v in MODES:
        e = est[v]
        prior = model["priors"].get(v)
        pr = (
            "none"
            if prior is None
            else f"{prior[2]} N(ln {_fmt(math.exp(prior[0]))}, {prior[1]:.2f}²)"
        )
        out.append(
            f"    {v:<5} prior {pr}; counts {e['counts']['uncensored']} uncensored, "
            f"{e['counts']['censored']} censored, effective {e['counts']['effective']:.1f}"
        )
        for kind in ("level", "pb"):
            r = e[kind]
            if r["estimate"] is None:
                out.append(f"      {kind:<6}no posterior")
                continue
            lo, mid, hi = r["estimate"]
            out.append(
                f"      {kind:<6}{_fmt(lo)}-{_fmt(hi)} (centre {_fmt(mid)}), narrowing "
                f"{r['narrowing']:.2f}, gate {'ENOUGH' if r['gate'].enough else 'waiting'}"
            )
    no_cens = fit(
        fin,
        cfg,
        physiology,
        tz,
        rows=[r for r in rows if not r["censored"]],
        priors=model["priors"],
        packing=packing,
    )
    for v in MODES:
        if v in model["post"] and v in no_cens["post"]:
            a, b = model["post"][v], no_cens["post"][v]
            out.append(
                f"    {v:<5} level posterior (same prior) with censored rows "
                f"{_fmt(math.exp(a[0]))} "
                f"(sd {math.sqrt(a[1]):.3f}), without {_fmt(math.exp(b[0]))} (sd "
                f"{math.sqrt(b[1]):.3f})"
            )
    out.append("")
    return {
        "model": model,
        "estimates": est,
        "half_life": best_h,
        "config": cfg,
        "chains": chains,
        "decidability": decide,
    }
