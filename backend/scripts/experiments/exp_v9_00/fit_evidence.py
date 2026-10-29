"""The gate's calibration check, inside the same forward chain (plan: The evidence gate, 3).

At every session k the gate is evaluated on the sessions before k, with the hysteresis memory
carried along the chain; when it says enough, the shown 80 % interval is scored on session k.
pb is scored on k's uncensored rows at that volume (a predictive interval, so 80 % is the target).
The delay's shown interval is on the mean, which is not meant to cover 80 % of new sessions, so
its coverage is reported and never moves a floor.
"""

import copy
import math

from archive import HARNESS_TIME_ZONE, started
from archive import finished as finished_sessions
from capacity import NO_DIVER, capacity_rows, v5_fit
from evidence import DEFAULT_FLOORS
from features import hold_model_rows
from fit_delay import KEYS, post_hold_delay, session_rows
from fit_hold_model import MODES, estimates, fit, model_rows
from numerics import nig_predictive_interval

GRID = (5, 6, 7, 8, 10, 12)


def pb_decisions(
    sessions,
    cfg,
    floors,
    physiology=None,
    tz=HARNESS_TIME_ZONE,
    packing=None,
    rule="v2",
    ctx=None,
):
    """(day, volume, covered) for every enough pb decision in the chain. v2 scores the next
    session's uncensored rows; v5 its holds at the limit (d ≥ 0.8 from the earlier sessions),
    a packed hold against the interval moved by its packing offset."""
    fin = finished_sessions(sessions)
    shown, out = set(), []
    for s in fin:
        train = [t for t in fin if started(t) < started(s)]
        if rule == "v5":
            model, rows, dist, _ = v5_fit(train, cfg, ctx or NO_DIVER, tz)
            if not rows:
                continue
            tests = [r for r in capacity_rows([s], dist) if not r["censored"]]
        else:
            if not model_rows(train):
                continue
            model = fit(train, cfg, physiology, tz, packing=packing)
            tests = [r for r in hold_model_rows(s) if not r["censored"]]
        est = estimates(model, train, frozenset(shown), floors, tz, rule=rule)
        for v in MODES:
            key, res = f"pb.{v}", est[v]["pb"]
            if not res["gate"].enough:
                shown.discard(key)
                continue
            shown.add(key)
            lo, _, hi = res["estimate"]
            for r in tests:
                if r["mode"] != v:
                    continue
                step = r["session"]["steps"][r["stepIndex"]]
                off = packing.offset(step) if (packing and rule == "v5") else 0.0
                out.append(
                    (
                        s["startedAt"][:10],
                        v,
                        lo * math.exp(off) <= r["actual"] <= hi * math.exp(off),
                    )
                )
    return out


def with_pb_floor(floors, n, rule="v2"):
    f = copy.deepcopy(floors)
    tasks, kind = (
        (("pb", "level"), "maxAttempts") if rule == "v2" else (("pbV5", "levelV5"), "holdsWithSpO2")
    )
    for task in tasks:
        f[task]["enough"][kind] = n
        f[task]["keep"][kind] = n - 1
    return f


def delay_decisions(sessions, prior):
    """[(key, day, mean covered, predictive covered)] for every enough decision."""
    fin = finished_sessions(sessions)
    out = []
    for key in KEYS:
        shown = False
        for s, x, _, _ in session_rows(fin, key):
            if x is None:
                continue
            train = [t for t in fin if started(t) < started(s)]
            r = post_hold_delay(train, key, prior, previously_shown=shown)
            shown = r["gate"].enough
            if not shown:
                continue
            e = r["estimate"]
            plo, _, phi_ = nig_predictive_interval(r["posterior"])
            out.append((key, s["startedAt"][:10], e["lower"] <= x <= e["upper"], plo <= x <= phi_))
    return out


def calibrate(
    sessions, cfg, delay_prior, physiology=None, out=None, packing=None, rule="v2", ctx=None
):
    floors = copy.deepcopy(DEFAULT_FLOORS)

    def decisions(f):
        return pb_decisions(sessions, cfg, f, physiology, packing=packing, rule=rule, ctx=ctx)

    base = decisions(floors)
    tried = []
    if base:
        for n in GRID:
            d = decisions(with_pb_floor(floors, n, rule))
            cov = sum(x[2] for x in d) / len(d) if d else None
            tried.append((n, len(d), cov))
    cov0 = sum(x[2] for x in base) / len(base) if base else None
    if not base:
        calibrated, chosen = False, floors
    elif 0.7 <= cov0 <= 0.9:
        calibrated, chosen = True, floors
    else:
        ok = [n for n, k, c in tried if k and 0.7 <= c <= 0.9]
        calibrated = bool(ok)
        chosen = with_pb_floor(floors, ok[0] if ok else GRID[-1], rule)
    if rule == "v2":
        pb = chosen["pb"]
        pb_text = f"pb/level {pb['enough']['maxAttempts']}/{pb['keep']['maxAttempts']}"
    else:
        pb = chosen["pbV5"]
        pb_text = (
            f"pb/level (v5) holds with SpO₂ {pb['enough']['holdsWithSpO2']}/"
            f"{pb['keep']['holdsWithSpO2']}, effective {pb['enough']['effectiveHolds']}/"
            f"{pb['keep']['effectiveHolds']}"
        )
    header = (
        f"calibrated: {'yes' if calibrated else 'no'}, coverage "
        f"{'n/a' if cov0 is None else f'{cov0:.0%}'} over {len(base)} decisions; floors "
        f"{pb_text}, delay "
        f"{chosen['delay']['enough']['sessionsWithOximeter']}/"
        f"{chosen['delay']['keep']['sessionsWithOximeter']}, axis 5/4, holdVolume 10"
    )
    delay = delay_decisions(sessions, delay_prior)
    if out is not None:
        out.append(
            "  Calibration (pb, scored on the next session's "
            + ("uncensored max attempts):" if rule == "v2" else "holds at the limit, v5):")
        )
        out.append(
            f"    {len(base)} enough-decisions with defaults"
            + (f", coverage {cov0:.0%}" if base else "")
        )
        for n, k, c in tried:
            out.append(
                f"    floor {n:>2}: {k} decisions" + ("" if c is None else f", coverage {c:.0%}")
            )
        if not base:
            out.append(
                "    No pb gate said enough anywhere in the chain, so nothing could be scored; "
                "floors stay\n    at the defaults."
            )
        for key in KEYS:
            d = [x for x in delay if x[0] == key]
            if d:
                out.append(
                    f"    delay.{key}: {len(d)} enough-decisions; the mean interval covered the "
                    "next "
                    f"session's mean {sum(x[2] for x in d)}/{len(d)}, the predictive interval "
                    f"{sum(x[3] for x in d)}/{len(d)}"
                )
        out.append("    (the delay shows an interval on the mean, so its coverage moves no floor)")
        out.append(f"  {header}")
        out.append("")
    return {"floors": chosen, "calibrated": calibrated, "header": header, "decisions": base}
