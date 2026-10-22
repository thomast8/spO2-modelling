"""The gate's calibration check, inside the same forward chain (plan: The evidence gate, 3).

At every session k the gate is evaluated on the sessions before k, with the hysteresis memory
carried along the chain; when it says enough, the shown 80 % interval is scored on session k.
pb is scored on k's uncensored rows at that volume (a predictive interval, so 80 % is the target).
The delay's shown interval is on the mean, which is not meant to cover 80 % of new sessions, so
its coverage is reported and never moves a floor.
"""

import copy

from archive import HARNESS_TIME_ZONE, started
from archive import finished as finished_sessions
from evidence import DEFAULT_FLOORS
from features import hold_model_rows
from fit_delay import KEYS, post_hold_delay, session_rows
from fit_hold_model import MODES, estimates, fit, model_rows
from numerics import nig_predictive_interval

GRID = (5, 6, 7, 8, 10, 12)


def pb_decisions(sessions, cfg, floors, physiology=None, tz=HARNESS_TIME_ZONE):
    fin = finished_sessions(sessions)
    shown, out = set(), []
    for s in fin:
        train = [t for t in fin if started(t) < started(s)]
        if not model_rows(train):
            continue
        est = estimates(fit(train, cfg, physiology, tz), train, frozenset(shown), floors, tz)
        for v in MODES:
            key, res = f"pb.{v}", est[v]["pb"]
            if not res["gate"].enough:
                shown.discard(key)
                continue
            shown.add(key)
            lo, _, hi = res["estimate"]
            for r in hold_model_rows(s):
                if r["mode"] == v and not r["censored"]:
                    out.append((s["startedAt"][:10], v, lo <= r["actual"] <= hi))
    return out


def with_pb_floor(floors, n):
    f = copy.deepcopy(floors)
    for task in ("pb", "level"):
        f[task]["enough"]["maxAttempts"] = n
        f[task]["keep"]["maxAttempts"] = n - 1
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


def calibrate(sessions, cfg, delay_prior, physiology=None, out=None):
    floors = copy.deepcopy(DEFAULT_FLOORS)
    base = pb_decisions(sessions, cfg, floors, physiology)
    tried = []
    if base:
        for n in GRID:
            d = pb_decisions(sessions, cfg, with_pb_floor(floors, n), physiology)
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
        chosen = with_pb_floor(floors, ok[0] if ok else GRID[-1])
    pb = chosen["pb"]
    header = (
        f"calibrated: {'yes' if calibrated else 'no'}, coverage "
        f"{'n/a' if cov0 is None else f'{cov0:.0%}'} over {len(base)} decisions; floors "
        f"pb/level {pb['enough']['maxAttempts']}/{pb['keep']['maxAttempts']}, delay "
        f"{chosen['delay']['enough']['sessionsWithOximeter']}/"
        f"{chosen['delay']['keep']['sessionsWithOximeter']}, axis 5/4, holdVolume 10"
    )
    delay = delay_decisions(sessions, delay_prior)
    if out is not None:
        out.append("  Calibration (pb, scored on the next session's uncensored max attempts):")
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
