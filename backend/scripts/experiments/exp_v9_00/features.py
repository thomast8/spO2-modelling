"""Contract §2, §3, §6, §7, §8, §9: the extractors the Swift learning core ports bit for bit.

Every function reads a session dict in Nadir wire encoding (see archive.py) and returns plain
values; fixtures (`emit_fixtures.py`) pin the Swift side to these numbers.
"""

import math
from datetime import timedelta
from zoneinfo import ZoneInfo

from archive import (
    PM100_PERIOD_START,
    best_lung_mode,
    breathe_up,
    days_between,
    finished,
    has_trusted_spo2,
    is_hold,
    is_imported_from_stamina,
    is_performed_hold,
    lowest_spo2,
    reoxygenation_target,
    started,
    style_source,
    symptoms,
    time_to_reoxygenation,
    trusted_series,
)
from numerics import median, weighted_median

# ---------------------------------------------------------------- §2 oximeter key


def oximeter_key(session, roster):
    """`Analytics.oximeterKey(of:roster:)`."""
    if is_imported_from_stamina(session):
        return "stamina"
    recording = session.get("recording")
    if recording and recording.get("oximeter") is not None:
        return recording["oximeter"]
    if recording is None and started(session) >= PM100_PERIOD_START and has_trusted_spo2(session):
        oximeters = [r for r in roster if r["kind"] == "oximeter"]
        if len(oximeters) == 1:
            return oximeters[0]["id"]
    return None


# ---------------------------------------------------------------- §3 recovery trace


def next_performed_hold(steps, index):
    return next(
        (j for j in range(index + 1, len(steps)) if is_performed_hold(steps[j])), len(steps)
    )


def recovery_trace(hold_index, session):
    """`Analytics.recoveryTrace(after:in:)` → {onsetSeconds, traceSeconds, version} or None."""
    steps = session["steps"]
    if not (0 <= hold_index < len(steps)) or not is_performed_hold(steps[hold_index]):
        return None
    readings, offset = [], 0
    for step in steps[hold_index + 1 : next_performed_hold(steps, hold_index)]:
        readings += [(offset + at, x) for at, x in trusted_series(step, "spo2")]
        offset += step["actualDuration"]
    if not readings:
        return None
    m = min(x for _, x in readings)
    k = max(i for i, (_, x) in enumerate(readings) if x == m)
    onset = next((t for t, x in readings[k + 1 :] if x >= m + 2), None)
    return {"onsetSeconds": onset, "traceSeconds": readings[-1][0], "version": 1}


def stamped_trace(hold_index, session):
    """What the fold reads: `step.recoveryTrace ?? Analytics.recoveryTrace(after:in:)`."""
    stored = session["steps"][hold_index].get("recoveryTrace")
    if stored is None:
        return recovery_trace(hold_index, session)
    # The wire omits a nil onset (§1), so a censored stamp has no onsetSeconds key.
    return {
        "onsetSeconds": stored.get("onsetSeconds"),
        "traceSeconds": stored["traceSeconds"],
        "version": stored["version"],
    }


def is_censored_trace(trace):
    return trace is not None and trace["onsetSeconds"] is None and trace["traceSeconds"] < 90


# ---------------------------------------------------------------- §6 hold facts and features


def hold_facts(session):
    """`HoldVolumeClassifier.facts(of:)`, one dict per hold step (stepIndex added for keying)."""
    out, rest = [], 0
    for index, step in enumerate(session["steps"]):
        if not is_hold(step):
            rest += step["actualDuration"]
            continue
        planned = step.get("plannedDuration")
        facts = {
            "stepID": step["id"],
            "stepIndex": index,
            "planned": planned if planned is not None else step["actualDuration"],
            "actual": step["actualDuration"],
            "restBefore": rest,
            "baseline": None,
            "fallBy120": None,
            "fallBy150": None,
            "minBy180": None,
            "spo2Seconds": None,
        }
        rest = 0
        trace = trusted_series(step, "spo2")
        early = [p for p in trace if p[0] <= 30]
        if early:
            # Swift's max(by: <=) keeps the last of equal maxima.
            peak = early[0]
            for p in early[1:]:
                if peak[1] <= p[1]:
                    peak = p
            after = [p for p in trace if p[0] >= peak[0]]

            def low(limit, after=after):
                vals = [x for t, x in after if t <= limit]
                return min(vals) if vals else None

            facts["baseline"] = peak[1]
            l120, l150 = low(120), low(150)
            facts["fallBy120"] = None if l120 is None else peak[1] - l120
            facts["fallBy150"] = None if l150 is None else peak[1] - l150
            by180 = [x for t, x in trace if t <= 180]
            facts["minBy180"] = min(by180) if by180 else None
            facts["spo2Seconds"] = trace[-1][0]
        out.append(facts)
    return out


FEATURE_NAMES = (
    "logPlanned",
    "plannedShareOfMax",
    "position",
    "isFirst",
    "isLast",
    "logRestBefore",
    "logRatioPrevious",
    "hasRVToken",
    "hasFullToken",
    "hasFRCToken",
    "hasTrace",
    "baseline",
    "fallBy120",
    "fallBy150",
    "minBy180",
)
RV_TOKENS, FULL_TOKENS, FRC_TOKENS = {"rv", "empty", "exhale"}, {"fl", "full", "max"}, {"frc"}


def name_tokens(name):
    """planName lowercased, split on non-letters, whole tokens."""
    tokens, cur = [], ""
    for ch in name.lower():
        if ch.isalpha():
            cur += ch
        elif cur:
            tokens.append(cur)
            cur = ""
    if cur:
        tokens.append(cur)
    return set(tokens)


def hold_features(session):
    """§6: [(facts, [15 features])] in facts order."""
    facts = hold_facts(session)
    n = len(facts)
    top = max((f["planned"] for f in facts), default=0)
    tokens = name_tokens(session.get("planName", ""))
    flags = [float(bool(tokens & t)) for t in (RV_TOKENS, FULL_TOKENS, FRC_TOKENS)]
    rows = []
    for i, f in enumerate(facts):
        log_planned = math.log(max(f["planned"], 1))
        prev = math.log(max(facts[i - 1]["planned"], 1)) if i > 0 else None
        has_trace = 1.0 if f["spo2Seconds"] is not None and f["spo2Seconds"] >= 60 else 0.0

        def traced(key, f=f, has_trace=has_trace):
            return f[key] if has_trace == 1.0 and f[key] is not None else 0.0

        rows.append(
            (
                f,
                [
                    log_planned,
                    f["planned"] / top if top != 0 else 1.0,
                    i / n,
                    1.0 if i == 0 else 0.0,
                    1.0 if i == n - 1 else 0.0,
                    math.log(1 + f["restBefore"]),
                    0.0 if prev is None else log_planned - prev,
                    *flags,
                    has_trace,
                    traced("baseline"),
                    traced("fallBy120"),
                    traced("fallBy150"),
                    traced("minBy180"),
                ],
            )
        )
    return rows


# ---------------------------------------------------------------- §7 hold markers


def hr_baseline(session, step_index):
    """§7 hrBaseline: median trusted HR in the 30 s before the hold."""
    steps, values, before_end = session["steps"], [], 0
    for step in reversed(steps[:step_index]):
        if is_performed_hold(step):
            break
        for at, hr in trusted_series(step, "heartRate"):
            if before_end + (step["actualDuration"] - at) <= 30:
                values.append(hr)
        before_end += step["actualDuration"]
    return median(values)


def _valid_oxygen(sample):
    """VitalsSummary's oxygen filter: contact, finite, inside DomainLimits.spo2Range."""
    x = sample.get("spo2")
    return (
        sample.get("contactDetected") is True
        and x is not None
        and math.isfinite(x)
        and 0.0 <= x <= 100.0
    )


def oxygen_readings(step):
    """`HoldOxygenDrop.readings`: per whole second inside 0...actualDuration, the mean of that
    second's trusted SpO₂ (VitalsSummary); seconds without one are left out."""
    by_second = {}
    for sample in step.get("vitals", []):
        if 0 <= sample["at"] <= step["actualDuration"]:
            by_second.setdefault(sample["at"], []).append(sample)
    out = []
    for second in sorted(by_second):
        values = [s["spo2"] for s in by_second[second] if _valid_oxygen(s)]
        if values:
            out.append((second, sum(values) / len(values)))
    return out


def hold_oxygen_drop(index, steps):
    """`HoldOxygenDrop.measure(at:in:)`: (onset second, points per minute | None) or None.

    Baseline: `reoxygenationTarget` over the steps since the previous hold, cleaned to trusted
    in-range samples inside each step, needing three trusted seconds. Onset: the first of two
    consecutive seconds at or below baseline − 2, with no gap before it. Rate: from onset to the
    first hold low, only over four or more continuous seconds that end lower.
    """
    before = []
    for step in reversed(steps[:index]):
        if is_hold(step):
            break
        clean = dict(step)
        clean["vitals"] = sorted(
            (
                v
                for v in step.get("vitals", [])
                if 0 <= v["at"] <= step["actualDuration"] and _valid_oxygen(v)
            ),
            key=lambda v: v["at"],
        )
        before.insert(0, clean)
    if sum(len(oxygen_readings(s)) for s in before) < 3:
        return None
    baseline = reoxygenation_target(len(before), before)
    if baseline is None:
        return None
    samples = oxygen_readings(steps[index])
    if not samples or samples[0][0] > 1:
        return None
    onset = None
    for p in range(1, len(samples)):
        if samples[p][0] - samples[p - 1][0] != 1:
            return None
        if samples[p - 1][1] <= baseline - 2 and samples[p][1] <= baseline - 2:
            onset = p - 1
            break
    if onset is None:
        return None
    lowest = min(range(onset, len(samples)), key=lambda i: samples[i][1])
    falling = samples[onset : lowest + 1]
    continuous = all(b[0] - a[0] == 1 for a, b in zip(falling, falling[1:], strict=False))
    (t0, x0), (t1, x1) = samples[onset], samples[lowest]
    rate = (
        (x0 - x1) * 60 / (t1 - t0)
        if len(falling) >= 4 and continuous and t1 > t0 and x1 < x0
        else None
    )
    return t0, rate


def hold_markers(session):
    """`Analytics.holdMarkers(of:)`: one dict per performed hold."""
    out = []
    for index, step in enumerate(session["steps"]):
        if not is_performed_hold(step):
            continue
        actual = step["actualDuration"]
        contractions = step.get("contractions") or []
        urges = step.get("urges") or []
        first = min(contractions) if contractions else None
        struggle = actual - first if first is not None else None
        spo2 = trusted_series(step, "spo2")
        hr = trusted_series(step, "heartRate")

        early = [x for t, x in spo2 if t <= 30]
        baseline = max(early) if early else None
        drop = hold_oxygen_drop(index, session["steps"])

        hr_base = hr_baseline(session, index)
        hr_min = min(x for _, x in hr) if hr else None
        drop_share = (
            (hr_base - hr_min) / hr_base
            if hr_base is not None and hr_min is not None and hr_base > 0
            else None
        )
        to_drop = None
        if hr_base is not None and hr_min is not None and hr_base - hr_min > 0:
            level = hr_base - 0.9 * (hr_base - hr_min)
            to_drop = next((t for t, x in hr if x <= level), None)
        trace = stamped_trace(index, session)

        out.append(
            {
                "stepID": step["id"],
                "stepIndex": index,
                "firstContractionSeconds": first,
                "firstContractionShare": first / actual if first is not None and actual else None,
                "struggleSeconds": struggle,
                "contractionRatePerMin": (
                    sum(1 for c in contractions if c >= first) / (struggle / 60)
                    if struggle is not None and struggle >= 10
                    else None
                ),
                "firstUrgeSeconds": min(urges) if urges else None,
                "spo2Baseline": baseline,
                "spo2End": spo2[-1][1] if spo2 else None,
                "spo2Low": lowest_spo2([index], session),
                "secondsAtOrAbove95": sum(1 for _, x in spo2 if x >= 95),
                "desaturationOnsetSeconds": None if drop is None else drop[0],
                "desaturationRatePerMin": None if drop is None else drop[1],
                "hrBaseline": hr_base,
                "hrMin": hr_min,
                "hrDropShare": drop_share,
                "secondsToHrDrop": to_drop,
                "reoxygenationSeconds": time_to_reoxygenation(session, index),
                "recoveryOnsetSeconds": trace["onsetSeconds"] if trace else None,
            }
        )
    return out


_markers_cache = {}


def cached_markers(session):
    key = id(session)
    if key not in _markers_cache:
        _markers_cache[key] = (session, {m["stepIndex"]: m for m in hold_markers(session)})
    return _markers_cache[key][1]


# ---------------------------------------------------------------- §8 diver axes

AXES = {
    "co2Tolerance": "firstContractionShare",
    "urgeOnset": "firstContractionSeconds",
    "desaturationRate": "desaturationRatePerMin",
    "diveReflex": "hrDropShare",
    "recovery": "reoxygenationSeconds",
}
AXIS_UNIT = {
    "co2Tolerance": "holdsWithContractions",
    "urgeOnset": "holdsWithContractions",
    "desaturationRate": "holdsWithSpO2",
    "diveReflex": "holdsWithSpO2",
    "recovery": "holdsWithSpO2",
}


def counts_for_learning(session, step):
    """Imported holds count only once the diver confirmed their volume."""
    return not is_imported_from_stamina(session) or style_source(step) == "confirmed"


def diver_axes(sessions, as_of, half_life_days):
    """§8, before gating: {axis: {value, count, effectiveCount}}; value None when count is 0."""
    rows = {name: ([], []) for name in AXES}
    for session in finished(sessions):
        if not started(session) < as_of:
            continue
        dt = days_between(as_of, started(session))
        w = 1.0 if half_life_days is None else 2 ** (-dt / half_life_days)
        markers = cached_markers(session)
        for index, step in enumerate(session["steps"]):
            if not is_performed_hold(step) or step["actualDuration"] < 60:
                continue
            if not counts_for_learning(session, step):
                continue
            for name, marker in AXES.items():
                v = markers[index][marker]
                if v is not None:
                    rows[name][0].append(v)
                    rows[name][1].append(w)
    return {
        name: {
            "value": weighted_median(vals, ws),
            "count": len(vals),
            "effectiveCount": sum(ws),
        }
        for name, (vals, ws) in rows.items()
    }


# ---------------------------------------------------------------- §9 hold context

CONTEXT_FIELDS = (
    "holdIndex",
    "logRestBefore",
    "warmUp",
    "breatheHyperventilation",
    "breathePacking",
    "openEnded",
    "logDaysSinceLastSession",
    "logDaysSinceLastMax",
    "timeOfDaySin",
    "timeOfDayCos",
    "preHoldHeartRate",
)


def hold_context(session, step_index, previous_sessions, time_zone):
    """`HoldContext.init(session:stepIndex:previousSessions:timeZone:)`."""
    steps = session["steps"]
    step = steps[step_index]
    hold_index = sum(1 for st in steps[:step_index] if is_performed_hold(st))
    rest = 0
    for st in reversed(steps[:step_index]):
        if is_performed_hold(st):
            break
        if not is_hold(st):
            rest += st["actualDuration"]
    up = breathe_up(step_index, session)
    up_style = up.get("style") if up else None

    start = started(session)
    earlier = [s for s in previous_sessions if s.get("endedAt") is not None and started(s) < start]
    last = max((started(s) for s in earlier), default=None)
    last_max = max(
        (started(s) for s in earlier if any(not r["censored"] for r in hold_model_rows(s))),
        default=None,
    )
    elapsed = sum(st["actualDuration"] for st in steps[:step_index])
    local = (start + timedelta(seconds=elapsed)).astimezone(ZoneInfo(time_zone))
    hour = local.hour + local.minute / 60 + local.second / 3600
    angle = 2 * math.pi * hour / 24
    return {
        "holdIndex": float(hold_index),
        "logRestBefore": math.log(1 + rest),
        "warmUp": 1.0 if hold_index > 0 else 0.0,
        "breatheHyperventilation": 1.0 if up_style == "hyperventilation" else 0.0,
        "breathePacking": 1.0 if up_style == "packing" else 0.0,
        "openEnded": 1.0 if step.get("plannedDuration") is None else 0.0,
        "logDaysSinceLastSession": None
        if last is None
        else math.log(1 + days_between(start, last)),
        "logDaysSinceLastMax": (
            None if last_max is None else math.log(1 + days_between(start, last_max))
        ),
        "timeOfDaySin": math.sin(angle),
        "timeOfDayCos": math.cos(angle),
        "preHoldHeartRate": hr_baseline(session, step_index),
    }


def lung_mode_of(step):
    return best_lung_mode(step.get("style"))


# ---------------------------------------------------------------- §10 rows (contract v2)


def eligible_for_model(session, step):
    if not is_performed_hold(step) or lung_mode_of(step) is None:
        return False
    source = style_source(step)
    if is_imported_from_stamina(session):
        return source == "confirmed"
    return source not in ("estimated", "imported")


def diver_ended(step):
    """§10 v2: the diver chose the end."""
    trigger = step.get("endTrigger")
    if trigger is not None:
        return trigger in ("tap", "skip")
    planned, actual = step.get("plannedDuration"), step["actualDuration"]
    kind, reason = step["outcome"]["type"], step["outcome"].get("reason")
    if planned is None:
        return kind == "completed"
    if (
        kind == "skipped" or (kind == "endedEarly" and reason == "tappedOut")
    ) and actual < planned - 1:
        return True
    return step.get("continuePastTarget") is True and kind == "completed" and actual > planned + 1


def in_series(step):
    count = step.get("iterationCount")
    return count is not None and count > 1 and step.get("maxAttempt") is not True


# §18 (#704): the model uses every hold, always; no symptom excludes one. A reported blackout,
# loss of motor control or samba marks the session's last performed hold as past the limit
# (d = 1). SessionFeedback.Symptom has no such case yet, so these are placeholders for the raw
# values #704 will add and nothing matches today. A "hypoxia" feeling is only a marker.
PAST_LIMIT_SYMPTOMS = frozenset({"blackout", "lossOfMotorControl", "samba"})


def past_limit_hold(session):
    """Index of the hold a reported blackout/LMC/samba marks as past the limit (the session's
    last performed hold), or None."""
    if not PAST_LIMIT_SYMPTOMS & set(symptoms(session)):
        return None
    return max((i for i, st in enumerate(session["steps"]) if is_performed_hold(st)), default=None)


def hold_model_rows(session):
    """§10 v2: each lung-mode slot's longest eligible hold (`BestDetection.candidateHolds`, first
    of equals) plus every other eligible hold marked `maxAttempt == true`. No symptom excludes a
    hold (§18)."""
    if session.get("endedAt") is None:
        return []
    steps = session["steps"]
    best = {}
    for i, st in enumerate(steps):
        if not eligible_for_model(session, st):
            continue
        slot = lung_mode_of(st)
        if st["actualDuration"] > (steps[best[slot]]["actualDuration"] if slot in best else 0):
            best[slot] = i
    marked = {
        i
        for i, st in enumerate(steps)
        if eligible_for_model(session, st) and st.get("maxAttempt") is True
    }
    return [
        {
            "session": session,
            "stepIndex": i,
            "mode": lung_mode_of(steps[i]),
            "actual": steps[i]["actualDuration"],
            "y": math.log(steps[i]["actualDuration"]),
            "censored": not (diver_ended(steps[i]) and not in_series(steps[i])),
        }
        for i in sorted(set(best.values()) | marked)
    ]
