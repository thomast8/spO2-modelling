"""Nadir sessions as plain dicts, plus the RecordedSession helpers the learning contract reads.

The export's sessions are kept verbatim (Nadir wire encoding: TimeSpans are whole seconds,
dates ISO-8601, outcomes tagged objects); nothing is resampled. Two things are added on top:

- the 6 August 2026 relabel from exp_v8_00's `LABEL_OVERRIDES`, written the way the app will
  write a diver's confirmation: `style` plus `styleSource: "confirmed"`;
- the 21 February 2026 STAmina CSV, rebuilt into a session the way Nadir's `StaminaImport`
  rebuilds one (it is a Masimo session that never went through the app).

Every helper mirrors a Swift computed property or function; the Swift name is in the docstring.
"""

import csv
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "exp_v8_00"))
from nadir_export import LABEL_OVERRIDES, _hms  # noqa: E402

# Contract §2. The export carries no sensor roster; the harness stands one in.
HARNESS_PM100_KEY = "pm100"
HARNESS_ROSTER = [{"id": HARNESS_PM100_KEY, "kind": "oximeter"}]
PM100_PERIOD_START = datetime(2026, 9, 2, tzinfo=UTC)
# The zone §9's time of day is read in. Not in the contract; the export's day-only STAmina
# imports start at 23:00Z in August, which is midnight in Europe/London (BST).
HARNESS_TIME_ZONE = "Europe/London"

STAMINA_TAGS = {"imported", "stamina"}
OVERRIDE_STYLE = {"RV": "rv", "FL": "full", "FRC": "frc"}
SPO2_RANGE = (0.0, 100.0)  # DomainLimits.spo2Range: decoding drops anything outside
HR_RANGE = (0.0, 300.0)  # DomainLimits.heartRateRange


# ---------------------------------------------------------------- dates


def parse_date(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def format_date(date):
    return date.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def started(session):
    return parse_date(session["startedAt"])


# ---------------------------------------------------------------- loading


def apply_label_overrides(sessions):
    """Writes `LABEL_OVERRIDES` as confirmed styles. Hold numbers count performed holds, 1-based."""
    for session in sessions:
        day = LABEL_OVERRIDES.get(session["startedAt"][:10])
        if not day:
            continue
        k = 0
        for step in session["steps"]:
            if not is_performed_hold(step):
                continue
            k += 1
            if k in day:
                step["style"] = OVERRIDE_STYLE[day[k]]
                step["styleSource"] = "confirmed"
    return sessions


def load_export(path):
    """The export's sessions, verbatim, with the 6 August relabel applied."""
    data = json.loads(Path(path).read_text())
    return apply_label_overrides(data["sessions"])


def finished(sessions):
    """Contract preamble: finished sessions, sorted by (startedAt, id), first copy of an id wins."""
    out, seen = [], set()
    for s in sorted(
        (s for s in sessions if s.get("endedAt") is not None), key=lambda s: (started(s), s["id"])
    ):
        if s["id"] not in seen:
            seen.add(s["id"])
            out.append(s)
    return out


def load_all(export_path, stamina_csvs=()):
    return finished(load_export(export_path) + [stamina_session(p) for p in stamina_csvs])


# ---------------------------------------------------------------- STAmina CSV


def _stamina_kind(kind):
    """`StaminaImport.kind(for:)`."""
    k = kind.lower()
    if k in ("apnea", "hold", "breathhold", "breath hold"):
        return "hold", "full", True
    if k in ("rest", "breathe", "breathing"):
        return "breathe", "recovery", True
    if k in ("cooldown", "cool down", "recovery"):
        return "rest", None, True
    return "rest", None, False


def stamina_session(path, time_zone=HARNESS_TIME_ZONE):
    """A STAmina CSV rebuilt as `StaminaImport.parse` rebuilds it, for the files this repo has.

    Covers what the February file uses: a day-first header date with the time zeroed, Round
    sections, a Biometrics section with HR and SpO₂. No Contractions section, no stamp in the
    file name. Session id is a readable stand-in for the importer's fingerprint.
    """
    rounds, samples, columns, section = [], [], None, None
    with open(path, newline="", encoding="utf-8-sig") as fh:
        lines = [row for row in csv.reader(fh) if row and row[0] != "sep="]
    name, header_date = lines[0][0] or "Imported session", lines[0][1]
    for row in lines[1:]:
        first = row[0].strip()
        if first.startswith("Round "):
            rounds.append([])
            section = "round"
        elif first == "Biometrics":
            section, columns = "bio", None
        elif section == "bio":
            if columns is None:
                columns = {h.strip().lower(): i for i, h in enumerate(row)}
                continue
            spo2 = row[columns["spo2"]].strip() if "spo2" in columns else ""
            hr = row[columns["hr"]].strip() if "hr" in columns else ""
            samples.append(
                {
                    "at": _hms(row[columns["time"]]),
                    "intervalAt": _hms(row[columns["interval time"]]),
                    "type": row[columns["interval type"]].strip(),
                    "spo2": float(spo2) if spo2 else None,
                    "heartRate": float(hr) if hr else None,
                }
            )
        elif section == "round" and first != "Number" and len(row) >= 3:
            rounds[-1].append((row[1].strip(), _hms(row[2])))
    planned = [(iv, r) for r, rnd in enumerate(rounds) for iv in rnd]

    segments = []  # StaminaImport.segments
    for s in samples:
        last = segments[-1] if segments else None
        if (
            last
            and s["intervalAt"] >= last["samples"][-1]["intervalAt"]
            and s["type"] == last["type"]
        ):
            last["samples"].append(s)
            continue
        resets = last is None or s["intervalAt"] < last["samples"][-1]["intervalAt"]
        start = max(0, s["at"] - s["intervalAt"]) if resets else s["at"]
        segments.append({"type": s["type"], "start": start, "samples": [s]})
    kept = segments[: len(planned)]

    steps = []
    for i, seg in enumerate(kept):
        (ptype, pdur), rnd = planned[i]
        if i + 1 < len(kept):
            actual = kept[i + 1]["start"] - seg["start"]
        else:
            actual = seg["samples"][-1]["intervalAt"] + 1
        actual = max(0, actual)
        kind, style, known = _stamina_kind(seg["type"] or ptype)
        if actual == 0:
            outcome = {"type": "skipped"}
        elif kind == "hold" and actual < pdur - 1:
            outcome = {"type": "endedEarly", "reason": "tappedOut"}
        else:
            outcome = {"type": "completed"}
        step = {
            "id": f"stamina#{i}",
            "kindName": kind,
            "iteration": rnd + 1,
            "iterationCount": len(rounds),
            "plannedDuration": pdur,
            "actualDuration": actual,
            "outcome": outcome,
            "contractions": [],
            "vitals": [
                {
                    k: v
                    for k, v in {
                        "at": s["intervalAt"],
                        "spo2": s["spo2"],
                        "heartRate": s["heartRate"],
                        "contactDetected": True,
                    }.items()
                    if v is not None
                }
                for s in seg["samples"]
            ],
        }
        if style:
            step["style"] = style
        if not known:
            step["label"] = seg["type"]
        steps.append(step)

    day, month, year = (int(x) for x in header_date.split(" ")[0].split("/"))
    start = datetime(year, month, day, tzinfo=ZoneInfo(time_zone))
    total = sum(st["actualDuration"] for st in steps)
    return {
        "id": f"stamina-{start.date().isoformat()}-{name.lower()}",
        "planName": name,
        "startedAt": format_date(start),
        "endedAt": format_date(start + timedelta(seconds=total)),
        "steps": steps,
        "tags": ["imported", "stamina"],
        "maxHoldAtTime": {},
        "completed": len(kept) == len(planned),
    }


# ---------------------------------------------------------------- RecordedStep / RecordedSession


def is_hold(step):
    return step["kindName"] == "hold"


def is_performed_hold(step):
    """`RecordedStep.isPerformedHold`."""
    return is_hold(step) and step["actualDuration"] > 0


def looks_like_max_attempt(step):
    """`RecordedStep.looksLikeMaxAttempt`."""
    if step.get("maxAttempt") is not None:
        return step["maxAttempt"]
    return step.get("plannedDuration") is None and step.get("iterationCount") is None


def is_imported_from_stamina(session):
    """`RecordedSession.isImportedFromStamina`."""
    return set(session.get("tags", [])) >= STAMINA_TAGS


def style_source(step):
    """`RecordedStep.styleSource` (contract §1): imported | estimated | confirmed | None."""
    return step.get("styleSource")


def trusted(sample, field):
    """A trusted reading (contract conventions), with the decoder's plausibility drop applied."""
    if sample.get("contactDetected") is not True:
        return None
    value = sample.get(field)
    lo, hi = SPO2_RANGE if field == "spo2" else HR_RANGE
    return value if value is not None and lo <= value <= hi else None


def trusted_series(step, field):
    """(at, value) for the step's trusted readings of `field`, in walk order."""
    out = []
    for v in step.get("vitals", []):
        x = trusted(v, field)
        if x is not None:
            out.append((v["at"], x))
    return out


def min_spo2(step):
    """`RecordedStep.minSpO2`."""
    values = [x for _, x in trusted_series(step, "spo2")]
    return min(values) if values else None


def has_trusted_spo2(session):
    return any(trusted_series(st, "spo2") for st in session["steps"])


def symptoms(session):
    return (session.get("feedback") or {}).get("symptoms") or []


def best_lung_mode(style):
    """`HoldStyle(rawValue:)?.bestLungMode` as a learning key (packed → full); None if unknown."""
    return {"full": "full", "packed": "full", "frc": "frc", "rv": "rv"}.get(style)


def reoxygenation_target(hold_index, steps):
    """`RecordedSession.reoxygenationTarget(before:in:)`."""
    for step in reversed(steps[:hold_index]):
        if is_hold(step):
            return None
        last = [x for _, x in trusted_series(step, "spo2")]
        if last:
            return min(last[-1], 99.0)
    return None


def time_to_reoxygenation(session, hold_index):
    """`RecordedSession.timeToReoxygenation(afterHoldAt:)`, in seconds."""
    steps = session["steps"]
    if not (0 <= hold_index < len(steps)) or not is_performed_hold(steps[hold_index]):
        return None
    baseline = reoxygenation_target(hold_index, steps)
    if baseline is None:
        return None
    nxt = next(
        (j for j in range(hold_index + 1, len(steps)) if is_performed_hold(steps[j])), len(steps)
    )
    readings, elapsed = [], 0
    for step in steps[hold_index + 1 : nxt]:
        readings += [(elapsed + at, x) for at, x in trusted_series(step, "spo2")]
        elapsed += step["actualDuration"]
    if not readings:
        return None
    lowest = min(x for _, x in readings)
    if not lowest < baseline:
        return None
    trough = max(i for i, (_, x) in enumerate(readings) if x == lowest)
    return next((t for t, x in readings[trough:] if x >= baseline), None)


def lowest_spo2(hold_indices, session):
    """`Analytics.lowestSpO2(ofHoldsAt:in:)`."""
    steps = session["steps"]
    holds = [i for i, st in enumerate(steps) if is_performed_hold(st)]
    if not holds:
        return None
    selected, low = set(hold_indices), None
    for pos, h in enumerate(holds):
        if h not in selected:
            continue
        start = 0 if h == holds[0] else h
        end = holds[pos + 1] if pos + 1 < len(holds) else len(steps)
        for step in steps[start:end]:
            v = min_spo2(step)
            if v is not None:
                low = v if low is None else min(low, v)
    return low


def breathe_up(index, session):
    """`Analytics.breatheUp(before:in:)`."""
    steps = session["steps"]
    if not 0 <= index < len(steps):
        return None
    for step in reversed(steps[:index]):
        if is_hold(step):
            return None
        if step["kindName"] == "breathe":
            return step
    return None


def days_between(later, earlier):
    return (later - earlier).total_seconds() / 86400.0
