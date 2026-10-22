"""Contract §13 and §16: the evidence gate and its English summary, as `EvidenceGate.swift` has
them.

Units are tuples (kind,) or (kind, parameter); `unit_name` gives the wire string `kind` or
`kind.parameter` (`maxAttempts.rv`, never `empty`). Floors are keyed by unit kind.
"""

import math
from dataclasses import dataclass, field

KINDS = (
    "maxAttempts",
    "sessionsWithOximeter",
    "holdsWithSpO2",
    "holdsWithContractions",
    "confirmedHolds",
    "profileField",
    "lungVolume",
    "readinessDays",
    "narrowing",
)


def _task(enough, keep, primary=None, rho_enough=None, rho_keep=None):
    return {
        "enough": enough,
        "keep": keep,
        "primary": primary,
        "rhoEnough": rho_enough,
        "rhoKeep": rho_keep,
    }


# The plan's hand-chosen defaults (`Priors.evidence`), before calibration.
DEFAULT_FLOORS = {
    "delay": _task(
        {"sessionsWithOximeter": 3}, {"sessionsWithOximeter": 2}, "sessionsWithOximeter", 0.10, 0.05
    ),
    "holdVolume": _task({"confirmedHolds": 10}, {"confirmedHolds": 10}),
    "spirometry": _task({"profileField": 1}, {"profileField": 1}),
    "oxygenStore": _task(
        {"profileField": 1, "lungVolume": 1}, {"profileField": 1, "lungVolume": 1}
    ),
    "axis": _task(
        {"holdsWithSpO2": 5, "holdsWithContractions": 5},
        {"holdsWithSpO2": 4, "holdsWithContractions": 4},
    ),
    "pb": _task({"maxAttempts": 5}, {"maxAttempts": 4}, "maxAttempts", 0.10, 0.05),
    "level": _task({"maxAttempts": 5}, {"maxAttempts": 4}, "maxAttempts", 0.10, 0.05),
    "readiness": _task({"readinessDays": 14}, {"readinessDays": 14}),
}

MODE_NAME = {"full": "full lung", "frc": "FRC", "rv": "RV"}
# §16 (23 Oct): units carry DiverProfile identifiers; the phrase uses display names.
FIELD_NAME = {
    "sex": "sex",
    "dateOfBirth": "date of birth",
    "heightCm": "height",
    "weightKg": "weight",
}
VOLUME_NAME = {"tlc": "TLC", "frc": "FRC", "rv": "RV", "vc": "VC", "fvc": "FVC"}


def unit_name(unit):
    return unit[0] if len(unit) == 1 else f"{unit[0]}.{unit[1]}"


def _order(unit):
    return KINDS.index(unit[0]), unit_name(unit)


@dataclass
class Gate:
    key: str
    enough: bool
    missing: list = field(default_factory=list)  # [(unit, have, need, reason)]
    narrowing: float | None = None
    counts: dict = field(default_factory=dict)
    inputs: dict = field(default_factory=dict)  # what evaluate saw, for the fixtures

    @property
    def summary(self):
        return None if self.enough else summary(self.missing)


def evaluate(key, floors, counts, narrowing, previously_shown):
    """`EvidenceGate.evaluate`. floors: a task of DEFAULT_FLOORS; counts {unit: int};
    narrowing None or (realised, rowsToReach | None)."""
    units = sorted(counts, key=_order)
    missing = []
    for unit in units:
        kind = unit[0]
        floor = floors["enough"].get(kind)
        if previously_shown:
            floor = floors["keep"].get(kind, floor)
        if floor is None:
            continue
        have = counts[unit]
        if have < floor:
            reason = "baseline" if kind == "readinessDays" else "floor"
            missing.append((unit, have, floor, reason))
    rho = floors["rhoEnough"]
    if previously_shown and floors["rhoKeep"] is not None:
        rho = floors["rhoKeep"]
    realised = None
    if narrowing is not None:
        realised = narrowing[0]
        if rho is not None and not (realised >= rho):  # NaN counts as not enough
            rows = narrowing[1]
            primary = next((u for u in units if u[0] == floors["primary"]), None)
            if rows is not None and primary is not None:
                have = counts[primary]
                missing.append((primary, have, have + rows, "narrowing"))
            else:
                missing.append((("narrowing",), 0, 0, "narrowing"))
    # §16: every failing requirement shows at least 1 more (the narrowing unit has no number).
    missing = [(u, h, n if u[0] == "narrowing" else max(n, h + 1), r) for u, h, n, r in missing]
    return Gate(
        key=key,
        enough=not missing,
        missing=missing,
        narrowing=None if realised is None or math.isnan(realised) else realised,
        counts={unit_name(u): c for u, c in counts.items()},
        inputs={
            "floors": floors,
            "counts": {unit_name(u): c for u, c in counts.items()},
            "narrowing": None
            if narrowing is None
            else {
                "realised": None if math.isnan(narrowing[0]) else narrowing[0],
                "rowsToReach": narrowing[1],
            },
            "previouslyShown": previously_shown,
        },
    )


def merged(missing):
    """One requirement per unit: first position, largest need − have."""
    out = []
    for req in missing:
        i = next((k for k, r in enumerate(out) if r[0] == req[0]), None)
        if i is None:
            out.append(req)
        elif req[2] - req[1] > out[i][2] - out[i][1]:
            out[i] = req
    return out


def _more(n, one, many):
    return f"{n} more {one if n == 1 else many}"


def _listed(items):
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def summary(missing):
    phrases, fields, at = [], [], None
    for unit, have, need, _ in merged(missing):
        n, kind = need - have, unit[0]
        if kind == "maxAttempts":
            phrases.append(_more(n, "max attempt", "max attempts") + f" at {MODE_NAME[unit[1]]}")
        elif kind == "sessionsWithOximeter":
            phrases.append(_more(n, "session", "sessions") + " with this oximeter")
        elif kind == "holdsWithSpO2":
            phrases.append(_more(n, "hold", "holds") + " with SpO₂")
        elif kind == "holdsWithContractions":
            phrases.append(_more(n, "hold", "holds") + " with contractions marked")
        elif kind == "confirmedHolds":
            phrases.append(
                f"{n} more confirmed {MODE_NAME[unit[1]]} {'hold' if n == 1 else 'holds'}"
            )
        elif kind == "profileField":
            if at is None:
                at = len(phrases)
                phrases.append("")
            fields.append(FIELD_NAME.get(unit[1], unit[1]))
        elif kind == "lungVolume":
            phrases.append(f"needs a {VOLUME_NAME.get(unit[1], unit[1])}")
        elif kind == "readinessDays":
            phrases.append(
                f"wait until {need} {'day' if need == 1 else 'days'} of readiness data with one "
                "source"
            )
        else:
            phrases.append("these max attempts vary too much to narrow the estimate yet")
    if at is not None:
        phrases[at] = f"needs your {_listed(fields)}"
    return "Not enough data yet: " + ", ".join(phrases) + "."


def gate_json(gate):
    return {
        "key": gate.key,
        "state": "enough" if gate.enough else "notEnough",
        "missing": [
            {"unit": unit_name(u), "have": h, "need": n, "reason": r} for u, h, n, r in gate.missing
        ],
        "narrowing": gate.narrowing,
        "counts": gate.counts,
        "summary": gate.summary,
    }
