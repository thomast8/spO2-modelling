"""Six small synthetic sessions for features.json's inline case (contract §16 layout).

Hand-built by the Nadir side for its parity suite and kept here so emit_fixtures.py is
reproducible anywhere: deterministic, no real recordings. `build()` returns
(sessions, roster, time_zone).
"""

PM100 = "6F1C2D3E-4B5A-4C6D-8E7F-90A1B2C3D4E5"
ROSTER = [
    {"id": PM100, "kind": "oximeter"},
    {"id": "0A0B0C0D-0000-4000-8000-000000000001", "kind": "heartRate"},
]
TIME_ZONE = "Europe/London"


def outcome(kind="completed", reason=None):
    return {"type": kind} if reason is None else {"type": kind, "reason": reason}


def step(id, kind, seconds, style=None, planned=None, vitals=(), contractions=(), **extra):
    s = {"id": id, "kindName": kind}
    if style is not None:
        s["style"] = style
    if planned is not None:
        s["plannedDuration"] = planned
    s["actualDuration"] = seconds
    s["outcome"] = extra.pop("outcome", outcome())
    s["contractions"] = list(contractions)
    s["vitals"] = list(vitals)
    s.update(extra)
    return s


def trace(points, hr=None, contact=True):
    """points: [(at, spo2|None)], hr: {at: bpm}."""
    hr = hr or {}
    out, seen = [], set()
    for at, spo2 in points:
        v = {"at": at, "contactDetected": contact}
        if spo2 is not None:
            v["spo2"] = spo2
        if at in hr:
            v["heartRate"] = hr[at]
        out.append(v)
        seen.add(at)
    for at, bpm in sorted(hr.items()):
        if at not in seen:
            out.append({"at": at, "heartRate": bpm, "contactDetected": contact})
    return sorted(out, key=lambda v: v["at"])


def falling(start, rate, seconds, every, floor=60):
    return [(t, max(floor, round(start - rate * t))) for t in range(0, seconds + 1, every)]


def recovery(low, start, dip_at, back_at, seconds, every):
    pts = []
    for t in range(0, seconds + 1, every):
        if t <= dip_at:
            x = start - (start - low) * t / dip_at
        else:
            x = low + (98 - low) * min(1, (t - dip_at) / max(1, back_at - dip_at))
        pts.append((t, round(x)))
    return pts


def heart(seconds, every, base, drop=0):
    return {t: round(base - drop * t / max(seconds, 1)) for t in range(0, seconds + 1, every)}


def build():
    sessions = []

    # 1. A PM100 RV table with no recording (rule B), a series of four timed holds.
    steps = [
        step(
            "0#-",
            "breathe",
            120,
            "calm",
            120,
            trace([(t, 97) for t in range(60, 121, 10)], heart(120, 5, 64)),
        )
    ]
    for i in range(4):
        last = i == 3
        hold_extra = {"iteration": i + 1, "iterationCount": 4}
        if last:
            hold_extra.update(outcome=outcome("endedEarly", "tappedOut"), endTrigger="tap")
        steps += [
            step(f"1.{i}.0#{i}", "finalBreath", 4, "rv", 4),
            step(
                f"1.{i}.1#{i}",
                "hold",
                55 if last else 60,
                "rv",
                60,
                trace(falling(97 - i, 0.35, 55 if last else 60, 3), heart(60, 5, 62 - i, 14)),
                contractions=[35 + i * 2, 48 + i],
                **hold_extra,
            ),
            step(
                f"1.{i}.2#{i}",
                "breathe",
                90,
                "recovery",
                90,
                trace(recovery(74 - 2 * i, 78 - i, 15 + 3 * i, 60, 90, 5)),
            ),
        ]
    sessions.append(
        {
            "id": "learning-pm100-rv-table",
            "planName": "RV table",
            "startedAt": "2026-09-10T18:00:00Z",
            "endedAt": "2026-09-10T18:16:00Z",
            "steps": steps,
            "tags": [],
            "completed": True,
        }
    )

    # 2. A packed full-lung max with its oximeter recorded and a short trace after it.
    steps = [
        step(
            "0#-",
            "breathe",
            180,
            "calm",
            180,
            trace([(t, 98) for t in range(120, 181, 15)], heart(180, 6, 58)),
        ),
        step(
            "1#-", "breathe", 30, "hyperventilation", 30, trace([], {5: 66, 15: 70, 25: 72, 29: 74})
        ),
        step("2.0#-", "finalBreath", 10, "packed", 10),
        step(
            "2.1#-",
            "hold",
            245,
            "packed",
            None,
            trace(falling(99, 0.06, 245, 7), heart(245, 7, 70, 22)),
            contractions=[150, 175, 200, 220, 235],
            urges=[210],
            maxAttempt=True,
            packCount=6,
            effort=9,
        ),
        step(
            "2.2#-",
            "breathe",
            25,
            "recovery",
            25,
            trace([(0, 86), (6, 83), (12, 81), (18, 81), (22, 82)]),
        ),
        step("3#-", "rest", 60, None, 60),
    ]
    sessions.append(
        {
            "id": "learning-full-max",
            "planName": "Max attempt",
            "startedAt": "2026-09-20T07:00:00Z",
            "endedAt": "2026-09-20T07:09:10Z",
            "steps": steps,
            "tags": [],
            "completed": True,
            "recording": {"device": "phone", "sensors": ["PM100"], "oximeter": PM100},
        }
    )

    # 3. A STAmina (Masimo) import: imported styles, one hold the diver confirmed as RV.
    steps = []
    for i, (planned, actual, source, style) in enumerate(
        [
            (90, 90, "imported", "full"),
            (105, 105, "imported", "full"),
            (120, 118, "confirmed", "rv"),
            (135, 64, "imported", "full"),
        ]
    ):
        steps += [
            step(
                f"{2 * i}#-",
                "breathe",
                120,
                "recovery",
                120,
                trace(
                    [(t, 97) for t in range(0, 121, 8)]
                    if i == 0
                    else recovery(88 - 2 * i, 93, 12 + 3 * i, 50, 120, 6),
                    heart(120, 8, 66),
                ),
            ),
            step(
                f"{2 * i + 1}#-",
                "hold",
                actual,
                style,
                planned,
                trace(falling(98, 0.12 + 0.03 * i, actual, 5), heart(actual, 5, 64, 10)),
                styleSource=source,
                outcome=outcome("skipped") if actual < planned - 1 else outcome(),
            ),
        ]
    sessions.append(
        {
            "id": "stamina-2026-08-20-o2",
            "planName": "O2 Table EXHALE",
            "startedAt": "2026-08-20T23:00:00Z",
            "endedAt": "2026-08-20T23:15:17Z",
            "steps": steps,
            "tags": ["imported", "stamina"],
            "completed": True,
        }
    )

    # 4. The alternating FRC / full-lung session with heart rate only: no SpO₂ anywhere.
    steps = [
        step("0#-", "breathe", 120, "calm", 120, trace([], heart(120, 4, 61))),
        step(
            "1#-",
            "hold",
            90,
            "frc",
            90,
            trace([], heart(90, 3, 60, 8)),
            contractions=[70],
            endTrigger="targetTime",
        ),
        step("2#-", "breathe", 120, "recovery", 120, trace([], heart(120, 4, 72, -10))),
        step(
            "3#-",
            "hold",
            150,
            "full",
            120,
            trace([], heart(150, 3, 62, 12)),
            contractions=[95, 120, 141],
            continuePastTarget=True,
            endTrigger="tap",
            targetReachedAtMilliseconds=120000,
        ),
        step("4#-", "breathe", 120, "recovery", 120, trace([], heart(120, 4, 75, -12))),
        step(
            "5#-",
            "hold",
            70,
            "frc",
            90,
            trace([], heart(70, 2, 63, 9)),
            outcome=outcome("endedEarly", "safetyFloor"),
            endTrigger="safetyFloor",
        ),
        step("6#-", "hold", 0, "full", 60, outcome=outcome("skipped")),
    ]
    sessions.append(
        {
            "id": "learning-frc-full",
            "planName": "FRC + FL alternating",
            "startedAt": "2026-10-02T06:15:00Z",
            "endedAt": "2026-10-02T06:26:50Z",
            "steps": steps,
            "tags": [],
            "completed": True,
            "recording": {"device": "watch", "sensors": ["Apple Watch"]},
        }
    )

    # 5. Unfinished: only ever a previous session that does not count.
    sessions.append(
        {
            "id": "learning-unfinished",
            "planName": "Max attempt",
            "startedAt": "2026-10-05T07:00:00Z",
            "steps": [step("0#-", "hold", 200, "full", None)],
            "tags": [],
            "completed": False,
        }
    )

    # 6. After the clocks went back, a max with hypoxia reported, and a censored trace.
    steps = [
        step(
            "0#-",
            "breathe",
            150,
            "calm",
            150,
            trace([(t, 98) for t in range(100, 151, 10)], heart(150, 5, 60)),
        ),
        step(
            "1#-",
            "hold",
            200,
            "full",
            None,
            trace(falling(98, 0.09, 200, 6), heart(200, 6, 61, 15)),
            contractions=[130, 160],
        ),
        step("2#-", "breathe", 40, "recovery", 40, trace([(0, 82), (10, 80), (20, 80), (30, 79)])),
    ]
    sessions.append(
        {
            "id": "learning-hypoxia",
            "planName": "Max attempt",
            "startedAt": "2026-11-03T08:00:00Z",
            "endedAt": "2026-11-03T08:06:30Z",
            "steps": steps,
            "tags": [],
            "completed": True,
            "feedback": {"difficulty": "hard", "symptoms": ["hypoxia"]},
        }
    )
    return sessions, ROSTER, TIME_ZONE
