"""Loader checks on synthetic exports. Run from backend/:

uv run --extra dev pytest scripts/experiments/exp_v8_00/test_nadir_export.py
"""

import json

import numpy as np
from nadir_export import load_nadir, load_stamina, qc


def _vitals(spo2s, contact=True):
    return [
        {"at": i, "contactDetected": contact, "heartRate": 60, "spo2": int(s)}
        for i, s in enumerate(spo2s)
    ]


def _hold(duration, vitals, style="rv"):
    return {
        "kindName": "hold",
        "actualDuration": duration,
        "style": style,
        "vitals": vitals,
        "outcome": {"type": "completed"},
        "contractions": [],
    }


def _step(kind, duration, vitals):
    return {"kindName": kind, "actualDuration": duration, "vitals": vitals}


def _descent(n, start=99, end=80):
    return np.linspace(start, end, n).round()


def _export(tmp_path, *sessions):
    path = tmp_path / "export.json"
    path.write_text(json.dumps({"sessions": list(sessions)}))
    return path


def _session(started, *steps):
    return {"startedAt": started, "steps": list(steps), "tags": []}


def test_terminal_dropout_counts_as_missing_and_keeps_the_hold_length(tmp_path):
    vitals = _vitals(_descent(60)) + [
        {"at": 60 + i, "contactDetected": False, "heartRate": 60, "spo2": 80} for i in range(60)
    ]
    path = _export(tmp_path, _session("2026-09-01T10:00:00Z", _hold(120, vitals)))
    (h,) = load_nadir(path)
    assert h["t_end"] == 119
    assert len(h["t_apnea"]) == 120
    assert h["meta"]["missing_frac"] == 0.5
    assert h["meta"]["max_gap"] >= 60
    assert qc(h) is not None


def test_two_sessions_on_one_day_stay_apart(tmp_path):
    hold = _hold(60, _vitals(_descent(60)))
    path = _export(
        tmp_path,
        _session("2026-08-21T15:00:00Z", hold),
        _session("2026-08-21T16:30:00Z", hold),
    )
    holds = load_nadir(path)
    assert len({h["meta"]["session"] for h in holds}) == 2
    assert len({h["id"] for h in holds}) == 2


def test_recovery_stops_at_the_next_hold(tmp_path):
    path = _export(
        tmp_path,
        _session(
            "2026-10-02T10:00:00Z",
            _hold(60, _vitals(_descent(60))),
            _step("finalBreath", 7, _vitals([86, 87, 88, 88, 89, 89, 90])),
            _hold(60, _vitals(_descent(60, 90, 80))),
        ),
    )
    first = load_nadir(path)[0]
    assert len(first["t_recovery"]) == 7
    assert first["t_recovery"][-1] == first["t_end"] + 7


def test_recovery_is_on_the_1hz_grid(tmp_path):
    # Steps that record a second or two past their duration (the PM100 sessions do) and a
    # recovery sample without contact.
    rec1 = _vitals([85, 86, 87, 88, 88, 89, 89, 90, 90, 91, 91, 92])
    rec1[4]["contactDetected"] = False
    path = _export(
        tmp_path,
        _session(
            "2026-10-16T10:00:00Z",
            _hold(60, _vitals(_descent(60))),
            _step("breathe", 10, rec1),
            _step("breathe", 20, _vitals(np.full(20, 92))),
        ),
    )
    (h,) = load_nadir(path)
    assert np.all(np.diff(h["t"]) == 1)
    assert h["t_recovery"][0] == h["t_end"] + 1


def _stamina_csv(path):
    rows = ["Biometrics,,", "Time,Interval Time,Interval Type,HR,SpO2"]
    t = 0
    for kind, spo2s in (("Rest", [98] * 20), ("Apnea", _descent(60)), ("Rest", [85] * 20)):
        for i, s in enumerate(spo2s):
            rows.append(f"{t // 60:02d}:{t % 60:02d},{i // 60:02d}:{i % 60:02d},{kind},60,{s:.0f}")
            t += 1
    path.write_text("\n".join(rows) + "\n")
    return path


def test_stamina_lung_volume_comes_from_the_session_not_the_hold_number(tmp_path):
    (feb,) = load_stamina(_stamina_csv(tmp_path / "Saturday, February 21, 2026.csv"))
    assert feb["type"] == "FL"
    assert load_stamina(_stamina_csv(tmp_path / "Sunday, March 1, 2026.csv")) == []
