"""Parity and numerics checks for the v9.00 harness, on hand-built sessions (no private data)."""

import copy
import math
import random

import pytest
from archive import lowest_spo2, reoxygenation_target, time_to_reoxygenation
from evidence import DEFAULT_FLOORS, evaluate, summary
from features import (
    diver_axes,
    hold_facts,
    hold_features,
    hold_model_rows,
    hr_baseline,
    name_tokens,
    oximeter_key,
    recovery_trace,
)
from fit_delay import post_hold_delay
from fit_hold_model import base_config, fit
from numerics import (
    T90,
    Z90,
    log_survival,
    mills,
    newton1d,
    nig_mean_interval,
    nig_update,
    t90,
    weighted_median,
)
from scipy import stats

# ---------------------------------------------------------------- builders


def v(at, spo2=None, hr=None, contact=True):
    out = {"at": at, "contactDetected": contact}
    if spo2 is not None:
        out["spo2"] = spo2
    if hr is not None:
        out["heartRate"] = hr
    return out


def st(kind, actual, vitals=(), sid=None, **kw):
    return {
        "id": sid or f"{kind}-{actual}-{random.random()}",
        "kindName": kind,
        "actualDuration": actual,
        "outcome": kw.pop("outcome", {"type": "completed"}),
        "contractions": kw.pop("contractions", []),
        "vitals": list(vitals),
        **kw,
    }


def session(steps, start="2026-09-10T18:00:00Z", sid="S1", **kw):
    return {
        "id": sid,
        "planName": kw.pop("planName", "Table"),
        "startedAt": start,
        "endedAt": kw.pop("endedAt", start),
        "steps": steps,
        "tags": kw.pop("tags", []),
        **kw,
    }


# ---------------------------------------------------------------- §3 recovery trace


def test_recovery_onset_after_last_minimum_and_walk_to_next_performed_hold():
    s = session(
        [
            st("hold", 60, style="rv"),
            st("breathe", 7, [v(t, x) for t, x in enumerate([95, 90, 88, 89, 88, 90, 91])]),
            st("hold", 0, style="rv"),  # zero-length: not a boundary
            st("rest", 10, [v(0, 92), v(1, 80, contact=False)]),
            st("hold", 50, style="rv", vitals=[v(0, 99)]),
        ]
    )
    trace = recovery_trace(0, s)
    # the 89 bounce after the first 88 is not the onset; 90 after the last 88 is
    assert trace == {"onsetSeconds": 5, "traceSeconds": 7, "version": 1}
    assert recovery_trace(1, s) is None
    assert recovery_trace(2, s) is None  # not performed


def test_recovery_without_rise_is_censored_when_short():
    s = session([st("hold", 60), st("breathe", 20, [v(0, 95), v(10, 90), v(19, 89)])])
    assert recovery_trace(0, s) == {"onsetSeconds": None, "traceSeconds": 19, "version": 1}


def test_stamped_trace_wins():
    from features import stamped_trace

    s = session(
        [st("hold", 60, recoveryTrace={"onsetSeconds": 33, "traceSeconds": 120, "version": 1})]
    )
    assert stamped_trace(0, s)["onsetSeconds"] == 33


# ---------------------------------------------------------------- Session.swift helpers


def test_reoxygenation_target_caps_at_99_and_stops_at_previous_hold():
    steps = [
        st("hold", 30, [v(0, 97)]),
        st("breathe", 60, [v(0, 100), v(50, 100), v(55, 98, contact=False)]),
        st("finalBreath", 5),
        st("hold", 60),
    ]
    assert reoxygenation_target(3, steps) == 99.0
    assert reoxygenation_target(1, steps) is None  # the hold before stops the search


def test_time_to_reoxygenation_from_last_lowest_reading():
    s = session(
        [
            st("breathe", 30, [v(25, 98)]),
            st("hold", 90),
            st("breathe", 10, [v(0, 98), v(4, 90), v(6, 97), v(8, 90)]),
            st("breathe", 10, [v(2, 97), v(5, 98)]),
        ]
    )
    # target 98; lowest 90 last seen at 8; first >= 98 after it: offset 10 + 5
    assert time_to_reoxygenation(s, 1) == 15
    assert time_to_reoxygenation(s, 0) is None


def test_lowest_spo2_first_block_includes_breathe_up_and_zero_hold_is_no_boundary():
    s = session(
        [
            st("breathe", 30, [v(0, 85)]),
            st("hold", 60, [v(0, 97)]),
            st("hold", 0),
            st("breathe", 30, [v(0, 80)]),
            st("hold", 60, [v(0, 70)]),
        ]
    )
    assert lowest_spo2([1], s) == 80
    assert lowest_spo2([4], s) == 70


# ---------------------------------------------------------------- §6 facts and features


def test_facts_peak_is_last_of_equal_maxima_and_rest_resets_on_every_hold():
    hold = st(
        "hold",
        200,
        [
            v(0, 98),
            v(10, 99),
            v(20, 99),
            v(30, 100, contact=False),
            v(100, 95),
            v(140, 88),
            v(170, 80),
        ],
        plannedDuration=210,
    )
    s = session(
        [st("breathe", 120), hold, st("rest", 0), st("hold", 0), st("breathe", 30), st("hold", 40)]
    )
    f = hold_facts(s)
    assert [x["restBefore"] for x in f] == [120, 0, 30]
    assert [x["planned"] for x in f] == [210, 0, 40]  # planned falls back to actual
    assert f[0]["baseline"] == 99 and f[0]["fallBy120"] == 4 and f[0]["fallBy150"] == 11
    assert f[0]["minBy180"] == 80 and f[0]["spo2Seconds"] == 170
    assert f[1]["baseline"] is None


def test_name_tokens_are_whole_words():
    assert name_tokens("Florian 2:00") == {"florian"}
    assert name_tokens("W1 - Ex 2 (FL)") == {"w", "ex", "fl"}


def test_feature_vector():
    s = session(
        [
            st("hold", 60, [v(t, 98 - t // 20) for t in range(0, 70, 5)], plannedDuration=60),
            st("breathe", 99),
            st("hold", 120, plannedDuration=120),
        ],
        planName="RV ladder",
    )
    (_, a), (_, b) = hold_features(s)
    assert a[:7] == [math.log(60), 0.5, 0.0, 1.0, 0.0, 0.0, 0.0]
    assert b[:7] == [math.log(120), 1.0, 0.5, 0.0, 1.0, math.log(100), math.log(120) - math.log(60)]
    assert a[7:10] == [1.0, 0.0, 0.0]
    assert a[10:] == [1.0, 98, 3, 3, 95] and b[10:] == [0.0, 0, 0, 0, 0]


# ---------------------------------------------------------------- §7 hrBaseline


def test_hr_baseline_counts_seconds_back_across_steps():
    s = session(
        [
            st("hold", 60),
            st("breathe", 40, [v(5, None, 80), v(15, None, 70), v(35, None, 60)]),
            st("finalBreath", 7, [v(3, None, 50)]),
            st("hold", 90),
        ]
    )
    # finalBreath sample 4 s before, breathe at 35 lies 7 + 5 = 12 s before, at 15 lies 32 s
    assert hr_baseline(s, 3) == 55.0


# ---------------------------------------------------------------- §2 and §10 rows


def test_oximeter_key_rule_b():
    roster = [{"id": "X", "kind": "oximeter"}]
    s = session([st("hold", 60, [v(0, 97)])], start="2026-09-03T10:00:00Z")
    assert oximeter_key(s, roster) == "X"
    assert (
        oximeter_key(session([st("hold", 60, [v(0, 97)])], start="2026-09-01T10:00:00Z"), roster)
        is None
    )
    assert oximeter_key(dict(s, tags=["imported", "stamina"]), roster) == "stamina"
    assert oximeter_key(s, roster + [{"id": "Y", "kind": "oximeter"}]) is None


def test_hold_model_rows_v2():
    s = session(
        [
            st("hold", 300, style="full"),  # open, completed, not in a series: uncensored
            st(
                "hold", 310, style="packed", plannedDuration=310
            ),  # timed completed: longer, censored
            st("hold", 140, style="rv", plannedDuration=None, iterationCount=5),  # series
            st("hold", 120, style="rv", plannedDuration=120, maxAttempt=True),
            st("hold", 312, style="full", plannedDuration=332, outcome={"type": "skipped"}),
        ]
    )
    rows = [(r["stepIndex"], r["mode"], r["censored"]) for r in hold_model_rows(s)]
    assert rows == [(2, "rv", True), (3, "rv", True), (4, "full", False)]
    s["steps"].pop()  # without the 312 s hold the packed 310 s timed hold takes the full slot
    assert [(r["stepIndex"], r["censored"]) for r in hold_model_rows(s) if r["mode"] == "full"] == [
        (1, True)
    ]
    s2 = session([st("hold", 312, style="full", plannedDuration=332, outcome={"type": "skipped"})])
    assert [r["censored"] for r in hold_model_rows(s2)] == [False]
    assert hold_model_rows(dict(s2, feedback={"symptoms": ["hypoxia"]})) == []
    assert hold_model_rows(dict(s2, tags=["imported", "stamina"])) == []


# ---------------------------------------------------------------- numerics


def test_t90_table_is_scipy_and_interpolates():
    assert list(T90) == [float(stats.t.ppf(0.9, df)) for df in range(1, 31)]
    assert t90(2.5) == T90[1] + 0.5 * (T90[2] - T90[1])
    assert t90(0.3) == T90[0] and t90(30) == T90[29] and t90(30.5) == Z90


def test_nig_update_by_hand():
    post = nig_update((35.0, 2.0, 2.0, 100.0), [30.0, 40.0])
    assert post == (35.0, 4.0, 3.0, 125.0)
    assert nig_update((1.0, 2.0, 3.0, 4.0), []) == (1.0, 2.0, 3.0, 4.0)
    lo, mid, hi = nig_mean_interval(post)
    assert mid == 35.0 and hi - mid == pytest.approx(t90(6) * math.sqrt(125 / 12))


def test_newton1d():
    x, h = newton1d(lambda x: (-((x - 3) ** 2), -2 * (x - 3), -2.0), 0.0)
    assert x == pytest.approx(3.0, abs=1e-12) and h == -2.0
    assert newton1d(lambda x: (x * x, 2 * x, 2.0), 1.0) is None

    # censored-only rows, the case that needs the 4-ulp slack
    data = [5.771441123130016, 5.7430031878094825, 5.7430031878094825]
    data += [5.805134968916488, 5.805134968916488]
    sd, m0, s0 = 0.15, 5.805134968916488, 0.5

    def fgh(mu):
        f, g, h = -((mu - m0) ** 2) / (2 * s0 * s0), -(mu - m0) / s0**2, -1 / s0**2
        for y in data:
            z = (y - mu) / sd
            lam = mills(z)
            f, g, h = f + log_survival(z), g + lam / sd, h - lam * (lam - z) / sd**2
        return f, g, h

    assert newton1d(fgh, m0)[0] == pytest.approx(6.107611435506671, abs=1e-9)


def test_log_survival_branches_meet():
    assert log_survival(29.999999) == pytest.approx(log_survival(30.0), rel=1e-6)
    assert log_survival(0.0) == pytest.approx(math.log(0.5))
    assert mills(0.0) == pytest.approx(math.sqrt(2 / math.pi))


def test_weighted_median_ties_and_half():
    assert weighted_median([3, 1, 2], [1, 1, 1]) == 2
    assert weighted_median([1, 2], [1, 1]) == 1  # cumulative 1 reaches half of 2
    assert weighted_median([], []) is None


# ---------------------------------------------------------------- §13 gate


def delay_gate(n, rho, shown):
    return evaluate(
        "delay.x", DEFAULT_FLOORS["delay"], {("sessionsWithOximeter", "x"): n}, (rho, 1), shown
    )


def test_gate_flicker():
    seq = [(3, 0.12), (3, 0.08), (3, 0.12)]
    shown, with_memory = False, []
    for n, rho in seq:
        shown = delay_gate(n, rho, shown).enough
        with_memory.append(shown)
    assert with_memory == [True, True, True]
    assert [delay_gate(n, rho, False).enough for n, rho in seq] == [True, False, True]


def test_gate_requirements_and_summary():
    g = evaluate(
        "level.frc", DEFAULT_FLOORS["level"], {("maxAttempts", "frc"): 2}, (0.05, 4), False
    )
    assert g.summary == "Not enough data yet: 4 more max attempts at FRC."
    g = evaluate("pb.rv", DEFAULT_FLOORS["pb"], {("maxAttempts", "rv"): 6}, (0.09, 0), False)
    assert g.missing == [(("maxAttempts", "rv"), 6, 7, "narrowing")]  # clamped to 1 more
    assert g.summary == "Not enough data yet: 1 more max attempt at RV."
    g = evaluate(
        "pb.rv", DEFAULT_FLOORS["pb"], {("maxAttempts", "rv"): 6}, (float("nan"), None), False
    )
    assert (
        g.summary
        == "Not enough data yet: these max attempts vary too much to narrow the estimate yet."
    )
    fields = {
        ("profileField", "weightKg"): 0,
        ("profileField", "heightCm"): 0,
        ("lungVolume", "tlc"): 0,
    }
    g = evaluate("oxygenStore", DEFAULT_FLOORS["oxygenStore"], fields, None, False)
    assert g.summary == "Not enough data yet: needs your height and weight, needs a TLC."
    assert summary([(("readinessDays",), 9, 14, "baseline")]) == (
        "Not enough data yet: wait until 14 days of readiness data with one source."
    )


# ---------------------------------------------------------------- determinism


def corpus():
    random.seed(1)
    out = []
    for k, day in enumerate(["2026-09-03", "2026-09-05", "2026-09-08", "2026-09-12", "2026-09-20"]):
        steps = []
        for j in range(4):
            steps.append(st("breathe", 60 + 10 * j, [v(55 + 10 * j, 98, 60 + j)], sid=f"b{j}"))
            steps.append(
                st(
                    "hold",
                    100 + 30 * j + k,
                    [v(t, 98 - t // 15, 60 - t // 30) for t in range(0, 100 + 30 * j, 5)],
                    sid=f"h{j}",
                    style="rv" if j < 3 else "full",
                    plannedDuration=None if j == 3 else 100 + 30 * j,
                    contractions=[40 + j, 60],
                )
            )
            steps.append(
                st(
                    "breathe",
                    60,
                    [v(t, min(99, 80 + t // 2)) for t in range(0, 60, 3)],
                    sid=f"r{j}",
                )
            )
        out.append(session(steps, start=f"{day}T18:00:00Z", sid=f"S{k}"))
    return out


def test_determinism_under_shuffle_and_duplication():
    base = corpus()
    noisy = copy.deepcopy(base) + copy.deepcopy(base[:2])
    random.Random(3).shuffle(noisy)
    roster = [{"id": "pm100", "kind": "oximeter"}]
    prior = (30.0, 2.0, 2.0, 100.0)
    a = post_hold_delay(base, "pm100", prior, roster)
    b = post_hold_delay(noisy, "pm100", prior, roster)
    assert a["posterior"] == b["posterior"] and a["sessions"] == 5
    ma, mb = fit(base, base_config(180.0)), fit(noisy, base_config(180.0))
    assert ma["post"] == mb["post"] and ma["sigma2"] == mb["sigma2"]
    as_of = ma["t_now"]
    assert diver_axes(base, as_of, 180.0) == diver_axes(noisy, as_of, 180.0)


def test_profile_gates_follow_identifiers_and_skip_tlc_without_fields():
    from evaluate import profile_gates

    spiro, o2 = profile_gates(None, None, None)
    assert spiro.summary == "Not enough data yet: needs your date of birth, height and sex."
    assert o2.summary == "Not enough data yet: needs your date of birth, height, sex and weight."


def test_personalised_physiology_replaces_end_saturation_and_spread():
    from fit_oxygen_store import physiology_prior, point_constants, stores
    from static_volumes import ratios_literal

    c = dict(point_constants(), arterialFraction=0.6, sigmaMC=0.3, sigmaMCPersonalised=0.2)
    c["sigmaFloor"] = 0.25
    prof = {"sex": "male", "age": 35, "heightCm": 180, "weightKg": 75}
    prof["measured"] = {"vc": 6.0, "tlc": 7.5}
    pop = physiology_prior(c, prof, None, ratios_literal())
    own = physiology_prior(c, prof, None, ratios_literal(), 60.0)
    assert pop["full"][1] == 0.3 and own["full"][1] == 0.25
    expected = stores(dict(c, arterialSaturationEnd=0.6), prof, {"full": 7.5})["T"]["full"]
    assert own["full"][0] == math.log(expected) and own["full"][0] != pop["full"][0]


@pytest.mark.parametrize(
    ("z", "rel"),
    [
        (-5.0, 1e-12),
        (-10.0, 1e-12),
        (-37.0, 1e-12),
        (-0.5, 1e-12),
        (2.0, 1e-12),
        (35.0, 1e-10),
    ],  # z ≥ 30 is the truncated asymptotic series
)
def test_log_survival_against_mpmath(z, rel):
    import mpmath

    mpmath.mp.dps = 50
    exact = mpmath.log1p(-mpmath.ncdf(z)) if z < 0 else mpmath.log(mpmath.ncdf(-z))
    assert log_survival(z) == pytest.approx(float(exact), rel=rel, abs=0)


def test_censored_stamp_without_onset_key():
    from features import hold_markers, is_censored_trace, stamped_trace
    from fit_delay import session_rows

    stamp = {"traceSeconds": 40, "version": 1}  # nil onset is omitted on the wire
    s = session(
        [st("hold", 60, [v(0, 97)], recoveryTrace=stamp), st("breathe", 40, [v(0, 90)])],
        start="2026-09-10T18:00:00Z",
    )
    assert stamped_trace(0, s) == {"onsetSeconds": None, "traceSeconds": 40, "version": 1}
    assert is_censored_trace(stamped_trace(0, s))
    assert hold_markers(s)[0]["recoveryOnsetSeconds"] is None
    roster = [{"id": "pm100", "kind": "oximeter"}]
    assert [(x, h, c) for _, x, h, c in session_rows([s], "pm100", roster)] == [(None, 0, 1)]


def test_admitted_inputs_get_coefficients_through_fit_and_emit():
    from emit_swift import swift
    from evidence import DEFAULT_FLOORS
    from fit_hold_model import with_coefficients

    sessions = corpus()
    cfg = base_config(None)
    cfg["covariates"], cfg["axes"] = ["logDaysSinceLastSession"], ["hypoxicTolerance"]
    cfg = with_coefficients(sessions, cfg)
    assert len(cfg["beta"]) == 1 and len(cfg["gamma"]) == 1
    assert len(cfg["betaCov"]) == 1 and len(cfg["gammaCov"][0]) == 1
    model = fit(sessions, cfg)
    assert model["post"]
    text = swift(
        {
            "calibration": {"header": "test", "floors": DEFAULT_FLOORS},
            "delay_prior": (30.0, 2.0, 2.0, 100.0),
            "hold_volume": {"model": None},
            "spirometry": None,
            "hold_model": {"config": cfg},
        },
        "test",
    )
    assert 'covariates: ["logDaysSinceLastSession"]' in text and "beta: []" not in text
    assert 'axes: ["hypoxicTolerance"]' in text and "gamma: []" not in text


def test_measured_tlc_spreads_are_the_ratio_sds():
    from static_volumes import ratios_literal, static_volumes

    ratios = ratios_literal()
    prof = {"sex": "male", "age": 40, "heightCm": 180, "measured": {"vc": 6.0, "tlc": 7.5}}
    sv = static_volumes(prof, None, ratios)
    assert sv["FRC"][1] == ratios["frcLogSD"] and sv["RV"][1] == ratios["rvLogSD"]
    est = static_volumes({**prof, "measured": {"vc": 6.0}}, None, ratios)
    assert est["TLC"][1] == ratios["tlcLogSD"]
    assert est["FRC"][1] == math.hypot(ratios["tlcLogSD"], ratios["frcLogSD"])
