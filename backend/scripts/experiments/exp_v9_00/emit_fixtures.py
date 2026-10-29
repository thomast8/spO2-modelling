"""Write the parity fixtures of contract §14 for Nadir's `fixtures/learning/`.

Usage (from backend/):
    uv run python scripts/experiments/exp_v9_00/emit_fixtures.py --nhanes-cache <dir> \
        [--out-dir <dir>]

Layout (§16): sessions.json holds the real sessions once (8, the export's wire encoding with notes
and rrIntervals removed); every other case carries "sessionIDs" resolved against it, or inline
"sessions" when hand-built (empty for gate-only and profile-only cases), never both. Units are §16
strings; lung modes are "full" | "frc" | "rv".
"""

import copy
import json
import math
from datetime import timedelta
from pathlib import Path

import hand_built
import numpy as np
from archive import HARNESS_ROSTER, HARNESS_TIME_ZONE, format_date, is_performed_hold, started
from evaluate import cli, run
from evidence import evaluate, gate_json, unit_name
from features import (
    AXES,
    AXIS_UNIT,
    diver_axes,
    hold_context,
    hold_features,
    hold_markers,
    hold_model_rows,
    oximeter_key,
    recovery_trace,
)
from fit_delay import KEYS, post_hold_delay
from fit_hold_model import MODES, base_config, estimates, fit
from fit_hold_volume import predict, verdict
from fit_lung_volume import predict as spiro_predict
from fit_oxygen_store import REFERENCE, headroom, physiology_prior, stores, volumes_for
from packing import Packing, packed_gate
from static_volumes import static_volumes

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "out" / "fixtures" / "learning"
# Coverage asked by §14: a Masimo import, the 6 August RV table, a PM100 RV table, a full-lung max
# with a short trace, 2 October, a session without SpO₂ (an uncensored row), a censored hold
# (aborted), and the other uncensored row with SpO₂.
FIXTURE_DAYS = (
    "2026-08-06",
    "2026-08-08",
    "2026-09-05",
    "2026-09-14",
    "2026-09-18",
    "2026-10-02",
    "2026-10-09",
    "2026-10-16",
)
PROFILES = {
    "reference": REFERENCE,
    "female50": {"sex": "female", "age": 50, "heightCm": 165, "weightKg": 60},
    "male22MeasuredVC": {
        "sex": "male",
        "age": 22,
        "heightCm": 175,
        "weightKg": 70,
        "measured": {"vc": 5.2},
    },
    "male72": {"sex": "male", "age": 72, "heightCm": 175, "weightKg": 80},
}


def scrub(session):
    s = copy.deepcopy(session)
    s.pop("notes", None)
    for st in s["steps"]:
        for v in st.get("vitals", []):
            v.pop("rrIntervals", None)
    return s


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def write(out_dir, name, generator, cases):
    path = out_dir / name
    text = json.dumps(
        jsonable({"generator": generator, "cases": cases}),
        separators=(",", ":"),
        ensure_ascii=False,
    )
    path.write_text(text + "\n")
    return path


def write_sessions(out_dir, generator, sessions):
    path = out_dir / "sessions.json"
    text = json.dumps(
        jsonable({"generator": generator, "sessions": sessions}),
        separators=(",", ":"),
        ensure_ascii=False,
    )
    path.write_text(text + "\n")
    return path


def gate_case(name, task, counts, narrowing, shown, floors):
    f = floors[task]
    g = evaluate(name, f, counts, narrowing, shown)
    return {
        "name": name,
        "sessions": [],
        "inputs": {
            "task": task,
            "floors": f,
            "counts": {unit_name(u): c for u, c in counts.items()},
            "narrowing": None
            if narrowing is None
            else {"realised": narrowing[0], "rowsToReach": narrowing[1]},
            "previouslyShown": shown,
        },
        "expected": gate_json(g),
    }


def features_expected(sessions, roster, tz):
    """features.json `expected` in the layout LearningFeatureParityTests reads."""
    out = {
        "oximeterKeys": {},
        "holdFacts": [],
        "recoveryTraces": [],
        "holdMarkers": [],
        "holdContext": [],
        "holdModelRows": [],
    }
    for s in sessions:
        sid = s["id"]
        out["oximeterKeys"][sid] = oximeter_key(s, roster)
        for facts, values in hold_features(s):
            out["holdFacts"].append({**facts, "sessionID": sid, "features": values})
        for i, st in enumerate(s["steps"]):
            if is_performed_hold(st):
                out["recoveryTraces"].append(
                    {"sessionID": sid, "stepIndex": i, "trace": recovery_trace(i, s)}
                )
                out["holdContext"].append(
                    {"sessionID": sid, "stepIndex": i, **hold_context(s, i, sessions, tz)}
                )
        out["holdMarkers"] += [{"sessionID": sid, **m} for m in hold_markers(s)]
        out["holdModelRows"] += [
            {
                "sessionID": sid,
                "stepIndex": r["stepIndex"],
                "mode": r["mode"],
                "seconds": r["actual"],
                "uncensored": not r["censored"],
            }
            for r in hold_model_rows(s)
        ]
    return out


def model_json(model, est):
    return {
        "sigma2": model["sigma2"],
        "an": model["an"],
        "volumes": {
            v: {
                "prior": None
                if v not in model["priors"]
                else {
                    "mean": model["priors"][v][0],
                    "sd": model["priors"][v][1],
                    "source": model["priors"][v][2],
                },
                "posterior": None
                if v not in model["post"]
                else {"mu": model["post"][v][0], "variance": model["post"][v][1]},
                "counts": model["counts"][v],
                "level": est[v]["level"]["estimate"],
                "pb": est[v]["pb"]["estimate"],
                "levelGate": gate_json(est[v]["level"]["gate"]),
                "pbGate": gate_json(est[v]["pb"]["gate"]),
            }
            for v in MODES
        },
    }


def emit(r, out_dir, export_name):
    out_dir.mkdir(parents=True, exist_ok=True)
    sessions = [s for s in r["sessions"] if s["startedAt"][:10] in FIXTURE_DAYS]
    sessions = [s for s in sessions if not s["id"].startswith("stamina-2026-02")]
    wire = [scrub(s) for s in sessions]
    t_now = max(started(s) for s in sessions)
    gen = {
        "script": "spo2-modelling backend/scripts/experiments/exp_v9_00/emit_fixtures.py",
        "contract": "learning-contract v1, §10 v2, §16 (22 October 2026)",
        "source": export_name,
        "timeZone": HARNESS_TIME_ZONE,
        "roster": HARNESS_ROSTER,
        "tolerance": {"closedForm": 1e-9, "newton": 1e-6},
    }
    written = []
    floors = r["calibration"]["floors"]  # the emitted Priors.evidence (§16)

    ids = [x["id"] for x in wire]
    written.append(write_sessions(out_dir, gen, wire))

    # features.json: the real sessions by id, then the Swift agent's hand-built sessions inline
    cases = [
        {
            "name": "real sessions",
            "timeZone": HARNESS_TIME_ZONE,
            "roster": HARNESS_ROSTER,
            "sessionIDs": ids,
            "expected": features_expected(sessions, HARNESS_ROSTER, HARNESS_TIME_ZONE),
        }
    ]
    hand = hand_built.build()
    hb, roster, tz = hand
    cases.append(
        {
            "name": "hand-built",
            "timeZone": tz,
            "roster": roster,
            "sessions": hb,
            "expected": features_expected(hb, roster, tz),
        }
    )
    written.append(write(out_dir, "features.json", gen, cases))

    # post-hold-delay.json: one case, the variants share the sessions
    prior = r["delay_prior"]
    variants, expected = [], []
    for key in KEYS:
        for shown in (False, True):
            d = post_hold_delay(sessions, key, prior, previously_shown=shown, floors=floors)
            variants.append({"key": key, "previouslyShown": shown})
            expected.append(
                {
                    "sessionCount": d["sessions"],
                    "holds": d["holds"],
                    "censored": d["censored"],
                    "posterior": dict(
                        zip(("mu", "kappa", "alpha", "beta"), d["posterior"], strict=True)
                    ),
                    "estimate": d["estimate"],
                    "priorInterval": d["prior_interval"],
                    "narrowing": d["narrowing"],
                    "gate": gate_json(d["gate"]),
                }
            )
    written.append(
        write(
            out_dir,
            "post-hold-delay.json",
            gen,
            [
                {
                    "name": "fixture sessions",
                    "sessionIDs": ids,
                    "inputs": {
                        "prior": dict(
                            zip(("mu0", "kappa0", "alpha0", "beta0"), prior, strict=True)
                        ),
                        "floors": floors["delay"],
                        "roster": HARNESS_ROSTER,
                        "variants": variants,
                    },
                    "expected": expected,
                }
            ],
        )
    )

    # hold-volume.json
    hv = r["hold_volume"]
    if hv["model"] is not None:
        m, (lo, hi) = hv["model"], hv["band"]
        prior_json = {
            "featureMeans": m["means"].tolist(),
            "featureScales": m["scales"].tolist(),
            "mean": m["mean"].tolist(),
            "covariance": m["cov"].tolist(),
            "lowerBand": lo,
            "upperBand": hi,
            "labelled": hv["labelled"],
        }
        verdicts = []
        for s in sessions:
            for f, x in hold_features(s):
                p = predict(m, x)
                verdicts.append(
                    {
                        "sessionID": s["id"],
                        "stepIndex": f["stepIndex"],
                        "stepID": f["stepID"],
                        "probabilityFull": p,
                        "style": verdict(p, lo, hi),
                    }
                )
        written.append(
            write(
                out_dir,
                "hold-volume.json",
                gen,
                [
                    {
                        "name": "fitted model on the fixture sessions",
                        "sessionIDs": ids,
                        "inputs": {"prior": prior_json},
                        "expected": {"verdicts": verdicts, "classifierGate": gate_json(hv["gate"])},
                    }
                ],
            )
        )

    # diver-axes.json: one case, variants by asOf and half-life
    variants, expected = [], []
    for as_of, h in (
        (t_now + timedelta(seconds=1), None),
        (t_now + timedelta(seconds=1), 180.0),
        (started(sessions[4]), None),
    ):
        ax = diver_axes(sessions, as_of, h)
        variants.append({"asOf": format_date(as_of), "halfLifeDays": h})
        expected.append(
            {
                n: {
                    **ax[n],
                    "gate": gate_json(
                        evaluate(
                            f"axis.{n}",
                            floors["axis"],
                            {(AXIS_UNIT[n],): ax[n]["count"]},
                            None,
                            False,
                        )
                    ),
                }
                for n in AXES
            }
        )
    written.append(
        write(
            out_dir,
            "diver-axes.json",
            gen,
            [
                {
                    "name": "fixture sessions",
                    "sessionIDs": ids,
                    "inputs": {"floors": floors["axis"], "variants": variants},
                    "expected": expected,
                }
            ],
        )
    )

    # hold-model.json: the fixture sessions under three variants, plus a no-rows case
    cfg = r["hold_model"]["config"]
    spiro, ratios, o2 = r["spirometry"], r["ratios"], r["oxygen_store"]
    variants, expected = [], []
    runs = [("placeholder prior", None, None, [])]
    if o2 is not None:
        runs.append(
            (
                "physiology prior (reference diver profile)",
                physiology_prior(o2, REFERENCE, spiro, ratios),
                REFERENCE,
                [],
            )
        )
    runs.append(("placeholder prior, full previously shown", None, None, ["pb.full", "level.full"]))
    for name, phys, prof, shown in runs:
        model = fit(sessions, cfg, phys)
        variants.append(
            {
                "name": name,
                "previouslyShown": shown,
                "profile": prof,
                "physiology": None
                if phys is None
                else {v: {"mean": p[0], "sd": p[1]} for v, p in phys.items()},
            }
        )
        expected.append(
            model_json(model, estimates(model, sessions, frozenset(shown), floors=floors))
        )
    model_floors = {"pb": floors["pb"], "level": floors["level"]}
    cases = [
        {
            "name": "fixture sessions",
            "sessionIDs": ids,
            "inputs": {
                "holdModel": cfg,
                "floors": model_floors,
                "timeZone": HARNESS_TIME_ZONE,
                "variants": variants,
            },
            "expected": expected,
        }
    ]
    only_rv = [s for s in sessions if s["startedAt"].startswith("2026-08-08")]  # unconfirmed import
    model = fit(only_rv, base_config())
    cases.append(
        {
            "name": "no rows (an unconfirmed STAmina import): every volume waits on a full-lung "
            "attempt",
            "sessionIDs": [x["id"] for x in only_rv],
            "inputs": {
                "holdModel": base_config(),
                "floors": model_floors,
                "timeZone": HARNESS_TIME_ZONE,
                "variants": [
                    {"name": "placeholder prior", "previouslyShown": [], "physiology": None}
                ],
            },
            "expected": [model_json(model, estimates(model, only_rv, floors=floors))],
        }
    )
    if o2 is not None:
        # The Nadir-side hand-built sessions under the population physiology prior; one of
        # them has a censored FRC row far below its prior (the log1p branch of logSurvival).
        # The second variant adds a synthetic packed measurement (§17): its packed hold with
        # packCount 6 then loses ln(U(V + ΔV) / U(V)) before it informs the full-lung level.
        hb, _, tz = hand
        t_end = max(started(x) for x in hb if x.get("endedAt") is not None)
        profile = dict(REFERENCE, dateOfBirth=f"{t_end.year - REFERENCE['age']}-{t_end:%m-%d}")
        packed_profile = dict(
            profile, measured={**REFERENCE["measured"], "vc": 5.6, "packedVC": 6.8, "packs": 10}
        )
        phys = physiology_prior(o2, profile, spiro, ratios)
        variants, expected = [], []
        for name, prof in (
            ("population physiology prior", profile),
            ("population physiology prior, packed measurement", packed_profile),
        ):
            pk = Packing(prof, o2, spiro, ratios)
            model = fit(hb, cfg, phys, tz, packing=pk)
            variants.append(
                {
                    "name": name,
                    "profile": prof,
                    "previouslyShown": [],
                    "physiology": {v: {"mean": q[0], "sd": q[1]} for v, q in phys.items()},
                    "packing": {
                        "efficiency": pk.e,
                        "holds": [
                            {
                                "sessionID": x["id"],
                                "stepIndex": i,
                                "packCount": st.get("packCount"),
                                "deltaV": pk.delta_v(st.get("packCount")),
                                "offset": pk.offset(st),
                            }
                            for x in hb
                            for i, st in enumerate(x["steps"])
                            if is_performed_hold(st) and st.get("style") == "packed"
                        ],
                    },
                }
            )
            est = estimates(model, hb, floors=floors, tz=tz)
            expected.append(
                {
                    **model_json(model, est),
                    "packedGate": gate_json(packed_gate(est["full"]["pb"]["gate"], pk)),
                }
            )
        cases.append(
            {
                "name": "hand-built, reference profile, population physiology prior",
                "sessions": hb,
                "inputs": {
                    "holdModel": cfg,
                    "floors": model_floors,
                    "timeZone": tz,
                    "variants": variants,
                },
                "expected": expected,
            }
        )
    written.append(write(out_dir, "hold-model.json", gen, cases))

    if spiro is not None:
        cases = []
        for name, p in PROFILES.items():
            exp = {}
            for meas in ("fvc", "fev1"):
                pr = spiro_predict(spiro, p["sex"], p["age"], p["heightCm"], meas)
                exp[meas] = (
                    None
                    if pr is None
                    else {
                        "lower": pr[0],
                        "value": pr[1],
                        "upper": pr[2],
                        "logSD": pr[3],
                        "confidence": 0.9,
                    }
                )
            fvc = exp["fvc"]
            if fvc:
                mu = math.log(fvc["value"])
                measured = 0.9 * fvc["value"]
                exp["compareFVCAt90Percent"] = {
                    "measured": measured,
                    "percent": 100 * measured / fvc["value"],
                    "z": (math.log(measured) - mu) / fvc["logSD"],
                }
            cases.append({"name": name, "sessions": [], "inputs": {"profile": p}, "expected": exp})
        cases.append(
            {
                "name": "age 85 outside the fit",
                "sessions": [],
                "inputs": {"profile": {"sex": "male", "age": 85, "heightCm": 175}},
                "expected": {"fvc": None, "fev1": None},
            }
        )
        written.append(write(out_dir, "spirometry.json", gen, cases))

        cases = []
        for name, p in {**PROFILES, "male80": {"sex": "male", "age": 80, "heightCm": 175}}.items():
            sv = static_volumes(p, spiro, ratios)
            cases.append(
                {
                    "name": name,
                    "sessions": [],
                    "inputs": {"profile": p},
                    "expected": None
                    if sv is None
                    else {
                        k: {"litres": v[0], "logSD": v[1], "source": v[2]} for k, v in sv.items()
                    },
                }
            )
        written.append(write(out_dir, "static-volumes.json", gen, cases))

        cases = []
        for name, p in PROFILES.items():
            vols = volumes_for(p, spiro, ratios)
            if vols is None:
                cases.append(
                    {"name": name, "sessions": [], "inputs": {"profile": p}, "expected": None}
                )
                continue
            st = stores(o2, p, {v: lv for v, (lv, _) in vols.items()})
            cases.append(
                {
                    "name": name,
                    "sessions": [],
                    "inputs": {
                        "profile": p,
                        "volumesLitres": {v: lv for v, (lv, _) in vols.items()},
                    },
                    "expected": {
                        "lungML": st["lung"],
                        "bloodML": st["blood"],
                        "muscleML": st["muscle"],
                        "vo2MLPerMin": st["vo2"],
                        "seconds": st["T"],
                        "usableML": st["usable"],
                        "physiologyPrior": {
                            v: {"mean": q[0], "sd": q[1]}
                            for v, q in physiology_prior(o2, p, spiro, ratios).items()
                        },
                    },
                }
            )
        written.append(write(out_dir, "oxygen-store.json", gen, cases))

        phys = physiology_prior(o2, REFERENCE, spiro, ratios)
        written.append(
            write(
                out_dir,
                "headroom.json",
                gen,
                [
                    {
                        "name": f"reference diver, {v} best {best} s",
                        "sessions": [],
                        "inputs": {
                            "best": best,
                            "lungMode": v,
                            "physiology": {"mean": phys[v][0], "sd": phys[v][1]},
                        },
                        "expected": {"percentile": headroom(best, phys[v])},
                    }
                    for v, best in (("full", 300), ("full", 390), ("rv", 150))
                ],
            )
        )

    # evidence.json: every gate on the export, then hand cases
    cases = []
    for g in r["gates"]:
        inputs = dict(g.inputs)
        re_run = evaluate(
            g.key,
            inputs["floors"],
            {tuple(k.split(".", 1)): c for k, c in inputs["counts"].items()},
            None
            if inputs["narrowing"] is None
            else (
                float("nan")
                if inputs["narrowing"]["realised"] is None
                else inputs["narrowing"]["realised"],
                inputs["narrowing"]["rowsToReach"],
            ),
            inputs["previouslyShown"],
        )
        # HoldModel adds a requirement when a volume has no prior; evaluate alone cannot.
        inputs["taskAddsRequirements"] = re_run.missing != g.missing
        cases.append(
            {"name": f"export: {g.key}", "sessions": [], "inputs": inputs, "expected": gate_json(g)}
        )
    flicker = []
    for shown_memory in (True, False):
        shown, states = False, []
        for n, rho in ((3, 0.12), (3, 0.08), (3, 0.12)):
            g = evaluate(
                "delay.pm100",
                floors["delay"],
                {("sessionsWithOximeter", "pm100"): n},
                (rho, 1),
                shown if shown_memory else False,
            )
            states.append(g.enough)
            shown = g.enough
        flicker.append(states)
    cases.append(
        {
            "name": "flicker: up, slight dip, up again (with / without previouslyShown)",
            "sessions": [],
            "inputs": {
                "sequence": [
                    {"count": 3, "narrowing": 0.12},
                    {"count": 3, "narrowing": 0.08},
                    {"count": 3, "narrowing": 0.12},
                ],
                "rowsToReach": 1,
                "floors": floors["delay"],
            },
            "expected": {"withMemory": flicker[0], "withoutMemory": flicker[1]},
        }
    )
    cases.append(
        gate_case(
            "narrowing unreachable", "pb", {("maxAttempts", "rv"): 6}, (0.02, None), False, floors
        )
    )
    cases.append(
        gate_case(
            "NaN narrowing", "pb", {("maxAttempts", "full"): 6}, (float("nan"), 3), False, floors
        )
    )
    cases.append(
        gate_case(
            "floor and narrowing merge",
            "level",
            {("maxAttempts", "frc"): 2},
            (0.05, 4),
            False,
            floors,
        )
    )
    cases.append(
        gate_case(
            "one more session",
            "delay",
            {("sessionsWithOximeter", "stamina"): 2},
            (0.3, 0),
            False,
            floors,
        )
    )
    cases.append(
        gate_case(
            "keep floor", "delay", {("sessionsWithOximeter", "stamina"): 2}, (0.07, 0), True, floors
        )
    )
    cases.append(
        gate_case(
            "profile fields",
            "oxygenStore",
            {
                ("profileField", "weightKg"): 0,
                ("profileField", "heightCm"): 0,
                ("profileField", "sex"): 1,
                ("profileField", "dateOfBirth"): 1,
            },
            None,
            False,
            floors,
        )
    )
    cases.append(
        gate_case(
            "every field, no TLC (outside the reference ranges)",
            "oxygenStore",
            {
                ("profileField", "sex"): 1,
                ("profileField", "dateOfBirth"): 1,
                ("profileField", "heightCm"): 1,
                ("profileField", "weightKg"): 1,
                ("lungVolume", "tlc"): 0,
            },
            None,
            False,
            floors,
        )
    )
    cases.append(
        gate_case("readiness baseline", "readiness", {("readinessDays",): 9}, None, False, floors)
    )
    cases.append(
        gate_case("one hold with SpO2", "axis", {("holdsWithSpO2",): 4}, None, False, floors)
    )
    written.append(write(out_dir, "evidence.json", gen, cases))
    return written


if __name__ == "__main__":
    p = cli()
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    a = p.parse_args()
    r = run(
        a.export,
        a.stamina,
        a.nhanes_cache,
        json.loads(a.profile) if a.profile else None,
        write=False,
        row_rule=a.row_rule,
    )
    for path in emit(r, Path(a.out_dir), Path(a.export).name):
        print(f"wrote {path} ({path.stat().st_size // 1024} KB)")
