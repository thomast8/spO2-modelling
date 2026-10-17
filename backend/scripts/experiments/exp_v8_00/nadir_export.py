"""Load breath-holds from a Nadir full Export (JSON) or a STAmina CSV into v7 hold dicts.

Both sources end up in the shape `exp_v7_08_ship.load_holds_with_recovery` produces, so the
v7 fit functions run unchanged on them:

    id, type (FL/FRC/RV), t, spo2, hr, t_end, t_apnea, spo2_apnea, t_recovery, spo2_recovery

plus a `meta` dict (session, duration, outcome, coverage) that v7 never had.

Decisions that v7's code forces:

- The gamma-kernel convolution in Stage B assumes a complete 1 Hz grid, so dropouts and
  samples without skin contact are filled by linear interpolation. The fraction filled and
  the longest gap are kept in `meta` and the QC rule rejects holds where it is too much.
- Recovery follows v7's rule: at most 90 s after the hold, stopping once SpO2 is back at
  97 % or more (after the first 5 s). In a Nadir export that means walking across the steps
  that follow the hold, since vitals are stored per step.

Usage:
    uv run python scripts/experiments/exp_v8_00/nadir_export.py <export.json | session.csv> ...
"""

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

RECOVERY_MAX_S = 90
RECOVERY_SPO2_CEILING = 97
MIN_HOLD_S = 30

STYLE_TO_TYPE = {"full": "FL", "frc": "FRC", "rv": "RV"}


def _fill_gaps(t, spo2):
    """Return a complete 1 Hz grid with linear interpolation over missing seconds."""
    grid = np.arange(t[0], t[-1] + 1, 1.0)
    filled = np.interp(grid, t, spo2)
    present = np.isin(grid, t)
    gaps = np.diff(t)
    # A long gap on the 99 % plateau costs nothing; one across the descent invents the curve.
    # v7 itself trained on FL#6 with a 6 s gap near its nadir, so the line is drawn above that.
    off_plateau = any(g > 10 and min(spo2[i], spo2[i + 1]) < 95 for i, g in enumerate(gaps))
    return grid, filled, 1.0 - present.mean(), float(gaps.max()) if len(gaps) else 0.0, off_plateau


def _make_hold(hold_id, hold_type, t_apnea, spo2_apnea, hr_apnea, t_rec, spo2_rec, hr_rec, meta):
    t_end = float(t_apnea[-1])
    t = np.concatenate([t_apnea, t_rec])
    spo2 = np.concatenate([spo2_apnea, spo2_rec])
    return {
        "id": hold_id,
        "type": hold_type,
        "t": t,
        "spo2": spo2,
        "hr": np.concatenate([hr_apnea, hr_rec]),
        "t_end": t_end,
        "t_apnea": t_apnea,
        "spo2_apnea": spo2_apnea,
        "t_recovery": t_rec,
        "spo2_recovery": spo2_rec,
        "meta": meta,
    }


def _trim_recovery(t_rec, spo2_rec, hr_rec):
    keep = []
    for i, (t, s) in enumerate(zip(t_rec, spo2_rec, strict=False)):
        if t > RECOVERY_MAX_S or (t > 5 and s >= RECOVERY_SPO2_CEILING):
            break
        keep.append(i)
    keep = np.array(keep, dtype=int)
    return t_rec[keep], spo2_rec[keep], hr_rec[keep]


# ---------------------------------------------------------------- Nadir export


def load_nadir(path):
    data = json.loads(Path(path).read_text())
    holds = []
    for session in sorted(data["sessions"], key=lambda s: s["startedAt"]):
        day = session["startedAt"][:10]
        steps = session["steps"]
        # A session Nadir rebuilt from a STAmina export was recorded by STAmina (Masimo
        # MightySat), not by the app and its connected oximeter. It is the only device
        # provenance the export carries, and its lung-volume labels are the importer's
        # default (`full`), not the diver's.
        imported = {"imported", "stamina"} <= set(session.get("tags", []))
        k = 0
        for i, step in enumerate(steps):
            if step.get("kindName") != "hold" or step.get("actualDuration", 0) <= 0:
                continue
            k += 1
            good = [
                v
                for v in step.get("vitals", [])
                if v.get("contactDetected", True) and v.get("spo2") is not None
            ]
            if not good:
                continue
            hold_type = STYLE_TO_TYPE.get(step.get("style"))
            if hold_type is None:
                print(f"  skip {day} hold {k}: lung volume {step.get('style')!r} has no v7 bounds")
                continue
            duration = float(step["actualDuration"])
            t_raw = np.array([round(v["at"]) for v in good], dtype=float)
            t_raw, idx = np.unique(t_raw, return_index=True)
            spo2_raw = np.array([good[j]["spo2"] for j in idx], dtype=float)
            hr_raw = np.array([good[j].get("heartRate") or np.nan for j in idx], dtype=float)
            t_apnea, spo2_apnea, missing, max_gap, off_plateau = _fill_gaps(t_raw, spo2_raw)
            hr_apnea = np.interp(t_apnea, t_raw, hr_raw)

            # Recovery: walk the following steps, offsetting each by the time before it.
            offset = duration
            rec = []
            for nxt in steps[i + 1 :]:
                for v in nxt.get("vitals", []):
                    if v.get("contactDetected", True) and v.get("spo2") is not None:
                        rec.append(
                            (
                                offset + round(v["at"]) - t_apnea[-1],
                                v["spo2"],
                                v.get("heartRate") or np.nan,
                            )
                        )
                offset += float(nxt.get("actualDuration", 0))
                if offset - t_apnea[-1] > RECOVERY_MAX_S:
                    break
            rec = sorted(r for r in rec if r[0] >= 1)
            t_rec = np.array([r[0] for r in rec], dtype=float)
            spo2_rec = np.array([r[1] for r in rec], dtype=float)
            hr_rec = np.array([r[2] for r in rec], dtype=float)
            t_rec, spo2_rec, hr_rec = _trim_recovery(t_rec, spo2_rec, hr_rec)
            t_rec += t_apnea[-1]

            outcome = step.get("outcome", {})
            outcome = (
                outcome.get("reason") or outcome.get("type")
                if isinstance(outcome, dict)
                else str(outcome)
            )
            holds.append(
                _make_hold(
                    f"{day}-{hold_type}{k}",
                    hold_type,
                    t_apnea,
                    spo2_apnea,
                    hr_apnea,
                    t_rec,
                    spo2_rec,
                    hr_rec,
                    {
                        "session": day,
                        "source": "stamina-import" if imported else "nadir",
                        "duration": duration,
                        "planned": step.get("plannedDuration"),
                        "outcome": outcome,
                        "contractions": len(step.get("contractions", [])),
                        "missing_frac": missing,
                        "max_gap": max_gap,
                        "gap_off_plateau": off_plateau,
                        "no_contact": sum(
                            1 for v in step.get("vitals", []) if not v.get("contactDetected", True)
                        ),
                        "recovery_s": float(len(t_rec)),
                    },
                )
            )
    return holds


# ---------------------------------------------------------------- STAmina CSV


def _hms(s):
    parts = [int(p) for p in s.split(":")]
    return parts[-1] + 60 * parts[-2] + (3600 * parts[-3] if len(parts) == 3 else 0)


def load_stamina(path):
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        started = False
        for row in csv.reader(fh):
            if not row:
                continue
            if row[0].strip() == "Biometrics":
                started = True
                continue
            if not started or row[0].strip() == "Time" or len(row) < 5:
                continue
            try:
                rows.append(
                    (_hms(row[0]), _hms(row[1]), row[2].strip(), float(row[3]), float(row[4]))
                )
            except ValueError:
                continue

    try:
        day = datetime.strptime(Path(path).stem, "%A, %B %d, %Y").date().isoformat()
    except ValueError:
        day = Path(path).stem
    blocks = []
    for r in rows:
        if blocks and blocks[-1][0] == r[2]:
            blocks[-1][1].append(r)
        else:
            blocks.append((r[2], [r]))

    holds = []
    k = 0
    for b, (kind, block) in enumerate(blocks):
        if kind != "Apnea":
            continue
        k += 1
        if block[-1][0] - block[0][0] < MIN_HOLD_S:
            continue
        t0 = block[0][0]
        t_raw = np.array([r[0] - t0 for r in block], dtype=float)
        t_apnea, spo2_apnea, missing, max_gap, off_plateau = _fill_gaps(
            t_raw, np.array([r[4] for r in block])
        )
        hr_apnea = np.interp(t_apnea, t_raw, [r[3] for r in block])
        t_rec = spo2_rec = hr_rec = np.array([], dtype=float)
        if b + 1 < len(blocks) and blocks[b + 1][0] in ("Rest", "Cooldown"):
            nb = blocks[b + 1][1]
            t_rec = np.array([r[1] + 1 for r in nb], dtype=float)
            spo2_rec = np.array([r[4] for r in nb], dtype=float)
            hr_rec = np.array([r[3] for r in nb], dtype=float)
            t_rec, spo2_rec, hr_rec = _trim_recovery(t_rec, spo2_rec, hr_rec)
            t_rec += t_apnea[-1]
        # Lung volume is not in a STAmina export; the February session's labels come from v7.
        hold_type = {1: "FL", 2: "FRC", 3: "RV", 4: "RV", 5: "FRC", 6: "FL"}.get(k, "FL")
        holds.append(
            _make_hold(
                f"{day}-{hold_type}{k}",
                hold_type,
                t_apnea,
                spo2_apnea,
                hr_apnea,
                t_rec,
                spo2_rec,
                hr_rec,
                {
                    "session": day,
                    "source": "stamina",
                    "duration": float(t_apnea[-1] + 1),
                    "planned": None,
                    "outcome": "completed",
                    "contractions": 0,
                    "missing_frac": missing,
                    "max_gap": max_gap,
                    "gap_off_plateau": off_plateau,
                    "no_contact": 0,
                    "recovery_s": float(len(t_rec)),
                },
            )
        )
    return holds


# ---------------------------------------------------------------- QC and inventory


def qc(hold):
    """v7's exclusion rules plus two coverage rules the Nadir data made necessary."""
    apnea = hold["spo2_apnea"]
    if apnea.max() - apnea.min() < 5:
        return "range < 5%"
    t_nadir = hold["t_apnea"][np.argmin(apnea)]
    if t_nadir < hold["t_end"] - 30:
        # v7 dropped RV#4 (nadir 67 s early: breathing resumed) and kept FRC#2 (12 s early,
        # flagged).
        return f"nadir {hold['t_end'] - t_nadir:.0f}s before end"
    if hold["meta"]["missing_frac"] > 0.2:
        return f"{hold['meta']['missing_frac']:.0%} of samples filled"
    if hold["meta"]["gap_off_plateau"]:
        return f"{hold['meta']['max_gap']:.0f}s gap below 95%"
    return None


def load(path):
    return load_stamina(path) if str(path).endswith(".csv") else load_nadir(path)


def inventory(holds):
    print(
        f"{'hold':<20}{'dur':>5} {'nadir':>5} {'range':>5} {'fill':>5} {'gap':>4} "
        f"{'rec':>4} {'contr':>5}  {'source':<15} qc"
    )
    for h in holds:
        m = h["meta"]
        reason = qc(h)
        print(
            f"{h['id']:<20}{m['duration']:>5.0f} {h['spo2_apnea'].min():>5.0f} "
            f"{h['spo2_apnea'].max() - h['spo2_apnea'].min():>5.0f} {m['missing_frac']:>5.0%} "
            f"{m['max_gap']:>4.0f} {m['recovery_s']:>4.0f} {m['contractions']:>5}  "
            f"{m['source']:<15} {reason or 'ok'}"
        )
    kept = [h for h in holds if qc(h) is None]
    by_type = {t: sum(1 for h in kept if h["type"] == t) for t in ("FL", "FRC", "RV")}
    print(
        f"\n{len(kept)}/{len(holds)} holds pass QC: {by_type}, "
        f"{len({h['meta']['session'] for h in kept})} sessions"
    )


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(f"\n== {p}")
        inventory(load(p))
