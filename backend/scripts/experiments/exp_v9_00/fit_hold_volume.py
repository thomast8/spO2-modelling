"""Contract §6: hold-volume classifier (full vs RV), Laplace logistic, leave-one-session-out.

Abstention band: on the LOSO probabilities, the widest band [lower, upper] (grid step 0.05) whose
non-abstained accuracy is ≥ 90 % with coverage ≥ 50 %; ties go to higher accuracy, then higher
coverage. No such band → no model is emitted (`Priors.holdVolume = nil`).
"""

import math

import numpy as np
from archive import finished, is_imported_from_stamina, style_source
from evidence import DEFAULT_FLOORS, evaluate
from features import FEATURE_NAMES, hold_features, lung_mode_of
from numerics import logistic_predictive

FEB_V7_LABELS = {1: "full", 2: "frc", 3: "rv", 4: "rv", 5: "frc", 6: "full"}  # exp_v8_00 loader
NAME_ONLY = [7, 8, 9]
DEPLOYMENT_DAY = "2026-08-06"
GRID = [round(0.05 * i, 2) for i in range(21)]


def label(session, step):
    """1 full/packed, 0 rv, 'frc' (counted, excluded), None (not a label)."""
    if is_imported_from_stamina(session) and style_source(step) != "confirmed":
        return None
    mode = lung_mode_of(step)
    return {"full": 1, "rv": 0, "frc": "frc"}.get(mode)


def labelled_rows(sessions):
    """[(session, facts, features, label)] for every hold with a label (frc included)."""
    out = []
    for s in finished(sessions):
        for facts, x in hold_features(s):
            y = label(s, s["steps"][facts["stepIndex"]])
            if y is not None:
                out.append((s, facts, x, y))
    return out


def standardise(feats):
    means = feats.mean(axis=0)
    scales = feats.std(axis=0)
    scales[scales == 0] = 1.0
    return means, scales


def design(feats, means, scales):
    return np.hstack([(feats - means) / scales, np.ones((len(feats), 1))])


def laplace_logistic(zmat, y):
    """Prior N(0, 1) on every weight; Newton to |Δ| < 1e-10 (≤ 100 steps); cov = (−H)⁻¹."""
    w = np.zeros(zmat.shape[1])
    for _ in range(100):
        p = 1 / (1 + np.exp(-(zmat @ w)))
        neg_h = zmat.T @ (zmat * (p * (1 - p))[:, None]) + np.eye(len(w))
        step = np.linalg.solve(neg_h, zmat.T @ (y - p) - w)
        w += step
        if np.max(np.abs(step)) < 1e-10:
            break
    p = 1 / (1 + np.exp(-(zmat @ w)))
    neg_h = zmat.T @ (zmat * (p * (1 - p))[:, None]) + np.eye(len(w))
    return w, np.linalg.inv(neg_h)


def fit(rows, columns=None):
    feats = np.array([r[2] for r in rows], float)
    y = np.array([r[3] for r in rows], float)
    if columns is not None:
        feats = feats[:, columns]
    means, scales = standardise(feats)
    w, cov = laplace_logistic(design(feats, means, scales), y)
    return {"means": means, "scales": scales, "mean": w, "cov": cov, "columns": columns}


def predict(model, x):
    x = np.array(x, float)
    if model["columns"] is not None:
        x = x[model["columns"]]
    z = list((x - model["means"]) / model["scales"]) + [1.0]
    return logistic_predictive(list(model["mean"]), model["cov"].tolist(), z)


def loso(rows, kind):
    """{row index: p(full)} with each session held out in turn."""
    out = {}
    ids = sorted({r[0]["id"] for r in rows})
    for sid in ids:
        train = [r for r in rows if r[0]["id"] != sid]
        test = [(i, r) for i, r in enumerate(rows) if r[0]["id"] == sid]
        if kind == "majority":
            p = sum(r[3] for r in train) / len(train)
            out.update({i: p for i, _ in test})
            continue
        model = fit(train, NAME_ONLY if kind == "name" else None)
        out.update({i: predict(model, r[2]) for i, r in test})
    return out


def band_stats(ps, ys, lower, upper):
    decided = [(p >= upper, y) for p, y in zip(ps, ys, strict=True) if p >= upper or p <= lower]
    coverage = len(decided) / len(ps)
    accuracy = sum(1 for full, y in decided if full == (y == 1)) / len(decided) if decided else None
    return coverage, accuracy


def choose_band(ps, ys):
    best = None
    for lo in GRID:
        for hi in GRID:
            if not 0 < lo <= 0.5 <= hi < 1:
                continue
            cov, acc = band_stats(ps, ys, lo, hi)
            if acc is None or acc < 0.9 or cov < 0.5:
                continue
            key = (round(hi - lo, 10), acc, cov)
            if best is None or key > best[0]:
                best = (key, lo, hi, cov, acc)
    return None if best is None else best[1:]


def verdict(p, lower, upper):
    return "full" if p >= upper else "rv" if p <= lower else None


def counts_by_class(sessions):
    counts = {"full": 0, "rv": 0, "frc": 0}
    for s in sessions:
        for facts, _ in hold_features(s):
            y = label(s, s["steps"][facts["stepIndex"]])
            if y is not None:
                counts[{1: "full", 0: "rv", "frc": "frc"}[y]] += 1
    return counts


def classifier_gate(labelled, floors=None):
    """`Analytics.holdVolumeClassifierEvidence()`: labelled counts per class against the floor."""
    units = [("confirmedHolds", m) for m in ("full", "rv", "frc")]
    f = (floors or DEFAULT_FLOORS)["holdVolume"]
    return evaluate("holdVolume", f, {u: labelled[u[1]] for u in units}, None, False)


def report(sessions, out):
    rows_all = labelled_rows(sessions)
    rows = [r for r in rows_all if r[3] != "frc"]
    ys = [r[3] for r in rows]
    labelled = counts_by_class(sessions)
    out.append("2. HOLD-VOLUME CLASSIFIER (contract §6)")
    out.append("-" * 39)
    out.append("")
    out.append(
        f"Labels: {labelled['full']} full (full or packed, recorded by Nadir, plus the confirmed "
        f"6 August hold 9),\n{labelled['rv']} RV (Nadir plus the confirmed 6 August holds 1-8), "
        f"{labelled['frc']} FRC (counted, excluded). {len({r[0]['id'] for r in rows})} sessions. "
        "Imported holds\nwithout a confirmation are not labels."
    )
    out.append("")
    preds = {kind: loso(rows, kind) for kind in ("model", "name", "majority")}
    out.append(
        "  Leave-one-session-out, all 15 features vs the name tokens alone vs the majority class:"
    )
    out.append(f"    {'model':<10}{'acc @0.5':>9}{'log loss':>10}{'brier':>8}")
    for kind, pr in preds.items():
        ps = [pr[i] for i in range(len(rows))]
        acc = sum(1 for p, y in zip(ps, ys, strict=True) if (p >= 0.5) == (y == 1)) / len(ys)
        ll = -sum(
            math.log(max(1e-12, p if y == 1 else 1 - p)) for p, y in zip(ps, ys, strict=True)
        ) / len(ys)
        brier = sum((p - y) ** 2 for p, y in zip(ps, ys, strict=True)) / len(ys)
        out.append(f"    {kind:<10}{acc:>9.2f}{ll:>10.3f}{brier:>8.3f}")
    out.append("")

    # The band is chosen without the 6 August session, so the check below is held out twice:
    # its probabilities come from a model fit without it, and its band was chosen without it.
    tune = [i for i, r in enumerate(rows) if not r[0]["startedAt"].startswith(DEPLOYMENT_DAY)]
    band = choose_band([preds["model"][i] for i in tune], [ys[i] for i in tune])
    out.append(
        "  Abstention band search (widest band, accuracy ≥ 90 %, coverage ≥ 50 %; LOSO "
        "probabilities\n  of every session but 6 August):"
    )
    for kind in ("model", "name", "majority"):
        b = choose_band([preds[kind][i] for i in tune], [ys[i] for i in tune])
        text = (
            "none qualifies"
            if b is None
            else f"[{b[0]:.2f}, {b[1]:.2f}]  coverage {b[2]:.0%}, accuracy {b[3]:.0%}"
        )
        out.append(f"    {kind:<10}{text}")
    out.append("")

    model = fit(rows)
    if band is None:
        out.append(
            "  No band qualifies for the full model: no classifier is emitted (holdVolume = nil)."
        )
        out.append("")
        return {
            "band": None,
            "model": None,
            "labelled": labelled,
            "gate": classifier_gate(labelled),
        }
    lower, upper, cov, acc = band

    out.append(
        f"  Deployment check, 6 August 2026 (held out, confirmed labels), band [{lower:.2f}, "
        f"{upper:.2f}]:"
    )
    aug = [(i, r) for i, r in enumerate(rows) if r[0]["startedAt"].startswith(DEPLOYMENT_DAY)]
    decided = sum(1 for i, _ in aug if verdict(preds["model"][i], lower, upper))
    right = sum(
        1
        for i, r in aug
        if verdict(preds["model"][i], lower, upper) == ("full" if r[3] == 1 else "rv")
    )
    for k, (i, r) in enumerate(aug, 1):
        v = verdict(preds["model"][i], lower, upper)
        truth = "full" if r[3] == 1 else "rv"
        out.append(
            f"    hold {k}  planned {r[1]['planned']:>4} s  p(full) {preds['model'][i]:.2f}  "
            f"verdict {v or 'abstain':<8} confirmed {truth}"
        )
    out.append(f"    decided {decided}/{len(aug)}, all decided correct: {right == decided}")
    feb = next((s for s in sessions if s["startedAt"].startswith("2026-02-21")), None)
    if feb:
        out.append("  21 February 2026 (Masimo CSV; v7 labels, never confirmed; final model):")
        for k, (facts, x) in enumerate(hold_features(feb), 1):
            p = predict(model, x)
            out.append(
                f"    hold {k}  planned {facts['planned']:>4} s  p(full) {p:.2f}  verdict "
                f"{verdict(p, lower, upper) or 'abstain':<8} v7 label {FEB_V7_LABELS.get(k)}"
            )
    imported = [
        (s, f, x)
        for s in sessions
        if is_imported_from_stamina(s)
        for f, x in hold_features(s)
        if style_source(s["steps"][f["stepIndex"]]) != "confirmed"
    ]
    tally = {"full": 0, "rv": 0, None: 0}
    for _, _, x in imported:
        tally[verdict(predict(model, x), lower, upper)] += 1
    out.append(
        f"  Unconfirmed imported holds ({len(imported)}): would be estimated full {tally['full']}, "
        f"RV {tally['rv']}, left as imported {tally[None]}."
    )
    out.append("")
    gate = classifier_gate(labelled)
    out.append(f"  Classes: {gate.summary or 'every class has its floor'}")
    out.append("")
    return {
        "band": (lower, upper),
        "model": model,
        "labelled": labelled,
        "gate": gate,
        "feature_names": FEATURE_NAMES,
    }
