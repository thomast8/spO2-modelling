"""Contract §11 and §16 spirometry: NHANES 2007-2012 reference fit, calibrated leave-one-cycle-out.

Cohort (pre-bronchodilator SPXN* only): ages 20-79 (SMQ020 is asked from 20; 80 is top-coded),
never-smokers (SMQ020 == 2), answered "no" to asthma, emphysema and chronic bronchitis (MCQ010,
MCQ160G, MCQ160K == 2), FVC and FEV1 both graded A or B (SPXNQFVC, SPXNQFV1), height present.
Unweighted OLS on ln litres; ln σ = c0 + c1·age by maximum likelihood, or constant when that is
better held out. Each cycle is held out in turn; k scales σ so the pooled held-out 5-95 % coverage
is 90 %; the final fit uses all three cycles with ln k folded into c0. The XPT files are read with
pandas.read_sas only.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
from ecsc1993 import route1
from literature import non2023
from scipy.optimize import minimize

Z95 = 1.6448536269514722
CYCLES = {"E": "2007-08", "F": "2009-10", "G": "2011-12"}
MEASURES = {"fvc": "SPXNFVC", "fev1": "SPXNFEV1"}


def _read(cache, stem, suffix, cols):
    df = pd.read_sas(Path(cache) / f"{stem}_{suffix}.xpt", format="xport", encoding="latin-1")
    return df[["SEQN", *cols]]


def cohort(cache, suffixes):
    frames = []
    for sx in suffixes:
        df = _read(cache, "SPX", sx, ["SPXNFVC", "SPXNFEV1", "SPXNQFVC", "SPXNQFV1"])
        for stem, cols in (
            ("DEMO", ["RIAGENDR", "RIDAGEYR"]),
            ("BMX", ["BMXHT"]),
            ("SMQ", ["SMQ020"]),
            ("MCQ", ["MCQ010", "MCQ160G", "MCQ160K"]),
        ):
            df = df.merge(_read(cache, stem, sx, cols), on="SEQN", how="inner")
        df["cycle"] = sx
        frames.append(df)
    df = pd.concat(frames)
    keep = (
        df.RIDAGEYR.between(20, 79)
        & (df.SMQ020 == 2)
        & (df.MCQ010 == 2)
        & (df.MCQ160G == 2)
        & (df.MCQ160K == 2)
        & df.SPXNQFVC.isin(["A", "B"])
        & df.SPXNQFV1.isin(["A", "B"])
        & (df.SPXNFVC > 0)
        & (df.SPXNFEV1 > 0)
        & df.BMXHT.notna()
    )
    df = df[keep].copy()
    df["sex"] = np.where(df.RIAGENDR == 1, "male", "female")
    return df


def design(df):
    a = df.RIDAGEYR.to_numpy(float)
    return np.column_stack([np.ones(len(df)), np.log(df.BMXHT.to_numpy(float)), a, a * a])


def fit_equation(df, col):
    y = np.log(df[col].to_numpy(float) / 1000)
    xmat = design(df)
    coef, *_ = np.linalg.lstsq(xmat, y, rcond=None)
    r = y - xmat @ coef
    age = df.RIDAGEYR.to_numpy(float)

    def nll(c):
        ls = c[0] + c[1] * age
        return float(np.sum(ls + r * r / (2 * np.exp(2 * ls))))

    c_const = (math.log(float(np.std(r))), 0.0)
    c_age = tuple(minimize(nll, np.array(c_const), method="Nelder-Mead").x)
    return coef, c_const, c_age


def standardised(df, col, coef, c):
    """Held-out z and mean log-likelihood under (coef, c)."""
    y = np.log(df[col].to_numpy(float) / 1000)
    sd = np.exp(c[0] + c[1] * df.RIDAGEYR.to_numpy(float))
    z = (y - design(df) @ coef) / sd
    return z, float(np.mean(-np.log(sd) - z * z / 2))


def fit_reference(cache, out=None):
    data = cohort(cache, tuple(CYCLES))
    ref = {"ageRange": (20.0, 79.0)}
    lines = []
    for sex in ("male", "female"):
        d = data[data.sex == sex]
        per = {"heightRange": (float(d.BMXHT.min()), float(d.BMXHT.max()))}
        for name, col in MEASURES.items():
            held = {form: {} for form in ("constant", "age")}
            for cyc in CYCLES:
                coef, c_const, c_age = fit_equation(d[d.cycle != cyc], col)
                te = d[d.cycle == cyc]
                held["constant"][cyc] = standardised(te, col, coef, c_const)
                held["age"][cyc] = standardised(te, col, coef, c_age)

            def pooled_ll(form, held=held):
                parts = held[form].values()
                return sum(ll * len(z) for z, ll in parts) / sum(len(z) for z, _ in parts)

            form = "age" if pooled_ll("age") > pooled_ll("constant") else "constant"
            z_all = np.concatenate([z for z, _ in held[form].values()])
            k = float(np.quantile(np.abs(z_all), 0.9)) / Z95
            coef, c_const, c_age = fit_equation(d, col)
            c = c_age if form == "age" else c_const
            per[name] = {
                "a0": float(coef[0]),
                "a1": float(coef[1]),
                "a2": float(coef[2]),
                "a3": float(coef[3]),
                "c0": float(c[0] + math.log(k)),
                "c1": float(c[1]) if form == "age" else 0.0,
            }
            before = {cyc: float(np.mean(np.abs(z) <= Z95)) for cyc, (z, _) in held[form].items()}
            after = {
                cyc: float(np.mean(np.abs(z) <= Z95 * k)) for cyc, (z, _) in held[form].items()
            }
            lines.append(
                f"    {sex:<7}{name:<5} n {len(d):>5}; spread {form}, k = {k:.3f}; held-out "
                "coverage by cycle "
                + ", ".join(f"{CYCLES[c_]} {before[c_]:.1%} -> {after[c_]:.1%}" for c_ in CYCLES)
                + f"; pooled {np.mean(np.abs(z_all) <= Z95):.1%} -> "
                f"{np.mean(np.abs(z_all) <= Z95 * k):.1%}"
            )
        ref[sex] = per
    if out is not None:
        out.append(
            f"  NHANES cohort: {len(data)} never-smokers aged 20-79 without asthma, emphysema or "
            "chronic bronchitis,\n  FVC and FEV1 graded A/B ("
            + ", ".join(f"{CYCLES[c]} {int((data.cycle == c).sum())}" for c in CYCLES)
            + "). Each cycle held out in turn; k scales σ to 90 % pooled coverage."
        )
        out += lines
        out.append("  Cross-check of the fitted medians (L):")
        out.append(
            f"    {'sex':<7}{'age':>4}{'cm':>5}  {'FVC fit':>8}{'Non23':>7}{'ECSC93':>8}  "
            f"{'FEV1 fit':>9}{'Non23':>7}"
        )
        for sex, heights in (("male", (165, 175, 185)), ("female", (155, 165, 175))):
            for age in (40, 60):
                for h in heights:
                    fvc = predict(ref, sex, age, h, "fvc")[1]
                    fev = predict(ref, sex, age, h, "fev1")[1]
                    fem = sex == "female"
                    out.append(
                        f"    {sex:<7}{age:>4}{h:>5}  "
                        f"{fvc:>8.2f}{non2023('non2023_fvc_median', age, fem, h):>7.2f}"
                        f"{route1(sex, h, age)['FVC']['pred']:>8.2f}  {fev:>9.2f}"
                        f"{non2023('non2023_fev1_median', age, fem, h):>7.2f}"
                    )
        out.append("")
    return ref


def predict(ref, sex, age, height_cm, measure):
    """(5th, median, 95th) in litres and σ, or None outside the fitted ranges."""
    lo_a, hi_a = ref["ageRange"]
    lo_h, hi_h = ref[sex]["heightRange"]
    if not (lo_a <= age <= hi_a and lo_h <= height_cm <= hi_h):
        return None
    e = ref[sex][measure]
    mu = e["a0"] + e["a1"] * math.log(height_cm) + e["a2"] * age + e["a3"] * age * age
    sd = math.exp(e["c0"] + e["c1"] * age)
    return math.exp(mu - Z95 * sd), math.exp(mu), math.exp(mu + Z95 * sd), sd
