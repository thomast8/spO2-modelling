"""Contract §12 and §16: the oxygen-store decomposition, its constants and its Monte Carlo spread.

Constants come from `literature.CONSTANTS` (statuses kept); the arterial saturations are the
Severinghaus ODC of this repo (`app.services.hill_model`) at the start and end alveolar tensions.
Stores in mL O₂ (usable = start minus end):
  lung    L_v = 1000·V_v·btpsToStpd·(fAO2Start − fAO2End)
  blood   B   = BV·Hb·10·hufner·[a·(SaS − SaE) + (1 − a)·(SvS − SvE)],  BV = Nadler (L)
  muscle  M   = weight·muscleFraction·myoglobinPerKgMuscle·myoglobinO2PerGram·usableFraction
  VO2 = weight·restingVO2PerKg·diveReflexFactor (mL/min);  T_v = 60·(L_v + B + M)/VO2 seconds.
"""

import math

import numpy as np
from literature import CONSTANTS, NADLER, NADLER_STATUS
from static_volumes import static_volumes

from app.services.hill_model import severinghaus_spo2

DRY_ALVEOLAR_MMHG = 760.0 - 47.0  # the pressures literature.btps_to_stpd is defined at
VENOUS_START = 0.75  # textbook resting mixed-venous saturation: unverified (contract §16)
REFERENCE = {"sex": "male", "age": 35, "heightCm": 180, "weightKg": 75, "measured": {"tlc": 7.5}}
N_MC, SEED = 20000, 7


def value(key):
    return CONSTANTS[key]["value"]


def odc(p_mmhg):
    return float(severinghaus_spo2(np.array([p_mmhg]))[0]) / 100


def point_constants():
    sa_end = odc(value("peto2_end_mmhg"))
    return {
        "btpsToStpd": value("btps_to_stpd_sea_level"),
        "fAO2Start": value("feto2_quiet_breathing"),
        "fAO2End": value("peto2_end_mmhg") / DRY_ALVEOLAR_MMHG,
        "arterialFraction": None,  # set to reproduce the Monte Carlo median blood store
        "arterialSaturationStart": odc(value("feto2_quiet_breathing") * DRY_ALVEOLAR_MMHG),
        "arterialSaturationEnd": sa_end,
        "venousSaturationStart": VENOUS_START,
        "venousSaturationEnd": sa_end,
        "hufner": value("hufner_ml_per_g_hb"),
        "male": {
            "haemoglobinGdL": value("hb_default_male_g_dl"),
            "nadlerK1": NADLER["male"][0],
            "nadlerK2": NADLER["male"][1],
            "nadlerK3": NADLER["male"][2],
            "muscleFraction": value("muscle_mass_fraction_male"),
        },
        "female": {
            "haemoglobinGdL": value("hb_default_female_g_dl"),
            "nadlerK1": NADLER["female"][0],
            "nadlerK2": NADLER["female"][1],
            "nadlerK3": NADLER["female"][2],
            "muscleFraction": value("muscle_mass_fraction_female"),
        },
        "myoglobinPerKgMuscle": value("mb_concentration_mg_per_g_wet"),  # mg/g = g/kg
        "myoglobinO2PerGram": value("mb_o2_binding_ml_per_g"),
        "myoglobinUsableFraction": value("muscle_o2_usable_fraction"),
        "restingVO2PerKg": value("met_ml_kg_min"),
        "diveReflexFactor": value("dive_reflex_factor"),
    }


def stores(c, profile, volumes):
    """{lung: {v: mL}, blood, muscle, vo2, T: {v: s}, total_start: {v: mL}} for one constant set."""
    sex, w, h = profile["sex"], profile["weightKg"], profile["heightCm"] / 100
    s = c[sex]
    bv = s["nadlerK1"] * h**3 + s["nadlerK2"] * w + s["nadlerK3"]
    capacity = bv * s["haemoglobinGdL"] * 10 * c["hufner"]
    a = c["arterialFraction"]
    blood = capacity * (
        a * (c["arterialSaturationStart"] - c["arterialSaturationEnd"])
        + (1 - a) * (c["venousSaturationStart"] - c["venousSaturationEnd"])
    )
    blood_start = capacity * (
        a * c["arterialSaturationStart"] + (1 - a) * c["venousSaturationStart"]
    )
    mb_total = w * s["muscleFraction"] * c["myoglobinPerKgMuscle"] * c["myoglobinO2PerGram"]
    muscle = mb_total * c["myoglobinUsableFraction"]
    vo2 = w * c["restingVO2PerKg"] * c["diveReflexFactor"]
    lung = {
        v: 1000 * vol * c["btpsToStpd"] * (c["fAO2Start"] - c["fAO2End"])
        for v, vol in volumes.items()
    }
    lung_start = {v: 1000 * vol * c["btpsToStpd"] * c["fAO2Start"] for v, vol in volumes.items()}
    return {
        "lung": lung,
        "blood": blood,
        "blood_start": blood_start,
        "muscle": muscle,
        "vo2": vo2,
        "T": {v: 60 * (lung[v] + blood + muscle) / vo2 for v in volumes},
        "usable": {v: lung[v] + blood + muscle for v in volumes},
        "total_start": {v: lung_start[v] + blood_start + mb_total for v in volumes},
    }


def _u(rng, key, n):
    e = CONSTANTS[key]
    return rng.uniform(e["low"], e["high"], n)


def monte_carlo(profile, volumes_sd, n=N_MC, seed=SEED):
    """Samples of stores; volumes_sd = {v: (litres, log sd)}."""
    rng = np.random.default_rng(seed)
    base = point_constants()
    pe = _u(rng, "peto2_end_mmhg", n)
    sa_end = severinghaus_spo2(pe) / 100
    sex = profile["sex"]
    draws = {
        "btps": _u(rng, "btps_to_stpd_sea_level", n),
        "hufner": _u(rng, "hufner_ml_per_g_hb", n),
        "hb": _u(rng, f"hb_default_{sex}_g_dl", n),
        "mf": _u(rng, f"muscle_mass_fraction_{sex}", n),
        "mb": _u(rng, "mb_concentration_mg_per_g_wet", n),
        "mbo2": _u(rng, "mb_o2_binding_ml_per_g", n),
        "mbu": _u(rng, "muscle_o2_usable_fraction", n),
        "vo2": _u(rng, "met_ml_kg_min", n),
        "dive": _u(rng, "dive_reflex_factor", n),
        "a": rng.uniform(0.25, 1.0, n),
    }
    vols = {v: lv * np.exp(sd * rng.standard_normal(n)) for v, (lv, sd) in volumes_sd.items()}
    w, h = profile["weightKg"], profile["heightCm"] / 100
    k1, k2, k3 = NADLER[sex]
    cap = (k1 * h**3 + k2 * w + k3) * draws["hb"] * 10 * draws["hufner"]
    blood = cap * (
        draws["a"] * (base["arterialSaturationStart"] - sa_end)
        + (1 - draws["a"]) * (VENOUS_START - sa_end)
    )
    muscle = w * draws["mf"] * draws["mb"] * draws["mbo2"] * draws["mbu"]
    vo2 = w * draws["vo2"] * draws["dive"]
    f_end = pe / DRY_ALVEOLAR_MMHG
    usable = {
        v: 1000 * vols[v] * draws["btps"] * (base["fAO2Start"] - f_end) + blood + muscle
        for v in vols
    }
    return {"blood": blood, "usable": usable, "T": {v: 60 * usable[v] / vo2 for v in vols}}


def volumes_for(profile, spiro, ratios):
    sv = static_volumes(profile, spiro, ratios)
    return None if sv is None else {"full": sv["TLC"][:2], "frc": sv["FRC"][:2], "rv": sv["RV"][:2]}


def fit_constants(spiro, ratios, out=None):
    """The emitted `OxygenStoreConstants`: literature point values, arterialFraction solved so the
    point blood store equals the Monte Carlo median, sigmaMC/sigmaMCPersonalised from the
    reference diver's full-lung T, sigmaFloor = sqrt(1 − r²)·σ_pop."""
    c = point_constants()
    ref = REFERENCE
    vols = volumes_for(ref, spiro, ratios)
    mc = monte_carlo(ref, vols)
    probe = dict(c, arterialFraction=1.0)
    cap_d_a = stores(probe, ref, {"full": vols["full"][0]})["blood"]
    probe["arterialFraction"] = 0.0
    cap_d_v = stores(probe, ref, {"full": vols["full"][0]})["blood"]
    med_blood = float(np.median(mc["blood"]))
    c["arterialFraction"] = (med_blood - cap_d_v) / (cap_d_a - cap_d_v)
    # §16: consumption is emitted as the value that puts the point T on the Monte Carlo median
    # (derived), so ln T_phys is centred on the distribution σ_phys describes.
    c["literatureRestingVO2PerKg"] = c["restingVO2PerKg"]
    usable = stores(c, ref, {"full": vols["full"][0]})["usable"]["full"]
    med_t = float(np.median(mc["T"]["full"]))
    c["restingVO2PerKg"] = 60 * usable / (med_t * ref["weightKg"] * c["diveReflexFactor"])
    c["sigmaMC"] = float(np.std(np.log(mc["T"]["full"])))
    # Personalising the end saturation by the diver's own lows is removed by decision (nadir#703,
    # #706: a low is never scored). The field stays for the parked Swift type and is unused.
    c["sigmaMCPersonalised"] = c["sigmaMC"]
    c["sigmaFloor"] = math.sqrt(1 - value("r2_lung_volume_single_predictor")) * value(
        "sigma_pop_ln_static"
    )
    if out is not None:
        report(c, mc, spiro, ratios, out)
    return c


def physiology_prior(constants, profile, spiro, ratios, vo2_ml_min=None):
    """{v: (ln T_phys,v, σ_phys)} for the hold model, or None when the profile cannot give it.

    Population constants; `vo2_ml_min` replaces the population consumption with the diver's
    fitted one (§18). Nothing about how low the diver's own SpO₂ went enters it.
    """
    vols = volumes_for(profile, spiro, ratios)
    if vols is None or profile.get("weightKg") is None:
        return None
    sigma = max(constants["sigmaMC"], constants["sigmaFloor"])
    st = stores(constants, profile, {v: lv for v, (lv, _) in vols.items()})
    t = st["T"] if vo2_ml_min is None else {v: 60 * u / vo2_ml_min for v, u in st["usable"].items()}
    return {v: (math.log(t[v]), sigma) for v in t}


def headroom(best_seconds, prior_v):
    from numerics import phi

    return phi((math.log(best_seconds) - prior_v[0]) / prior_v[1])


STATUS_KEYS = {
    "btpsToStpd": "btps_to_stpd_sea_level",
    "fAO2Start": "feto2_quiet_breathing",
    "PETO2 end (mmHg)": "peto2_end_mmhg",
    "hufner": "hufner_ml_per_g_hb",
    "haemoglobin male": "hb_default_male_g_dl",
    "haemoglobin female": "hb_default_female_g_dl",
    "muscleFraction male": "muscle_mass_fraction_male",
    "muscleFraction female": "muscle_mass_fraction_female",
    "myoglobinPerKgMuscle": "mb_concentration_mg_per_g_wet",
    "myoglobinO2PerGram": "mb_o2_binding_ml_per_g",
    "myoglobinUsableFraction": "muscle_o2_usable_fraction",
    "restingVO2PerKg (MET)": "met_ml_kg_min",
    "diveReflexFactor": "dive_reflex_factor",
    "sigmaFloor: σ_pop": "sigma_pop_ln_static",
    "sigmaFloor: r²": "r2_lung_volume_single_predictor",
}


def report(c, mc, spiro, ratios, out):
    ref = REFERENCE
    vols = volumes_for(ref, spiro, ratios)
    point = stores(c, ref, {v: lv for v, (lv, _) in vols.items()})
    out.append("  Constants (value [Monte Carlo range], status):")
    for name, key in STATUS_KEYS.items():
        e = CONSTANTS[key]
        rng = "" if e["low"] is None else f" [{e['low']}, {e['high']}]"
        out.append(f"    {name:<28}{str(e['value']) + rng:<20} {e['status']}")
    out.append(
        f"    {'Nadler k1, k2, k3':<28}{NADLER['male']} / {NADLER['female']}  {NADLER_STATUS}"
    )
    out.append(f"    {'fAO2End':<28}{c['fAO2End']:.4f}  derived (PETO2 end / 713)")
    out.append(
        f"    {'restingVO2PerKg emitted':<28}{c['restingVO2PerKg']:.3f}  derived (puts the "
        "reference "
        f"diver's point T on the Monte Carlo median; MET {c['literatureRestingVO2PerKg']})"
    )
    out.append(
        f"    {'arterial saturation':<28}{c['arterialSaturationStart']:.3f} -> "
        f"{c['arterialSaturationEnd']:.3f}"
        "  derived (Severinghaus ODC at 0.1444·713 and 26.9 mmHg)"
    )
    out.append(
        f"    {'venous saturation':<28}{VENOUS_START} -> {c['venousSaturationEnd']:.3f}  start "
        "unverified "
        "(textbook), end = arterial end"
    )
    out.append(
        f"    {'arterialFraction':<28}{c['arterialFraction']:.3f} [0.25, 1.0]  unverified "
        "(uncited split; "
        "the value reproduces the median blood store)"
    )
    out.append(
        f"    sigmaMC {c['sigmaMC']:.3f}, floor {c['sigmaFloor']:.3f} (= sqrt(1 − 0.29)·0.30, "
        "derived). Personalising\n    the physiology by the diver's own SpO₂ lows is removed by "
        "decision (nadir#703, #706); sigmaMCPersonalised\n    is emitted equal to sigmaMC only "
        "to keep the parked Swift type, and personalisePhysiology is false."
    )
    u = mc["usable"]["full"]
    lo, mid, hi = (float(np.percentile(u, q)) / 1000 for q in (10, 50, 90))
    out.append(
        f"  Golden row (#701): male 1.80 m, 75 kg, TLC 7.5 L measured: usable O₂ "
        f"{point['usable']['full'] / 1000:.2f} L "
        f"at the point constants; Monte Carlo 80 % {lo:.2f}-{hi:.2f} L (median {mid:.2f}). #701 "
        "says about 1.2 L."
    )
    share = point["usable"]["full"] / point["total_start"]["full"]
    out.append(
        f"  Cross-check: usable share of the starting store {share:.2f} against Schagatay 2009's "
        f"{value('usable_fraction_nondiver')} (non-diver) to {value('usable_fraction_trained')} "
        "(trained); "
        "the starting store here leaves out\n  tissue-dissolved O₂ and the spleen."
    )
    rahn = stores(c, {"sex": "male", "heightCm": 175, "weightKg": 70}, {"full": 5.5})
    out.append(
        f"  Cross-check: blood store of a 70 kg, 1.75 m man (height assumed): "
        f"{rahn['blood_start']:.0f} mL at the "
        f"start, {rahn['blood']:.0f} mL usable, against Rahn's {value('rahn_blood_o2_ml')} mL "
        "(70 kg, total)."
    )
    out.append(
        f"  Reference diver T_phys: full {point['T']['full']:.0f} s, FRC "
        f"{point['T']['frc']:.0f} s, "
        f"RV {point['T']['rv']:.0f} s (FRC and RV from the route-2 ratios at 35 y); ratios "
        f"FRC/full {point['T']['frc'] / point['T']['full']:.2f}, RV/full "
        f"{point['T']['rv'] / point['T']['full']:.2f}.\n  These ratios are a population statement "
        "only. They never replace DryBestEstimate's 1/2 and\n  1/3 or feed any planning (§16): "
        "they would raise FRC and RV table targets above what a diver has\n  demonstrated. The "
        "app shows the usable stores in litres, nothing in seconds."
    )
    elite = {"sex": "male", "age": 30, "heightCm": 184, "weightKg": 82, "measured": {"vc": 7.3}}
    pv = physiology_prior(c, elite, spiro, ratios)
    lo, hi = (math.exp(pv["full"][0] + s * 1.2815515655446004 * pv["full"][1]) for s in (-1, 1))
    out.append(
        f"  Group check, Schagatay 2012 elite men (184 cm, 82 kg, VC 7.3 L, age assumed 30): "
        "full-lung "
        f"T_phys 80 % {lo:.0f}-{hi:.0f} s against their competition mean 353 s (SE 39); the "
        "decomposition "
        "is known to\n  under-predict elites (packing, spleen, trained end saturation)."
    )
    out.append("")
