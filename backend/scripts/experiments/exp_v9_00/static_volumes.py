"""Contract §11 static volumes, route 2 (anchor ÷ ECSC ratios), against route 1 (ECSC directly).

Anchor: measured VC, else measured FVC, else the predicted FVC median. TLC = anchor / ρ_VC,
FRC = ρ_FRC·TLC, RV = ρ_RV·TLC with ρ = r0 + r1·max(age, ageFloor); nil outside ageRange.
A measured TLC/FRC/RV wins for that volume (HARNESS: and a measured TLC is then the TLC the
FRC and RV ratios multiply).
"""

import math

from ecsc1993 import AGE_FLOOR, AGE_RANGE, ratio_lines, ratio_log_sds, route1
from fit_lung_volume import predict


def ratios_literal():
    sds = [ratio_log_sds(sex) for sex in ("male", "female")]
    return {
        **ratio_lines(),
        "tlcLogSD": max(s["tlcLogSD"] for s in sds),
        "frcLogSD": max(s["frcLogSD"] for s in sds),
        "rvLogSD": max(s["rvLogSD"] for s in sds),
        "ageFloor": AGE_FLOOR,
        "ageRange": (float(AGE_RANGE[0]), float(AGE_RANGE[1])),
    }


def static_volumes(profile, spiro, ratios):
    """{TLC, FRC, RV, anchor}: (litres, log sd, source) each, or None outside the age range."""
    age, sex = profile["age"], profile["sex"]
    if not ratios["ageRange"][0] <= age <= ratios["ageRange"][1]:
        return None
    measured = profile.get("measured", {})
    if "vc" in measured:
        anchor = (measured["vc"], 0.0, "measured VC")
    elif "fvc" in measured:
        anchor = (measured["fvc"], 0.0, "measured FVC")
    else:
        p = predict(spiro, sex, age, profile["heightCm"], "fvc")
        if p is None:
            return None
        anchor = (p[1], p[3], "predicted FVC")
    a = max(age, ratios["ageFloor"])
    rho = {k: r0 + r1 * a for k, (r0, r1) in ratios[sex].items()}
    if "tlc" in measured:
        tlc = (measured["tlc"], 0.0, "measured")
    else:
        tlc = (anchor[0] / rho["vcOverTLC"], math.hypot(anchor[1], ratios["tlcLogSD"]), "estimated")
    out = {"anchor": anchor, "TLC": tlc}
    for name, key, sd in (("FRC", "frcOverTLC", "frcLogSD"), ("RV", "rvOverTLC", "rvLogSD")):
        if name.lower() in measured:
            out[name] = (measured[name.lower()], 0.0, "measured")
        else:
            # §16: σ_x = sqrt(σ_TLC² + ratioLogSD_x²); a measured TLC has σ_TLC = 0
            out[name] = (rho[key] * tlc[0], math.hypot(tlc[1], ratios[sd]), "estimated")
    return out


def compare_routes(spiro, ratios, out):
    out.append(
        "  Static volumes, route 2 (predicted FVC ÷ ECSC ratios) against route 1 (ECSC TLC/FRC/RV):"
    )
    out.append(
        f"    {'profile':<22}{'TLC r2':>8}{'r1':>6}{'FRC r2':>8}{'r1':>6}{'RV r2':>7}{'r1':>6}"
    )
    for sex, h in (("male", 180), ("female", 165)):
        for age in (25, 45, 65):
            sv = static_volumes({"sex": sex, "age": age, "heightCm": h}, spiro, ratios)
            r1 = route1(sex, h, age)
            out.append(
                f"    {sex} {h} cm, {age} y{'':<4}{sv['TLC'][0]:>8.2f}{r1['TLC']['pred']:>6.2f}"
                f"{sv['FRC'][0]:>8.2f}{r1['FRC']['pred']:>6.2f}{sv['RV'][0]:>7.2f}{r1['RV']['pred']:>6.2f}"
            )
    out.append(
        f"    each ratio's own log-SD (derived from the ECSC RSDs at 45 y, the larger sex): VC/TLC "
        f"{ratios['tlcLogSD']:.3f}, FRC/TLC {ratios['frcLogSD']:.3f}, RV/TLC "
        f"{ratios['rvLogSD']:.3f}. "
        "Route 2 ships."
    )
    out.append("")
