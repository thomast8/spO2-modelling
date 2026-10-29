"""ECSC/ERS 1993 (Quanjer et al., Eur Respir J 1993;6 Suppl 16, Table 6) as the static-volume
ratios of contract §11, and route 1 (TLC, FRC, RV straight from height and age) for comparison.

Every coefficient comes from `literature.ECSC1993`; nothing is transcribed here.
"""

from literature import ECSC1993, ECSC1993_AGE_RANGE_Y, ecsc1993

AGE_FLOOR = 25.0  # the source enters ages 18-25 as 25
REFERENCE_AGE = 45.0  # where the ratio spreads are read off (middle of 25-70)


def ratio_lines():
    """Per sex: ρ = r0 + r1·age for VC/TLC, FRC/TLC, RV/TLC (fractions, not %).

    RV/TLC and FRC/TLC are the source's own equations (no height term); VC/TLC = 1 − RV/TLC
    because TLC = VC + RV.
    """
    out = {}
    for sex, rows in ECSC1993.items():
        _, rv_a, rv_c, _, _ = rows["RV/TLC"]
        _, frc_a, frc_c, _, _ = rows["FRC/TLC"]
        out[sex] = {
            "vcOverTLC": (1 - rv_c / 100, -rv_a / 100),
            "frcOverTLC": (frc_c / 100, frc_a / 100),
            "rvOverTLC": (rv_c / 100, rv_a / 100),
        }
    return out


def ratio_log_sds(sex="male", age=REFERENCE_AGE):
    """Each ratio's own log-scale spread by the delta method on the source's RSDs (derived):
    sd(ln ρ) ≈ RSD/ρ, with VC/TLC = 1 − RV/TLC sharing RV/TLC's RSD. §16 combines them:
    σ_TLC = sqrt(σ_anchor² + tlcLogSD²), σ_x = sqrt(σ_TLC² + xLogSD²), or xLogSD alone when TLC
    is measured."""
    rows = ECSC1993[sex]
    a = max(age, AGE_FLOOR)
    rho_rv = (rows["RV/TLC"][2] + rows["RV/TLC"][1] * a) / 100
    rho_frc = (rows["FRC/TLC"][2] + rows["FRC/TLC"][1] * a) / 100
    sd_rv = rows["RV/TLC"][3] / 100
    sd_frc = rows["FRC/TLC"][3] / 100
    return {
        "tlcLogSD": sd_rv / (1 - rho_rv),
        "frcLogSD": sd_frc / rho_frc,
        "rvLogSD": sd_rv / rho_rv,
    }


def route1(sex, height_cm, age):
    """TLC, FRC, RV (L) and their RSDs straight from the source's equations."""
    return {k: ecsc1993(k, sex, height_cm / 100, age) for k in ("TLC", "FRC", "RV", "FVC")}


AGE_RANGE = ECSC1993_AGE_RANGE_Y
