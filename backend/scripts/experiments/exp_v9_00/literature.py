"""literature.py - cited constants for Nadir retrospective analytics.

Every entry: value, low, high, units, citation, url, quote, status (+ note).
status in {"verified-in-source", "verified-in-secondary", "derived", "unverified"}.
Compiled 2026-10-22 from primary/secondary reads plus an adversarial re-check.
"""

import math


def _c(value, low, high, units, citation, url, quote, status, note=""):
    return dict(value=value, low=low, high=high, units=units, citation=citation,
                url=url, quote=quote, status=status, note=note)


# --- recurring sources -------------------------------------------------------
SCHAGATAY_2009 = ("Schagatay E. Predicting performance in competitive apnoea diving. "
                  "Part I: static apnoea. Diving Hyperb Med 2009;39(2):88-99")
SCHAGATAY_2009_URL = ("https://www.dhmjournal.com/images/IndividArticles/39June/"
                      "Schagatay_dhm.39.2.88-99.pdf")
RAHN_VIA_SCHAGATAY = ("Rahn H. Oxygen stores of man. In: Dickens F, Neil E, eds. Oxygen in the "
                      "animal organism. Pergamon/Macmillan; 1964:609-18; as reported in "
                      + SCHAGATAY_2009)
PATRICIAN_2021 = ("Patrician A, et al. Breath-hold diving - the physiology of diving deep and "
                  "returning. Front Physiol 2021;12:639377")
PATRICIAN_2021_URL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC8176094/"
LINDHOLM_2006 = ("Lindholm P, Lundgren CE. Undersea Hyperb Med 2006;33(6):463-467. "
                 "PMID 17274316 (title not recorded in our reads)")
LINDHOLM_2006_URL = ("https://www.ebi.ac.uk/europepmc/webservices/rest/search?"
                     "query=EXT_ID:17274316&resultType=core&format=json")
FERRETTI_2001 = ("Ferretti G. Extreme human breath-hold diving. Eur J Appl Physiol "
                 "2001;84(4):254-271. doi:10.1007/s004210000377. PMID 11374109")
FERRETTI_2001_URL = "https://pubmed.ncbi.nlm.nih.gov/11374109/"
ANDERSSON_2025 = ("Andersson JPA, Bacanovic T, Chen P, Lodin-Sundstrom A, Halder A, Persson G, "
                  "Liner MH, Sjogreen B. Time courses for pulmonary oxygen uptake and "
                  "cardiovascular responses are similar during apnea in resting humans. "
                  "Front Physiol 2025;16:1524237. doi:10.3389/fphys.2025.1524237")
ANDERSSON_2025_URL = ("https://www.frontiersin.org/journals/physiology/articles/"
                      "10.3389/fphys.2025.1524237/full")
JANSSEN_2000 = ("Janssen I, Heymsfield SB, Wang ZM, Ross R. Skeletal muscle mass and "
                "distribution in 468 men and women aged 18-88 yr. J Appl Physiol "
                "2000;89(1):81-88. doi:10.1152/jappl.2000.89.1.81")
JANSSEN_2000_URL = "https://pubmed.ncbi.nlm.nih.gov/10904038/"
WHO_2024 = ("WHO. Guideline on haemoglobin cutoffs to define anaemia in individuals and "
            "populations. Geneva: WHO; 2024. ISBN 978-92-4-008854-2, Table 2")
WHO_2024_URL = ("https://iris.who.int/server/api/core/bitstreams/"
                "f9f74397-1440-478d-a63c-26f29a01552f/content")
SCHAGATAY_2012 = ("Schagatay E, Richardson MX, Lodin-Sundstrom A. Size matters: spleen and lung "
                  "volumes predict performance in human apneic divers. Front Physiol "
                  "2012;3:173. doi:10.3389/fphys.2012.00173")
SCHAGATAY_2012_URL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC3376424"
FERNANDEZ_2017 = ("De Asis Fernandez F, Gonzalez-Rave JM, Juarez D. Breath-hold diving "
                  "performance factors. J Hum Sport Exerc 2017;12(3):582-592. "
                  "doi:10.14198/jhse.2017.123.03")
FERNANDEZ_2017_URL = "https://www.redalyc.org/pdf/3010/301053359003.pdf"
BAIN_2017 = ("Bain AR, Barak OF, Hoiland RL, Drvis I, Bailey DM, Dujic Z, Mijacika T, "
             "Santoro A, DeMasi DK, MacLeod DB, Ainslie PN. Forced vital capacity and not "
             "central chemoreflex predicts maximal hyperoxic breath-hold duration in elite "
             "apneists. Respir Physiol Neurobiol 2017;242:8-11. doi:10.1016/j.resp.2017.02.015")
BAIN_2017_URL = ("https://pure.southwales.ac.uk/en/publications/"
                 "forced-vital-capacity-and-not-central-chemoreflex-predicts-maxima/")
NON_2023 = ("Non AL, Bailey B, Bhatt SP, Casaburi R, Regan EA, Wang A, Limon A, Rabay C, "
            "Diaz AA, Baldomero AK, Kinney G, Young KA, Felts B, Hand C, Conrad DJ. "
            "Race-Specific Spirometry Equations Do Not Improve Models of Dyspnea and "
            "Quantitative Chest CT Phenotypes. Chest 2023;164(6):1492-1504. "
            "doi:10.1016/j.chest.2023.07.019. PMID 37507005")
NON_2023_URL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC10925545/"
NON_2023_SUPP_URL = ("https://www.ebi.ac.uk/europepmc/webservices/rest/"
                     "PMC10925545/supplementaryFiles")


CONSTANTS = {
    # ---------------- O2 carriage -------------------------------------------
    "hufner_ml_per_g_hb": _c(
        1.34, 1.31, 1.39, "mL O2 / g Hb",
        "Thomas C, Lumb AB. Physiology of haemoglobin. Contin Educ Anaesth Crit Care Pain "
        "2012;12(5):251-256. doi:10.1093/bjaceaccp/mks025 (range); derangedphysiology.com "
        "'Oxygen carrying capacity of whole blood' (1.34 convention)",
        "https://www.e-safe-anaesthesia.org/e_library/02/Physiology_of_haemoglobin_CEACCP_2012.pdf",
        "a theoretical value for BO2 of 1.39 ml g-1 is easily obtained. However, in vivo "
        "experiments produce values anywhere between 1.31 and 1.37",
        "verified-in-secondary",
        "1.34 is the textbook convention. 1.36 (used by Patrician 2021) has no sourced "
        "rationale; do not label it IUPAC."),
    "mb_o2_binding_ml_per_g": _c(
        1.34, 1.26, 1.34, "mL O2 / g Mb",
        "Arregui M, et al. Myoglobin concentration and oxygen stores in different functional "
        "muscle groups from three small cetacean species. Animals (Basel) 2021;11(2):451. "
        "doi:10.3390/ani11020451",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC7915992/",
        "O2 binding capacity of Mb (O2 Mb: 1.34 mL O2 g-1 Mb, assuming complete Mb saturation",
        "verified-in-secondary",
        "Marine-mammal convention. low = stoichiometric estimate 22.4 L/mol / 17.0-17.8 kDa "
        "(derived)."),
    "btps_to_stpd_sea_level": _c(
        0.826, 0.803, 0.826, "dimensionless",
        "Definitions from 'Gas volume corrections in intensive care unit: needed or "
        "pointless?' J Appl Physiol 2023;135:334-342. doi:10.1152/japplphysiol.00225.2023",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC10390049",
        "STPD = standard temperature (273 Kelvin), atmospheric pressure (760 mmHg), dry",
        "derived",
        "(713/760)*(273/310) at PB 760, PH2O 47, 37 C. low = PB 740. Use btps_to_stpd()."),

    # ---------------- haemoglobin --------------------------------------------
    "who_anaemia_cutoff_male_g_l": _c(
        130, None, None, "g/L", WHO_2024, WHO_2024_URL,
        "Adults, 15-65 years, men <130", "verified-in-source",
        "Anaemia cutoff (5th percentile), not a default."),
    "who_anaemia_cutoff_female_g_l": _c(
        120, None, None, "g/L", WHO_2024, WHO_2024_URL,
        "Adults, 15-65 years, nonpregnant women <120", "verified-in-source",
        "Pregnancy: <110 / <105 / <110 g/L by trimester."),
    "hb_default_male_g_dl": _c(
        15.0, 13.0, 17.0, "g/dL", "Modelling choice; low = WHO 2024 cutoff", WHO_2024_URL,
        "", "unverified",
        "No single sourced mean. Botswana HIV-negative men 15.1 g/dL (Afr J Lab Med 2012, "
        "doi:10.4102/ajlm.v1i1.5). high is judgement."),
    "hb_default_female_g_dl": _c(
        13.5, 12.0, 15.5, "g/dL", "Modelling choice; low = WHO 2024 cutoff", WHO_2024_URL,
        "", "unverified",
        "Botswana women 12.4 g/dL is a weak anchor (paper says its range is below White "
        "populations). high is judgement."),
    "hb_elite_breathhold_divers_g_l": _c(
        153, 147, 159, "g/L",
        "Elia 2019b as tabulated in Elia A, et al. Eur J Appl Physiol 2021. "
        "doi:10.1007/s00421-021-04664-x, Table 4",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC8144079/",
        "EBHD: 153 +/- 6 g/L* ... ND: 148 +/- 3 g/L", "verified-in-secondary",
        "Same table: Elia 2021b found no difference (148 vs 149 g/L). Bounds = +/-1 SD."),

    # ---------------- Rahn / Schagatay O2 store decomposition -----------------
    "rahn_total_o2_store_ml": _c(
        1996, None, None, "mL O2", RAHN_VIA_SCHAGATAY, SCHAGATAY_2009_URL,
        "calculated by Rahn to be 1,996 ml, with 820 ml of the storage in the lungs, "
        "880 ml in the blood, 240 ml in the myoglobin, and 56 ml in physical solution",
        "verified-in-secondary",
        "70 kg, TLC 5.5 L. Quote verified in Schagatay p.95-96; Rahn chapter not opened. "
        "STPD vs BTPS not stated."),
    "rahn_lung_o2_ml": _c(820, None, None, "mL O2", RAHN_VIA_SCHAGATAY, SCHAGATAY_2009_URL,
                          "820 ml of the storage in the lungs", "verified-in-secondary"),
    "rahn_blood_o2_ml": _c(880, None, None, "mL O2", RAHN_VIA_SCHAGATAY, SCHAGATAY_2009_URL,
                           "880 ml in the blood", "verified-in-secondary",
                           "No Hb, arterial/venous split or saturations given."),
    "rahn_myoglobin_o2_ml": _c(
        240, 170, 240, "mL O2", RAHN_VIA_SCHAGATAY, SCHAGATAY_2009_URL,
        "240 ml in the myoglobin", "verified-in-secondary",
        "low = 70 kg x 0.384 x 4.7 mg/g x 1.34 mL/g = 169 mL (derived). Schagatay keeps 240 "
        "for the trained diver too."),
    "rahn_tissue_dissolved_o2_ml": _c(
        56, None, None, "mL O2", RAHN_VIA_SCHAGATAY, SCHAGATAY_2009_URL,
        "56 ml in physical solution in tissues", "verified-in-secondary",
        "Fig 7 rounds to 60."),
    "rahn_blood_o2_ml_per_kg": _c(
        12.6, None, None, "mL O2 / kg", RAHN_VIA_SCHAGATAY, SCHAGATAY_2009_URL, "", "derived",
        "880/70. Linear mass scaling is an assumption not made by Rahn or Schagatay."),
    "rahn_mb_plus_tissue_o2_ml_per_kg": _c(
        4.2, None, None, "mL O2 / kg", RAHN_VIA_SCHAGATAY, SCHAGATAY_2009_URL, "", "derived",
        "(240+56)/70. Same scaling caveat."),
    "lung_o2_fraction_of_tlc": _c(
        0.15, 0.14, 0.16, "L O2 / L TLC", RAHN_VIA_SCHAGATAY, SCHAGATAY_2009_URL, "", "derived",
        "820/5500 = 0.149 and 1650/11000 = 0.150. Store-to-TLC ratio only, not an FAO2. "
        "Gas condition unknown; at BTPS it overstates STPD O2 by about 10%. Schagatay "
        "states lung-packing effects on lung O2 concentration were not modelled."),
    "spleen_o2_nondiver_ml": _c(
        100, None, None, "mL O2", SCHAGATAY_2009, SCHAGATAY_2009_URL,
        "a normal-sized spleen emptying by two thirds", "verified-in-source",
        "Non-diver total becomes 2,100 mL."),
    "spleen_o2_trained_ml": _c(
        150, None, None, "mL O2", SCHAGATAY_2009, SCHAGATAY_2009_URL,
        "spleen volume 600 ml giving 150 ml (Fig 7, trained diver)", "verified-in-source"),
    "schagatay_trained_total_o2_ml": _c(
        3200, None, None, "mL O2", SCHAGATAY_2009, SCHAGATAY_2009_URL,
        "lungs 1,650 + blood 1,100 + spleen 150 + myoglobin 240 + tissue 60 (p.96, Fig 7)",
        "verified-in-source",
        "Hypothetical elite, TLC after packing 11 L. Components sum to 3,200 once myoglobin "
        "240 is counted separately."),
    "usable_fraction_nondiver": _c(
        0.5, None, None, "fraction of total store", SCHAGATAY_2009, SCHAGATAY_2009_URL,
        "only about half of this amount, however, can be consumed before the normal person "
        "loses consciousness", "verified-in-source", "Paired with SaO2 floor 50%."),
    "usable_fraction_trained": _c(
        0.7, None, None, "fraction of total store", SCHAGATAY_2009, SCHAGATAY_2009_URL,
        "Fig 7 trained diver: 70% of store used, lowest SaO2 30%", "verified-in-source"),
    "muscle_o2_usable_fraction": _c(
        0.5, 0.0, 1.0, "fraction", SCHAGATAY_2009, SCHAGATAY_2009_URL,
        "It may seem less likely that myoglobin stores could be recruited for the central "
        "circulation during voluntary, conscious apnoea", "unverified",
        "No source gives a number. 0.5 is an assumption; sample the full 0-1 range."),

    # ---------------- muscle ----------------------------------------------------
    "muscle_mass_fraction_male": _c(
        0.384, 0.33, 0.44, "fraction of body mass", JANSSEN_2000, JANSSEN_2000_URL,
        "relative to body mass (38.4 vs. 30.6%)", "verified-in-source",
        "MRI cohort mean, ages 18-88. Bounds are judgement. 2014 erratum "
        "(J Appl Physiol 116(10):1342) unchecked."),
    "muscle_mass_fraction_female": _c(
        0.306, 0.26, 0.35, "fraction of body mass", JANSSEN_2000, JANSSEN_2000_URL,
        "relative to body mass (38.4 vs. 30.6%)", "verified-in-source",
        "Same caveats as male."),
    "mb_concentration_mg_per_g_wet": _c(
        4.7, 3.5, 6.0, "mg Mb / g wet muscle",
        "Moller P, Sylven C. Myoglobin in human skeletal muscle. Scand J Clin Lab Invest "
        "1981;41(5):479-82. doi:10.3109/00365518109090486. PMID 7313529",
        "https://pubmed.ncbi.nlm.nih.gov/7313529/",
        "The mean myoglobin content in leg skeletal muscle of 13 females and 18 males was "
        "4.7 mg/g wet weight.", "verified-in-source",
        "Bounds are judgement. Do not mix with fibre-level mM values (Jacobs 2023)."),

    # ---------------- metabolic rate ----------------------------------------------
    "met_ml_kg_min": _c(
        3.5, 3.5, 4.1, "mL O2 / kg / min",
        "Jette M, Sidney K, Blumchen G. Metabolic equivalents (METS) in exercise testing, "
        "exercise prescription, and evaluation of functional capacity. Clin Cardiol "
        "1990;13(8):555-565. doi:10.1002/clc.4960130809",
        "https://pubmed.ncbi.nlm.nih.gov/2204507/",
        "is equal to 3.5 ml O2 per kg body weight x min", "verified-in-source",
        "high = Andersson 2025 measured prone baseline 4.07."),
    "vo2_rest_prone_ml_kg_min": _c(
        4.07, 3.38, 4.76, "mL O2 / kg / min", ANDERSSON_2025, ANDERSSON_2025_URL,
        "The baseline, eupneic pulmonary O2 uptake before apnea was 4.07 (0.69)",
        "verified-in-source", "Prone, n=18 (group I). Bounds +/-1 SD."),
    "apnoea_uptake_ratio_2min_air": _c(
        0.88, None, None, "x baseline", ANDERSSON_2025, ANDERSSON_2025_URL,
        "reduced by 12% and 16%, respectively, compared to baseline", "verified-in-source",
        "3.57 (0.48) mL/min/kg. Apnoea at 85% of prone VC, not maximal."),
    "apnoea_uptake_ratio_2min_face_immersion": _c(
        0.84, None, None, "x baseline", ANDERSSON_2025, ANDERSSON_2025_URL,
        "The O2 uptake was even further reduced during the entire 2-min apnea with face "
        "immersion, to 3.42 (0.48)", "verified-in-source", "Face in 9-11 C water."),
    "apnoea_uptake_ratio_30_120s_air": _c(
        0.71, None, None, "x baseline", ANDERSSON_2025, ANDERSSON_2025_URL,
        "the O2 uptake was 71 (14) % and 68 (12) % of baseline O2 uptake",
        "verified-in-source"),
    "apnoea_uptake_ratio_30_120s_face_immersion": _c(
        0.68, None, None, "x baseline", ANDERSSON_2025, ANDERSSON_2025_URL,
        "the O2 uptake was 71 (14) % and 68 (12) % of baseline O2 uptake",
        "verified-in-source"),
    "dive_reflex_factor": _c(
        0.85, 0.68, 1.0, "x resting VO2",
        "Built from " + ANDERSSON_2025 + "; Andersson 2004 J Appl Physiol 96(3):1005-10",
        ANDERSSON_2025_URL, "", "derived",
        "Judgement on verified pulmonary-uptake ratios (0.88/0.84 over 2 min, 0.71/0.68 over "
        "30-120 s; 0.78/0.75 in exercise, Andersson 2004). Pulmonary uptake, not whole-body "
        "VO2, so it understates metabolic rate beyond 2-3 min."),
    "vo2_schagatay_nondiver_ml_min": _c(
        300, None, None, "mL O2 / min", SCHAGATAY_2009, SCHAGATAY_2009_URL,
        "Fig 7 non-diver: 300 ml/min, 3 min 30 s", "verified-in-source",
        "Standardized model input, not a measurement."),
    "vo2_schagatay_trained_ml_min": _c(
        200, None, None, "mL O2 / min", SCHAGATAY_2009, SCHAGATAY_2009_URL,
        "Fig 7 trained diver: 200 ml/min, 11 min 12 s", "verified-in-source",
        "Hypothetical relaxed elite. Loose lower bound only (0.67x of 300)."),

    # ---------------- gas tensions and saturation ---------------------------------
    "feto2_quiet_breathing": _c(
        0.1444, None, None, "fraction O2", PATRICIAN_2021, PATRICIAN_2021_URL,
        "FETO2 (0.1444 during pre/post dive breathing and 0.16 during the dive due to lung "
        "packing)", "verified-in-secondary", "Model input in a review, Fig 1 legend."),
    "feto2_lung_packing": _c(
        0.16, None, None, "fraction O2", PATRICIAN_2021, PATRICIAN_2021_URL,
        "0.16 during the dive due to lung packing", "verified-in-secondary"),
    "peto2_start_hyperventilated_mmhg": _c(
        131.7, 126.3, 137.1, "mmHg", LINDHOLM_2006, LINDHOLM_2006_URL,
        "End-tidal PO2 was 131.7 +/- 2.7 mmHg before apnea and 26.9 +/- 7.5 mmHg at apnea "
        "termination.", "verified-in-source",
        "Competitive divers, n=9, after usual pre-apnoea hyperventilation (FETO2 about 0.18). "
        "Bounds +/-2x the reported spread; SD vs SE not stated."),
    "peto2_end_mmhg": _c(
        26.9, 19.4, 34.4, "mmHg", LINDHOLM_2006, LINDHOLM_2006_URL,
        "26.9 +/- 7.5 mmHg at apnea termination", "verified-in-source", "Bounds +/-1 spread."),
    "peto2_motor_control_loss_mmhg": _c(
        20.3, 19.6, 21.0, "mmHg", "Lindholm & Lundgren 2006 as reported in " + PATRICIAN_2021,
        PATRICIAN_2021_URL, "PETO2 20.3 mmHg (range 19.6-21.0) with loss of motor control",
        "verified-in-secondary",
        "Symptomatic subset only; holds ending near 23 mmHg did not cause it."),
    "pao2_end_max_dry_upper_mmhg": _c(
        30, None, None, "mmHg (upper bound)", FERRETTI_2001, FERRETTI_2001_URL,
        "particularly low alveolar oxygen (< 30 mmHg)", "verified-in-source",
        "Abstract; PACO2 > 50 mmHg at the same point."),
    "pao2_end_elite_radial_mmhg": _c(
        29.6, 23.0, 36.2, "mmHg", "Willie 2015 as reported in " + PATRICIAN_2021,
        PATRICIAN_2021_URL, "29.6 +/- 6.6 mmHg", "verified-in-secondary",
        "Radial arterial. Lowest reported PaO2 23 mmHg after 435 s (Bailey 2017)."),
    "sao2_floor_untrained_pct": _c(
        50, None, None, "% SaO2", SCHAGATAY_2009, SCHAGATAY_2009_URL,
        "While an SaO2 of 50 is considered to threaten consciousness in untrained individuals",
        "verified-in-source"),
    "sao2_floor_trained_pct": _c(
        30, 25, 40, "% SaO2", SCHAGATAY_2009, SCHAGATAY_2009_URL,
        "SaO2 levels of approximately 30 per cent have been recorded in apparently unaffected "
        "divers after maximal apnoeas", "verified-in-source",
        "Underlying data are 'unpublished observations'. Bounds are judgement. 50-60% as an "
        "elite end-state is not supported."),

    # ---------------- population (static apnoea) ----------------------------------
    "schagatay2012_static_mean_s": _c(
        353, None, None, "s (mean, SE 39, n=14)", SCHAGATAY_2012, SCHAGATAY_2012_URL,
        "5 min 53 (39) s for static apnea", "verified-in-source",
        "Mean (SE). World-championship competition performances, not lab maxima. SD 146 s "
        "is derived (SE x sqrt 14)."),
    "schagatay2012_vc_l": _c(
        7.3, 5.5, 8.9, "L (mean, SE 0.25)", SCHAGATAY_2012, SCHAGATAY_2012_URL,
        "Mean (SE) VC was 7.3 (0.25) L with a range from 5.5 to 8.9 L.", "verified-in-source",
        "14 elite men, height 184 cm (177-196), weight 82 kg (72-118)."),
    "schagatay2012_r_vc_total_score": _c(
        0.54, None, None, "Pearson r", SCHAGATAY_2012, SCHAGATAY_2012_URL,
        "There was a positive correlation between VC and total competition score "
        "(r = 0.54; P < 0.05).", "verified-in-source",
        "Against total score, NOT static time. No static-only correlation is reported."),
    "fernandez2017_static_s": _c(
        197, 138, 256, "s (mean +/- 59)", FERNANDEZ_2017, FERNANDEZ_2017_URL,
        "STA performance was established in 197+/-59 seconds", "verified-in-source",
        "56 men, pool test after warm-up, deep but not maximal inspiration. '+/-' almost "
        "certainly SD though the paper labels dispersion SE."),
    "fernandez2017_vc_l": _c(
        5.59, 3.7, 9.1, "L (mean, dispersion 1.1)", FERNANDEZ_2017, FERNANDEZ_2017_URL,
        "VC 5.59 (1.1) L, range 3.7-9.1", "verified-in-source", "Height 176 (7) cm, 163-197."),
    "fernandez2017_r_vc_points": _c(
        0.54, 0.539, 0.601, "Pearson r", FERNANDEZ_2017, FERNANDEZ_2017_URL,
        "There was a positive correlation between VC and performance score (r =0.54; "
        "P <0.05; Figure 1; Table 1).", "verified-in-source",
        "Against AIDA points. Paper reports 0.539 / 0.54 / 0.601 in different places."),
    "bain2017_r2_fvc_hyperoxic": _c(
        0.45, 0.01, 0.82, "r^2", BAIN_2017, BAIN_2017_URL,
        "related to the FVC (r2 = 0.45, p < 0.05), but not the HCVR (r2 < 0.01, p > 0.05)",
        "verified-in-source",
        "Abstract. n=11, hyperoxic (100% O2) static 807-1262 s, mean 1034 s. Bounds = Fisher-z "
        "95% CI (derived). Do not use for air-breathing checks."),
    "r2_lung_volume_single_predictor": _c(
        0.29, 0.29, 0.45, "r^2",
        "Computed from Schagatay 2012, Fernandez 2017, Bain 2017", SCHAGATAY_2012_URL, "",
        "derived",
        "Lung volume alone explains about 30-45% of between-diver variance in small cohorts, "
        "on different outcomes. No multivariable R^2 exists in these papers."),
    "sigma_pop_ln_static": _c(
        0.30, 0.29, 0.40, "SD of ln(seconds)",
        "Derived from Fernandez 2017 (CV 0.30) and Schagatay 2012 (CV 0.41)",
        FERNANDEZ_2017_URL, "", "derived",
        "sqrt(ln(1+CV^2)). Assumes lognormal; no source publishes a log-scale SD."),

    # ---------------- Non 2023 race-free (AGH) spirometry, NHANES nh3700 ----------
    # Predicted or LLN (L) = exp(b0 + b_age*age_y + b_female*female + b_height*height_cm)
    "non2023_fev1_median": _c(
        (-0.30931, -0.00966, -0.16027, 0.01177), None, None,
        "(b0, age/y, female=1, height/cm) on ln(L)", NON_2023, NON_2023_URL,
        "predicted (or lower limit of normal) FEV1 (L)= e^(b0 + Age*b1 + Gender code term*b2 "
        "+ Height*b3 + Race code term*b4)", "verified-in-source", "Table 2, R1 0.485."),
    "non2023_fev1_p05": _c(
        (-0.37667, -0.01069, -0.16976, 0.01091), None, None,
        "(b0, age/y, female=1, height/cm) on ln(L)", NON_2023, NON_2023_URL,
        "Table 2, NHANES AGH 5th percentile", "verified-in-source"),
    "non2023_fvc_median": _c(
        (-0.44468, -0.00789, -0.14771, 0.01335), None, None,
        "(b0, age/y, female=1, height/cm) on ln(L)", NON_2023, NON_2023_SUPP_URL,
        "e-Table 2, NHANES FVC AGH median", "verified-in-source", "Supplement mmc1, R1 0.472."),
    "non2023_fvc_p05": _c(
        (-0.41954, -0.00834, -0.18078, 0.01185), None, None,
        "(b0, age/y, female=1, height/cm) on ln(L)", NON_2023, NON_2023_SUPP_URL,
        "e-Table 2, NHANES FVC AGH 5th percentile", "verified-in-source"),
}


def non2023(key, age_y, female, height_cm):
    """Non 2023 AGH equation in litres. Fitted on ages 35-79."""
    b0, ba, bf, bh = CONSTANTS[key]["value"]
    return math.exp(b0 + ba * age_y + bf * (1 if female else 0) + bh * height_cm)


def btps_to_stpd(pb_mmhg=760.0, temp_c=37.0, ph2o_mmhg=47.0):
    """Gas-law factor from definitions (derived, no source states the number)."""
    return (pb_mmhg - ph2o_mmhg) / 760.0 * 273.0 / (273.0 + temp_c)
ECSC1993_CITATION = (
    "Quanjer PH, Tammeling GJ, Cotes JE, Pedersen OF, Peslin R, Yernault JC. Lung volumes "
    "and forced ventilatory flows. Eur Respir J 1993;6 Suppl 16:5-40, Table 6 (p.26)")
ECSC1993_URL = ("https://web.archive.org/web/20230208012841id_/"
                "https://erj.ersjournals.com/content/erj/6/Suppl_16/5.full.pdf")
ECSC1993_STATUS = "verified-in-source"  # 1995 erratum (Eur Respir J 1995;8:1629) not seen
ECSC1993_QUOTE = ("The lower 5 or upper 95 percentiles are obtained by subtracting or adding "
                  "the figure in the last column from the predicted mean.")

# (height coef [per m], age coef [per y], intercept, RSD, printed 1.64*RSD)
ECSC1993 = {
    "male": {
        "TLC":     (7.99,  0.0,   -7.08, 0.70, 1.15),   # L
        "RV":      (1.31,  0.022, -1.23, 0.41, 0.67),   # L
        "FRC":     (2.34,  0.009, -1.09, 0.60, 0.99),   # L; 0.009 per original, not 0.01
        "RV/TLC":  (0.0,   0.39,  13.96, 5.46, 9.0),    # %
        "FRC/TLC": (0.0,   0.21,  43.8,  6.74, 11.1),   # %
        "IVC":     (6.10, -0.028, -4.65, 0.56, 0.92),   # L
        "FVC":     (5.76, -0.026, -4.34, 0.61, 1.00),   # L
    },
    "female": {
        "TLC":     (6.60,  0.0,   -5.79, 0.60, 0.99),
        "RV":      (1.81,  0.016, -2.00, 0.35, 0.58),
        "FRC":     (2.24,  0.001, -1.00, 0.50, 0.82),
        "RV/TLC":  (0.0,   0.34,  18.96, 5.83, 9.6),
        "FRC/TLC": (0.0,   0.16,  45.1,  5.93, 9.8),
        "IVC":     (4.66, -0.026, -3.28, 0.42, 0.69),   # 0.026 per original, not 0.024
        "FVC":     (4.43, -0.026, -2.89, 0.43, 0.71),
    },
}
ECSC1993_UNITS = {"TLC": "L", "RV": "L", "FRC": "L", "IVC": "L", "FVC": "L",
                  "RV/TLC": "%", "FRC/TLC": "%"}
ECSC1993_AGE_RANGE_Y = (18, 70)
ECSC1993_HEIGHT_RANGE_M = {"male": (1.55, 1.95), "female": (1.45, 1.80)}


def ecsc1993(param, sex, height_m, age_y):
    """ECSC/ERS 1993 summary equation, adults of European descent.

    Returns pred, rsd, lower5 (pred - 1.64 RSD), upper95, in_range.
    Ages 18-25 are entered as 25, as the source instructs. Take percentiles from the
    1.64*RSD column, not from the swapped 'CI' columns in Stocks & Quanjer 1995 Table 8.
    """
    bh, ba, c, rsd, k = ECSC1993[sex][param]
    pred = bh * height_m + ba * max(age_y, 25.0) + c
    lo_h, hi_h = ECSC1993_HEIGHT_RANGE_M[sex]
    lo_a, hi_a = ECSC1993_AGE_RANGE_Y
    in_range = lo_a <= age_y <= hi_a and lo_h <= height_m <= hi_h
    return dict(pred=pred, rsd=rsd, lower5=pred - k, upper95=pred + k, in_range=in_range)
NADLER_CITATION = (
    "Nadler SB, Hidalgo JH, Bloch T. Prediction of blood volume in normal human adults. "
    "Surgery 1962;51(2):224-232. PMID 21936146 (no DOI). Coefficients as reproduced in "
    "Jt Dis Relat Surg 2026, doi:10.52312/jdrs.2026.2482 (PMC12788779) and PLoS One 2026, "
    "doi:10.1371/journal.pone.0356264 (PMC13541158)")
NADLER_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12788779/fullTextXML"
NADLER_STATUS = "verified-in-secondary"  # 1962 original not opened
NADLER_QUOTE = "PBV (L) = k1 x height (m)^3 + k2 x weight (kg) + k3"

NADLER = {"male": (0.3669, 0.03219, 0.6041), "female": (0.3561, 0.03308, 0.1833)}


def nadler_blood_volume_l(height_m, weight_kg, sex):
    k1, k2, k3 = NADLER[sex]
    return k1 * height_m ** 3 + k2 * weight_kg + k3
NHANES = {
    "cycles": {"2007-2008": ("2007", "E"), "2009-2010": ("2009", "F"), "2011-2012": ("2011", "G")},
    "xpt_url": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{stem}_{suffix}.xpt",
    "doc_url": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{stem}_{suffix}.htm",
    # Content-Length in bytes, HTTP 200 on 2026-10-22 (E, F, G). None = exists, size not recorded.
    "files": {
        "SPX":  (2841920, 3007360, 2751120),
        "DEMO": (3498080, 3631600, 3753760),
        "BMX":  (1800240, 1890560, 1946720),
        "SMQ":  (2449520, 2580560, 1946960),
        "MCQ":  (7127840, 6562720, 6905520),
        "RDQ":  (None, None, None),
    },
    "join_key": "SEQN",
    "variables": {
        "SPX": {
            "SPXNFVC": "baseline FVC, mL (divide by 1000 for L)",
            "SPXNFEV1": "baseline FEV1, mL",
            "SPXNQFVC": "FVC quality grade A-F (A/B meet ATS; D, F not used)",
            "SPXNQFV1": "FEV1 quality grade A-F",
            "SPXNSTAT": "1 complete (best FEV1 and FVC graded A-C), 2 partial, 3 not done, "
                        "4 safety exclusion",
            "SPXBFVC": "post-bronchodilator FVC, mL (only those with obstruction, 611 in 2007-08)",
            "SPXBFEV1": "post-bronchodilator FEV1, mL",
            "ENQ100": "recent respiratory illness, suggested analytic filter",
        },
        "DEMO": {
            "RIAGENDR": "1 male, 2 female",
            "RIDAGEYR": "age at screening, 80+ topcoded to 80",
            "RIDRETH1": "1 Mexican American, 2 Other Hispanic, 3 NH White, 4 NH Black, 5 Other",
            "WTMEC2YR": "MEC weight; pooled 3 cycles: MEC6YR = WTMEC2YR / 3",
        },
        "BMX": {"BMXHT": "standing height, cm (hard edits 70.0-208.4)"},
        "SMQ": {
            "SMQ020": "smoked >=100 cigarettes in life: 1 yes, 2 no, 7 refused, 9 don't know; "
                      "asked from age 20",
            "SMQ040": "now smoke: 1 every day, 2 some days, 3 not at all (only if SMQ020 == 1)",
            "SMQ620": "youth 12-19, E and F only: ever tried cigarettes, 2 = no",
            "SMQ621": "youth 12-19, G only: 1 never, not even a puff",
        },
        "MCQ": {
            "MCQ010": "asthma ever (1 yes, 2 no, 7, 9), age 1+",
            "MCQ160G": "emphysema ever, age 20+",
            "MCQ160K": "chronic bronchitis ever, age 20+",
        },
        "RDQ": {  # names from the Non 2023 supplement; response codes NOT verified
            "RDQ031": "chronic cough over three months",
            "RDQ050": "phlegm over three months",
            "RDQ070": "wheeze in past year",
            "RDQ140": "dry cough in past year",
        },
    },
    # Mirror of Non et al. 2023 nh3700 (e-Appendix 1, e-Figure 1A, Table 3).
    "cohort_non2023": {
        "start": "20,050 participants with spirometry, 2007-2012",
        "age": "RIDAGEYR > 35 and <= 79 (text: 'greater than age 35'; Table 3: '35-79 y')",
        "never_smoker": "SMQ020 == 2 (supplement misprints it as SMG020)",
        "exclude_symptoms": ["RDQ070", "RDQ140", "RDQ050", "RDQ031"],
        "exclude_missing": ["RIAGENDR", "RIDAGEYR", "BMXHT"],
        "volume_source": "SPXB* when present, else SPXN* (post-BD used when available)",
        "outliers": "Mahalanobis, alpha 0.2, on ppFEV1, FEV1/FVC and bronchodilator response "
                    "(4258 -> 3700)",
        "quality_grade_filter": None,   # none stated in the supplement
        "mcq_exclusions": None,         # none stated in the supplement
        "expected_n": 3700,
        "expected_female": 2160,
        "exact_seqn_lists": "supplement mmc2.docx (3,700 SEQN) and mmc3.docx (785 SEQN, nh785)",
        "supplement_url": "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10925545/supplementaryFiles",
        "fit": "quantile regression, tau 0.5 (predicted) and 0.05 (LLN), "
               "ln(L) ~ age + female + height_cm, R quantreg 5.55; weighting not stated",
    },
}
# Template copied from https://wwwn.cdc.gov/nchs/nhanes/NhanesCitation.aspx (fetched 2026-10-22).
NHANES_CITATION_TEMPLATE = (
    "Centers for Disease Control and Prevention (CDC). National Center for Health Statistics "
    "(NCHS). National Health and Nutrition Examination Survey Data. Hyattsville, MD: U.S. "
    "Department of Health and Human Services, Centers for Disease Control and Prevention, "
    "[appropriate year][web URL].")
NHANES_CITATION = (
    "Centers for Disease Control and Prevention (CDC). National Center for Health Statistics "
    "(NCHS). National Health and Nutrition Examination Survey Data. Hyattsville, MD: U.S. "
    "Department of Health and Human Services, Centers for Disease Control and Prevention, "
    "2007-2012. https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/SPX_E.htm")
if __name__ == "__main__":
    assert abs(non2023("non2023_fev1_median", 40, False, 175) - 3.91) < 0.01
    assert abs(non2023("non2023_fvc_median", 40, False, 175) - 4.84) < 0.01
    assert abs(non2023("non2023_fvc_p05", 40, False, 175) - 3.75) < 0.01
    assert abs(nadler_blood_volume_l(1.75, 80, "male") - 5.15) < 0.01
    assert abs(btps_to_stpd() - 0.826) < 0.001
    r = ecsc1993("FVC", "male", 1.80, 40)
    assert abs(r["pred"] - 4.988) < 1e-9 and abs(r["lower5"] - 3.988) < 1e-9 and r["in_range"]
    assert ecsc1993("TLC", "female", 1.65, 20)["pred"] == ecsc1993("TLC", "female", 1.65, 25)["pred"]
    for sex, rows in ECSC1993.items():
        for p, (_, _, _, rsd, k) in rows.items():
            assert abs(1.64 * rsd - k) <= 0.01 * k + 0.01, (sex, p)  # printed column ~ 1.64*RSD
    s = CONSTANTS
    assert sum(s[k]["value"] for k in ("rahn_lung_o2_ml", "rahn_blood_o2_ml",
               "rahn_myoglobin_o2_ml", "rahn_tissue_dissolved_o2_ml")) == 1996
    for k, e in s.items():
        assert e["status"] in {"verified-in-source", "verified-in-secondary", "derived", "unverified"}, k
    print("ok")
