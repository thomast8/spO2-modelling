/**
 * Shared model parameter descriptions and model documentation content.
 * Single source of truth for tooltips, descriptions, and the About page.
 */

export const PARAM_DESCRIPTIONS: Record<string, string> = {
  pao2_0:
    "Initial alveolar partial pressure of oxygen (mmHg). Depends on lung volume " +
    "and pre-oxygenation: ~100-200 mmHg for full lungs (FL), ~80-120 at FRC, " +
    "~70-110 at residual volume (RV). Higher values extend the SpO\u2082 plateau.",
  pvo2:
    "Mixed venous PO\u2082 (mmHg). The asymptotic floor that alveolar PO\u2082 decays " +
    "toward during apnea. Represents the PO\u2082 of blood returning to the lungs. " +
    "Normal resting value is ~40 mmHg.",
  tau_washout:
    "Exponential O\u2082 washout time constant (seconds). Controls how fast alveolar " +
    "O\u2082 equilibrates with venous blood. Larger values mean slower O\u2082 depletion " +
    "and a longer SpO\u2082 plateau. Depends on lung volume and metabolic rate.",
  bohr_max:
    "Maximum Bohr effect P50 shift (mmHg). The asymptotic upper bound of the " +
    "CO\u2082-driven P50 increase during apnea. Real physiological Bohr shift during " +
    "a 3-minute apnea is ~3-6 mmHg. Accelerates late-phase desaturation.",
  tau_bohr:
    "CO\u2082 accumulation time constant (seconds). Controls how fast the Bohr " +
    "P50 shift saturates. Smaller values mean CO\u2082 effects kick in earlier. " +
    "Typical values are 80-150 seconds.",
  p50_base:
    "Fixed constant: baseline P50 of the oxygen-haemoglobin dissociation curve " +
    "(26.6 mmHg). The PaO\u2082 at which SpO\u2082 equals 50%. Not fitted \u2014 this is a " +
    "haemoglobin biochemistry constant.",
  gamma:
    "Steepness exponent for the Severinghaus ODC. Controls how abrupt the " +
    "sigmoidal transition is. 1.0 = standard population-average curve, " +
    ">1.0 = steeper. Captures individual variation in Hb cooperativity.",
  r_offset:
    "Constant SpO\u2082 offset for sensor calibration (%). Accounts for systematic " +
    "bias from sensor placement, calibration drift, or consistent differences " +
    "between arterial and peripheral saturation.",
};

export const MODEL_SUMMARY = {
  title: "Exponential Alveolar Washout + Bohr Effect Model",
  description:
    "Predicts SpO\u2082 desaturation during breath-hold apnoea by modelling " +
    "exponential alveolar O\u2082 washout (gradient-driven equilibration with venous blood), " +
    "the Bohr effect (CO\u2082-driven rightward ODC shift), and the Severinghaus equation.",
  equations: [
    {
      label: "Alveolar washout",
      formula: "PAO\u2082(t) = PvO\u2082 + (PAO\u2082\u2080 \u2212 PvO\u2082) \u00D7 e^(\u2212t / \u03C4)",
      latex: "PAO_2(t) = PvO_2 + (PAO_{2,0} - PvO_2) \\cdot e^{-t / \\tau}",
    },
    {
      label: "Saturating Bohr effect",
      formula: "P50_eff(t) = P50_BASE + bohr_max \u00D7 (1 \u2212 e^(\u2212t / \u03C4_bohr))",
      latex: "P_{50,\\text{eff}}(t) = P_{50,\\text{BASE}} + \\Delta_{\\max} \\cdot \\left(1 - e^{-t / \\tau_{\\text{bohr}}}\\right)",
    },
    {
      label: "Severinghaus equation",
      formula: "SpO\u2082(t) = r_offset + 100 / (1 + 23400/(PAO\u2082_adj\u00B3 + 150\u00D7PAO\u2082_adj))",
      latex: "SpO_2(t) = r_{\\text{offset}} + \\frac{100}{1 + \\frac{23400}{\\widetilde{P}^{\\,3} + 150\\,\\widetilde{P}}}",
    },
  ],
};

export const MODEL_COMPONENTS = [
  {
    title: "Exponential Alveolar Washout",
    icon: "lungs",
    summary:
      "Alveolar O\u2082 declines exponentially as O\u2082 transfers from alveoli to blood " +
      "at a rate proportional to the alveolar-venous PO\u2082 gradient.",
    detail:
      "Unlike linear depletion, the exponential model captures the physiological " +
      "reality that O\u2082 transfer slows as the gradient between alveolar and venous " +
      "PO\u2082 shrinks. PAO\u2082 starts at pao2_0 (depending on lung volume and " +
      "pre-oxygenation) and decays toward pvo2 (mixed venous PO\u2082, ~40 mmHg). " +
      "The time constant \u03C4 depends on lung volume: larger lungs (FL) have larger " +
      "\u03C4, meaning slower washout and a longer SpO\u2082 plateau.",
    equation: "PAO\u2082(t) = PvO\u2082 + (PAO\u2082\u2080 \u2212 PvO\u2082) \u00D7 e^(\u2212t / \u03C4)",
    latex: "PAO_2(t) = PvO_2 + (PAO_{2,0} - PvO_2) \\cdot e^{-t / \\tau}",
    params: ["pao2_0", "pvo2", "tau_washout"],
  },
  {
    title: "Dissociation Curve (Severinghaus Equation)",
    icon: "curve",
    summary:
      "The Severinghaus (1979) equation converts PAO\u2082 into SpO\u2082 percentage " +
      "using a rational polynomial that captures the asymmetric ODC shape.",
    detail:
      "The Severinghaus equation models the oxygen-haemoglobin dissociation curve " +
      "with a rational polynomial: at high PAO\u2082 levels (>80 mmHg), saturation " +
      "plateaus near 100% (the flat shoulder), but once PAO\u2082 drops into the " +
      "40-60 mmHg range, saturation falls steeply. Unlike the symmetric Hill equation, " +
      "Severinghaus naturally captures the real ODC\u2019s asymmetry (steeper in 40-80 mmHg). " +
      "The gamma steepness exponent adjusts the curve slope to capture individual " +
      "variation in haemoglobin cooperativity. The Bohr effect is implemented via " +
      "virtual PO\u2082 scaling, which shifts the curve rightward as CO\u2082 accumulates.",
    equation: "SpO\u2082 = 100 / (1 + 23400 / (P\u0303\u00B3 + 150 \u00D7 P\u0303))",
    latex: "SpO_2 = \\frac{100}{1 + \\frac{23400}{\\widetilde{P}^{\\,3} + 150\\,\\widetilde{P}}}",
    params: ["p50_base", "gamma"],
  },
  {
    title: "Saturating Bohr Effect",
    icon: "correction",
    summary:
      "CO\u2082 accumulation during apnea lowers blood pH, progressively " +
      "right-shifting the ODC and accelerating late-phase desaturation.",
    detail:
      "During breath-holding, CO\u2082 is continuously produced by metabolism but " +
      "cannot be exhaled. Rising PCO\u2082 decreases blood pH via the Henderson-Hasselbalch " +
      "equation. This pH drop increases P50 (the Bohr effect), meaning haemoglobin " +
      "releases oxygen more readily. The model uses a saturating exponential: " +
      "P50 rises toward a maximum shift (bohr_max, typically 3-6 mmHg) with time " +
      "constant tau_bohr. This prevents the unphysical unbounded P50 growth that " +
      "a linear model would produce at long apnea durations.",
    equation: "P50_eff(t) = P50_BASE + bohr_max \u00D7 (1 \u2212 e^(\u2212t / \u03C4_bohr))",
    latex: "P_{50,\\text{eff}}(t) = P_{50,\\text{BASE}} + \\Delta_{\\max} \\cdot \\left(1 - e^{-t / \\tau_{\\text{bohr}}}\\right)",
    params: ["bohr_max", "tau_bohr"],
  },
  {
    title: "Sensor Calibration Offset",
    icon: "offset",
    summary:
      "A constant offset for sensor calibration bias.",
    detail:
      "The offset accounts for systematic bias from sensor placement " +
      "and calibration drift, or consistent differences between arterial " +
      "and peripheral saturation. Typically small (\u00B13%).",
    equation: "SpO\u2082_final = SpO\u2082_model + r_offset",
    latex: "SpO_{2,\\text{final}} = SpO_{2,\\text{model}} + r_{\\text{offset}}",
    params: ["r_offset"],
  },
];

export const HOLD_TYPE_DESCRIPTIONS: Record<string, { name: string; description: string; o2Range: string }> = {
  FRC: {
    name: "Functional Residual Capacity",
    description:
      "Hold initiated at the end of a normal exhalation, with the lungs at " +
      "their natural resting volume. Represents the most common starting " +
      "condition for involuntary apnoea events.",
    o2Range: "PAO\u2082 80 \u2013 120 mmHg",
  },
  RV: {
    name: "Residual Volume",
    description:
      "Hold initiated after a maximal forced exhalation, with the lungs at " +
      "their minimum volume. Produces the fastest desaturation due to the " +
      "smallest O\u2082 reserve and fastest washout.",
    o2Range: "PAO\u2082 70 \u2013 110 mmHg",
  },
  FL: {
    name: "Full Lungs",
    description:
      "Hold initiated after a maximal inhalation (total lung capacity). " +
      "Provides the largest O\u2082 reserve and slowest washout, " +
      "often used for longer breath-hold tests.",
    o2Range: "PAO\u2082 100 \u2013 200 mmHg",
  },
};

export const FITTING_DESCRIPTION = {
  title: "Fitting Process",
  summary:
    "Parameters are optimised using differential evolution, a global optimisation " +
    "algorithm that explores the parameter space without requiring gradient information.",
  details: [
    "The optimizer minimises the sum of squared errors between observed and " +
    "predicted SpO\u2082 across all holds of the same type simultaneously.",
    "Differential evolution uses a population of candidate solutions (60 members) " +
    "that evolve over up to 5,000 iterations, with mutation and crossover operations " +
    "to explore the search space.",
    "After the evolutionary search, a local polish step (L-BFGS-B) refines the " +
    "best solution for maximum precision.",
    "Each hold type has its own parameter bounds reflecting the expected physiological " +
    "range for that lung volume condition.",
  ],
};
