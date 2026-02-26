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
  paco2_0:
    "Initial arterial CO\u2082 partial pressure (mmHg). Normal resting PaCO\u2082 is " +
    "~40 mmHg. After hyperventilation it can drop to 20-30 mmHg, starting with " +
    "a left-shifted ODC (higher O\u2082 affinity) before CO\u2082 accumulation reverses it.",
  k_co2:
    "CO\u2082 rise rate during apnea (mmHg/s). Metabolic CO\u2082 production causes " +
    "a linear PaCO\u2082 increase. Typical values are 0.05-0.15 mmHg/s " +
    "(3-9 mmHg/min). Higher values accelerate late-phase desaturation via " +
    "the Bohr effect.",
  p50_base:
    "Fixed constant: baseline P50 of the oxygen-haemoglobin dissociation curve " +
    "(26.6 mmHg). The PaO\u2082 at which SpO\u2082 equals 50%. Not fitted \u2014 this is a " +
    "haemoglobin biochemistry constant.",
  bohr_beta:
    "Fixed constant: Bohr coefficient (0.48 mmHg P50 shift per mmHg CO\u2082). " +
    "Determines how much the P50 changes per unit change in PaCO\u2082. " +
    "Well-established from literature. Not fitted.",
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
  title: "Exponential Alveolar Washout + CO\u2082-Driven Bohr Effect Model",
  description:
    "Predicts SpO\u2082 desaturation during breath-hold apnoea by modelling " +
    "exponential alveolar O\u2082 washout (gradient-driven equilibration with venous blood), " +
    "a CO\u2082-driven Bohr effect (linear PaCO\u2082 rise with physiological Bohr coefficient), " +
    "and the Severinghaus equation.",
  equations: [
    {
      label: "Alveolar washout",
      formula: "PAO\u2082(t) = PvO\u2082 + (PAO\u2082\u2080 \u2212 PvO\u2082) \u00D7 e^(\u2212t / \u03C4)",
      latex: "PAO_2(t) = PvO_2 + (PAO_{2,0} - PvO_2) \\cdot e^{-t / \\tau}",
    },
    {
      label: "CO\u2082-driven Bohr effect",
      formula: "PaCO\u2082(t) = PaCO\u2082\u2080 + k_CO\u2082 \u00D7 t;  P50_eff = 26.6 + 0.48 \u00D7 (PaCO\u2082 \u2212 40)",
      latex: "P_{50,\\text{eff}}(t) = P_{50,\\text{BASE}} + \\beta \\cdot \\left(\\underbrace{PaCO_{2,0} + k_{CO_2} \\cdot t}_{PaCO_2(t)} - 40\\right)",
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
    title: "CO\u2082-Driven Bohr Effect",
    icon: "correction",
    summary:
      "CO\u2082 accumulation during apnea lowers blood pH, progressively " +
      "right-shifting the ODC and accelerating late-phase desaturation.",
    detail:
      "During breath-holding, CO\u2082 is continuously produced by metabolism but " +
      "cannot be exhaled. PaCO\u2082 rises linearly from paco2_0 (which may be below 40 mmHg " +
      "after hyperventilation) at rate k_co2. The Bohr coefficient (\u03B2 = 0.48 mmHg/mmHg, " +
      "fixed from literature) converts the CO\u2082 change into a P50 shift: " +
      "\u0394P50 = 0.48 \u00D7 (PaCO\u2082 \u2212 40). This can start negative " +
      "(left-shifted, higher affinity) if the subject hyperventilated, then cross " +
      "zero and become a rightward shift as CO\u2082 accumulates beyond 40 mmHg.",
    equation: "P50_eff(t) = P50_BASE + \u03B2 \u00D7 (PaCO\u2082\u2080 + k_CO\u2082\u00D7t \u2212 40)",
    latex: "P_{50,\\text{eff}}(t) = P_{50,\\text{BASE}} + \\beta \\cdot \\left(PaCO_{2,0} + k_{CO_2} \\cdot t - 40\\right)",
    params: ["paco2_0", "k_co2", "bohr_beta"],
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
