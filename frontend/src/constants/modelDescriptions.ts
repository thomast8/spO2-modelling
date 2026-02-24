/**
 * Shared model parameter descriptions and model documentation content.
 * Single source of truth for tooltips, descriptions, and the About page.
 */

export const PARAM_DESCRIPTIONS: Record<string, string> = {
  o2_start:
    "Total oxygen stored in the lungs at the start of the breath hold (mL). " +
    "Depends on lung volume at hold onset: ~2000-2800 mL for full lungs (FL), " +
    "~800-1500 mL at FRC, ~400-1000 mL at residual volume (RV). " +
    "Higher values delay the onset of desaturation.",
  vo2:
    "Oxygen consumption rate (mL/min). Reflects the subject's metabolic demand " +
    "during apnea. Typical resting values are 200-250 mL/min. " +
    "Higher VO\u2082 causes faster O\u2082 depletion and earlier desaturation.",
  scale:
    "Conversion factor from remaining O\u2082 (mL) to an effective PaO\u2082-like " +
    "unit used by the Hill equation. Acts as a bridge between the O\u2082 store " +
    "model and the dissociation curve. Not a direct physiological measurement.",
  p50:
    "The effective PaO\u2082 value at which SpO\u2082 equals 50% (mmHg-equivalent). " +
    "Controls the horizontal position of the sigmoidal dissociation curve. " +
    "Higher P50 shifts the curve rightward, meaning saturation drops at higher O\u2082 levels.",
  n:
    "Hill coefficient representing haemoglobin cooperativity. Controls the " +
    "steepness of the sigmoidal dissociation curve. Normal physiological value " +
    "is ~2.7. Higher values produce a steeper transition between the plateau " +
    "and the steep desaturation region.",
  r_offset:
    "Constant residual offset added to the base SpO\u2082 prediction (% SpO\u2082). " +
    "Accounts for systematic bias from sensor placement, calibration drift, " +
    "or consistent differences between arterial and peripheral saturation.",
  r_decay:
    "Amplitude of the transient residual correction (% SpO\u2082). Models the " +
    "initial ischaemic response at the sensor site that decays over time. " +
    "Positive values add an initial upward bias that fades as blood flow equilibrates.",
  tau_decay:
    "Time constant for the exponential decay of the transient residual (seconds). " +
    "Controls how quickly the initial sensor transient dissipates. " +
    "Larger values mean the transient effect persists longer into the hold.",
  lag:
    "Circulatory delay between arterial O\u2082 changes and finger pulse oximeter " +
    "readings (seconds). Accounts for the transit time from lungs to the " +
    "peripheral measurement site. Typically 10-30 seconds.",
};

export const MODEL_SUMMARY = {
  title: "Hill Equation Oxygen-Haemoglobin Dissociation Model",
  description:
    "Predicts SpO\u2082 desaturation during breath-hold apnoea by modelling " +
    "three coupled processes: linear O\u2082 depletion from lung stores, " +
    "the sigmoidal oxygen-haemoglobin dissociation curve (Hill equation), " +
    "and a residual correction for sensor and circulatory artefacts.",
  equations: [
    {
      label: "O\u2082 depletion",
      formula: "O\u2082(t) = O\u2082_start \u2212 (VO\u2082 / 60) \u00D7 max(t \u2212 lag, 0)",
      latex: "O_2(t) = O_{2,\\text{start}} - \\frac{\\dot{V}O_2}{60} \\cdot \\max(t - \\text{lag},\\, 0)",
    },
    {
      label: "Effective PaO\u2082",
      formula: "PaO\u2082_eff = O\u2082(t) / scale",
      latex: "PaO_{2,\\text{eff}} = \\frac{O_2(t)}{\\text{scale}}",
    },
    {
      label: "Hill equation",
      formula: "SpO\u2082_base = 100 \u00D7 PaO\u2082_eff\u207F / (PaO\u2082_eff\u207F + P50\u207F)",
      latex: "SpO_{2,\\text{base}} = 100 \\cdot \\frac{PaO_{2,\\text{eff}}^{\\,n}}{PaO_{2,\\text{eff}}^{\\,n} + P_{50}^{\\,n}}",
    },
    {
      label: "Final prediction",
      formula: "SpO\u2082(t) = SpO\u2082_base + r_offset + r_decay \u00D7 e^(\u2212t / \u03C4_decay)",
      latex: "SpO_2(t) = SpO_{2,\\text{base}} + r_{\\text{offset}} + r_{\\text{decay}} \\cdot e^{-t / \\tau_{\\text{decay}}}",
    },
  ],
};

export const MODEL_COMPONENTS = [
  {
    title: "Oxygen Depletion",
    icon: "lungs",
    summary:
      "During a breath hold, lung O\u2082 stores deplete at a constant rate " +
      "determined by the subject's metabolic demand (VO\u2082).",
    detail:
      "The model starts with an initial oxygen store (O\u2082_start) that depends on " +
      "the lung volume at the onset of the hold. Oxygen is consumed at a constant " +
      "rate (VO\u2082), but consumption only begins after the circulatory lag period. " +
      "This simple linear depletion captures the dominant mechanism driving " +
      "desaturation: the finite O\u2082 reservoir shrinks over time.",
    equation: "O\u2082(t) = O\u2082_start \u2212 (VO\u2082 / 60) \u00D7 max(t \u2212 lag, 0)",
    latex: "O_2(t) = O_{2,\\text{start}} - \\frac{\\dot{V}O_2}{60} \\cdot \\max(t - \\text{lag},\\, 0)",
    params: ["o2_start", "vo2"],
  },
  {
    title: "Dissociation Curve (Hill Equation)",
    icon: "curve",
    summary:
      "The sigmoidal oxygen-haemoglobin dissociation curve converts O\u2082 levels " +
      "into SpO\u2082 percentage using the Hill equation.",
    detail:
      "The Hill equation models how haemoglobin binds oxygen cooperatively: " +
      "at high O\u2082 levels, saturation plateaus near 100% (the flat shoulder), " +
      "but once O\u2082 drops below a critical threshold, saturation falls steeply. " +
      "P50 determines where this transition occurs, while the Hill coefficient (n) " +
      "controls how abrupt the transition is. The scale parameter converts the " +
      "remaining O\u2082 volume into an effective partial pressure that the Hill " +
      "equation can use.",
    equation: "SpO\u2082_base = 100 \u00D7 PaO\u2082_eff\u207F / (PaO\u2082_eff\u207F + P50\u207F)",
    latex: "SpO_{2,\\text{base}} = 100 \\cdot \\frac{PaO_{2,\\text{eff}}^{\\,n}}{PaO_{2,\\text{eff}}^{\\,n} + P_{50}^{\\,n}}",
    params: ["scale", "p50", "n"],
  },
  {
    title: "Residual Correction",
    icon: "correction",
    summary:
      "A combined constant + exponentially decaying correction term that accounts " +
      "for sensor bias and transient ischaemic effects.",
    detail:
      "Finger pulse oximetry doesn't perfectly track arterial saturation. " +
      "The residual offset (r_offset) captures constant biases from sensor " +
      "placement and calibration. The decaying transient (r_decay \u00D7 e^(\u2212t/\u03C4)) " +
      "models the initial ischaemic response at the fingertip: when a breath hold " +
      "begins, peripheral blood flow and oxygenation can temporarily differ from " +
      "arterial values before equilibrating. Together, these two terms allow the " +
      "model to match the real sensor signal rather than ideal arterial SpO\u2082.",
    equation: "residual(t) = r_offset + r_decay \u00D7 e^(\u2212t / \u03C4_decay)",
    latex: "\\text{residual}(t) = r_{\\text{offset}} + r_{\\text{decay}} \\cdot e^{-t \\,/\\, \\tau_{\\text{decay}}}",
    params: ["r_offset", "r_decay", "tau_decay"],
  },
  {
    title: "Finger-to-Arterial Lag",
    icon: "lag",
    summary:
      "A time delay accounting for the circulatory transit time from lungs " +
      "to the fingertip sensor.",
    detail:
      "Changes in arterial oxygen at the lungs take several seconds to reach " +
      "the finger pulse oximeter. This lag parameter shifts the effective time " +
      "axis so that O\u2082 depletion begins after the delay, matching the observed " +
      "sensor response. The lag is typically 10-30 seconds and depends on cardiac " +
      "output and peripheral vascular resistance.",
    equation: "t_eff = max(t \u2212 lag, 0)",
    latex: "t_{\\text{eff}} = \\max(t - \\text{lag},\\, 0)",
    params: ["lag"],
  },
];

export const HOLD_TYPE_DESCRIPTIONS: Record<string, { name: string; description: string; o2Range: string }> = {
  FRC: {
    name: "Functional Residual Capacity",
    description:
      "Hold initiated at the end of a normal exhalation, with the lungs at " +
      "their natural resting volume. Represents the most common starting " +
      "condition for involuntary apnoea events.",
    o2Range: "800 \u2013 1,500 mL",
  },
  RV: {
    name: "Residual Volume",
    description:
      "Hold initiated after a maximal forced exhalation, with the lungs at " +
      "their minimum volume. Produces the fastest desaturation due to the " +
      "smallest initial O\u2082 store.",
    o2Range: "400 \u2013 1,000 mL",
  },
  FL: {
    name: "Full Lungs",
    description:
      "Hold initiated after a maximal inhalation (total lung capacity). " +
      "Provides the largest O\u2082 store and therefore the slowest desaturation, " +
      "often used for longer breath-hold tests.",
    o2Range: "1,800 \u2013 2,800 mL",
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
