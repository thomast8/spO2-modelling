/**
 * Shared v7 model parameter descriptions and model documentation content.
 * Single source of truth for tooltips, descriptions, and the About page.
 */

export const PARAM_DESCRIPTIONS: Record<string, string> = {
  tau_0:
    "Base sensor lag (seconds). The mean delay between arterial SaO\u2082 changes " +
    "and pulse oximeter SpO\u2082 readings. Modelled as a gamma kernel convolution.",
  p:
    "Power-law descent exponent. Controls the shape of the latent SpO\u2082 descent " +
    "during apnea. Higher values produce a sharper transition near the end of apnea.",
  delta:
    "Per-hold lag offset (seconds). Adjusts the effective sensor lag for each hold " +
    "relative to the global tau_0. Positive = more delay, negative = less.",
  s_min:
    "Nadir SpO\u2082 in the latent template (%). The lowest point the latent signal " +
    "reaches during apnea, before sensor smoothing is applied.",
  v_up:
    "Recovery slope (%/s). Rate of SpO\u2082 increase after the apnea nadir. " +
    "Higher values mean faster recovery.",
  k_co2:
    "CO\u2082 accumulation rate (mmHg/s). Rate at which alveolar PCO\u2082 rises " +
    "during apnea, driving the Bohr effect rightward shift of the ODC. " +
    "Typical value is ~0.06 mmHg/s.",
  gamma:
    "Severinghaus ODC steepness exponent. Controls how abrupt the " +
    "sigmoidal transition is. 1.0 = standard population-average curve, " +
    ">1.0 = steeper. Captures individual variation in Hb cooperativity.",
  tau_washout:
    "Exponential O\u2082 washout time constant (seconds). Controls how fast alveolar " +
    "O\u2082 equilibrates with venous blood. Depends on lung volume: " +
    "FL ~50-500s, FRC ~20-100s, RV ~10-80s.",
  paco2_0:
    "Initial alveolar PCO\u2082 (mmHg). Starting CO\u2082 level at the beginning of " +
    "apnea. Normal is ~40 mmHg. Affects the initial P50 via the Bohr effect.",
  p50_base:
    "Fixed constant: baseline P50 of the oxygen-haemoglobin dissociation curve " +
    "(26.6 mmHg). Not fitted.",
  pvo2_fixed:
    "Fixed mixed venous PO\u2082 (25 mmHg). The asymptotic floor that alveolar " +
    "PO\u2082 decays toward during apnea.",
};

export const MODEL_SUMMARY = {
  title: "v7 Two-Stage Apnea Desaturation Model",
  description:
    "Two-stage architecture: Stage A calibrates the pulse oximeter sensor response " +
    "(power-law latent + gamma kernel convolution), Stage B fits the underlying " +
    "physiology (exponential PaO\u2082 washout + linear CO\u2082 accumulation + " +
    "Severinghaus ODC) with the sensor model frozen from Stage A.",
  equations: [
    {
      label: "Alveolar washout",
      formula: "PAO\u2082(t) = PvO\u2082 + (PAO\u2082\u2080 \u2212 PvO\u2082) \u00D7 e^(\u2212t / \u03C4)",
      latex: "PAO_2(t) = PvO_2 + (PAO_{2,0} - PvO_2) \\cdot e^{-t / \\tau_{\\text{washout}}}",
    },
    {
      label: "CO\u2082 Bohr effect",
      formula: "P50_eff(t) = P50_BASE + 0.48 \u00D7 (PACO\u2082(t) \u2212 40)",
      latex: "P_{50,\\text{eff}}(t) = P_{50,\\text{BASE}} + 0.48 \\cdot (P_{ACO_2,0} + k_{CO_2} \\cdot t - 40)",
    },
    {
      label: "Severinghaus ODC",
      formula: "SaO\u2082 = 100 / (1 + 23400/(PAO\u2082_adj\u00B3 + 150\u00D7PAO\u2082_adj))",
      latex: "SaO_2 = \\frac{100}{1 + \\frac{23400}{\\widetilde{P}^{\\,3} + 150\\,\\widetilde{P}}}",
    },
    {
      label: "Sensor convolution",
      formula: "SpO\u2082(t) = GammaKernel(SaO\u2082, tau_0 + delta, cv)",
      latex: "SpO_2(t) = (SaO_2 * h_{\\Gamma})(t), \\quad h \\sim \\Gamma(\\tau_0 + \\delta, \\text{cv})",
    },
  ],
};

export const MODEL_COMPONENTS = [
  {
    title: "Stage A: Power-Law Latent Template",
    icon: "sensor",
    summary:
      "Models the SpO\u2082 probe's intrinsic response pattern during apnea: " +
      "a power-law descent to S_min, followed by linear recovery at rate v_up.",
    detail:
      "The latent template captures the idealized shape of the SpO\u2082 signal " +
      "before sensor smoothing. During apnea (t <= t_end), SpO\u2082 descends from " +
      "S_start to S_min following a power-law curve with exponent p. After t_end, " +
      "recovery is linear at rate v_up. This template is then convolved with a " +
      "gamma kernel to model the sensor's temporal smoothing.",
    equation: "L(t) = S_min + (S_start - S_min) * (1 - (t/t_end)^p)",
    latex: "L(t) = S_{\\min} + (S_{\\text{start}} - S_{\\min}) \\cdot (1 - (t/t_{\\text{end}})^p)",
    params: ["tau_0", "p", "delta", "s_min", "v_up"],
  },
  {
    title: "Stage A: Gamma Kernel Sensor Delay",
    icon: "delay",
    summary:
      "A discrete gamma distribution kernel smooths and delays the latent signal, " +
      "modelling the pulse oximeter's temporal response characteristics.",
    detail:
      "The gamma kernel has mean lag = tau_0 + delta and coefficient of variation " +
      "CV = 0.15 (fixed). Convolution with this kernel produces the characteristic " +
      "delayed, smoothed SpO\u2082 trace seen in real pulse oximetry data. " +
      "Per-hold delta offsets allow individual hold timing variations.",
    equation: "SpO\u2082(t) = (L * h_\u0393)(t)",
    latex: "SpO_2(t) = (L * h_{\\Gamma})(t)",
    params: ["tau_0", "delta"],
  },
  {
    title: "Stage B: Exponential Alveolar Washout",
    icon: "lungs",
    summary:
      "Alveolar O\u2082 declines exponentially as O\u2082 transfers from alveoli to blood " +
      "at a rate proportional to the alveolar-venous PO\u2082 gradient.",
    detail:
      "PAO\u2082 starts at pao2_0 (derived from paco2_0 via the alveolar gas equation) " +
      "and decays toward PvO\u2082 = 25 mmHg (fixed). The time constant tau_washout " +
      "depends on lung volume: larger lungs (FL) have larger tau, meaning slower " +
      "washout and a longer SpO\u2082 plateau.",
    equation: "PAO\u2082(t) = PvO\u2082 + (PAO\u2082\u2080 - PvO\u2082) * e^(-t/\u03C4)",
    latex: "PAO_2(t) = PvO_2 + (PAO_{2,0} - PvO_2) \\cdot e^{-t / \\tau_{\\text{washout}}}",
    params: ["tau_washout", "paco2_0"],
  },
  {
    title: "Stage B: Linear CO\u2082 Bohr Effect",
    icon: "correction",
    summary:
      "CO\u2082 accumulates linearly during apnea, progressively right-shifting " +
      "the ODC via the Bohr effect and accelerating late-phase desaturation.",
    detail:
      "During breath-holding, CO\u2082 rises at rate k_co2 mmHg/s. The Bohr coefficient " +
      "(0.48 mmHg P50 shift per mmHg CO\u2082 above normal) converts CO\u2082 accumulation " +
      "into a P50 increase, meaning haemoglobin releases oxygen more readily. " +
      "Unlike the saturating Bohr model in earlier versions, this linear model is " +
      "appropriate because apnea durations are short enough that CO\u2082 accumulation " +
      "remains approximately linear.",
    equation: "P50_eff(t) = 26.6 + 0.48 * (PACO\u2082\u2080 + k_co2*t - 40)",
    latex: "P_{50,\\text{eff}}(t) = P_{50,\\text{BASE}} + 0.48 \\cdot (P_{ACO_2,0} + k_{CO_2} \\cdot t - 40)",
    params: ["k_co2", "paco2_0"],
  },
  {
    title: "Stage B: Severinghaus ODC with Gamma Steepness",
    icon: "curve",
    summary:
      "The Severinghaus (1979) equation converts PAO\u2082 into SaO\u2082, with a " +
      "gamma exponent controlling individual ODC steepness.",
    detail:
      "The Severinghaus equation models the oxygen-haemoglobin dissociation curve. " +
      "The Bohr effect is implemented via virtual PO\u2082 scaling (right-shifting the curve), " +
      "and the gamma exponent adjusts the overall steepness to capture individual " +
      "haemoglobin cooperativity variation. gamma = 1.0 gives the standard curve; " +
      "typical fitted values are 1.5-3.0.",
    equation: "SaO\u2082 = 100 / (1 + 23400 / (P\u0303\u00B3 + 150 * P\u0303))",
    latex: "SaO_2 = \\frac{100}{1 + \\frac{23400}{\\widetilde{P}^{\\,3} + 150\\,\\widetilde{P}}}",
    params: ["gamma", "p50_base"],
  },
];

export const HOLD_TYPE_DESCRIPTIONS: Record<string, { name: string; description: string; o2Range: string }> = {
  FRC: {
    name: "Functional Residual Capacity",
    description:
      "Hold initiated at the end of a normal exhalation, with the lungs at " +
      "their natural resting volume. Represents the most common starting " +
      "condition for involuntary apnoea events.",
    o2Range: "tau_washout 20-100s",
  },
  RV: {
    name: "Residual Volume",
    description:
      "Hold initiated after a maximal forced exhalation, with the lungs at " +
      "their minimum volume. Produces the fastest desaturation due to the " +
      "smallest O\u2082 reserve and fastest washout.",
    o2Range: "tau_washout 10-80s",
  },
  FL: {
    name: "Full Lungs",
    description:
      "Hold initiated after a maximal inhalation (total lung capacity). " +
      "Provides the largest O\u2082 reserve and slowest washout, " +
      "often used for longer breath-hold tests.",
    o2Range: "tau_washout 50-500s",
  },
};

export const FITTING_DESCRIPTION = {
  title: "Two-Stage Fitting Process",
  summary:
    "The v7 model uses a two-stage fitting process. Stage A calibrates the sensor " +
    "model, then Stage B fits the physiology with the sensor frozen. " +
    "Both stages use differential evolution, a global optimizer.",
  details: [
    "Stage A fits the sensor model (tau_0, p, per-hold delta/S_min/v_up) using " +
    "Student-t negative log-likelihood loss and Huber nadir timing penalty. " +
    "Regularization includes LogNormal priors on tau_0 and p, StudentT-like " +
    "shrinkage on deltas, and a zero-sum constraint.",
    "Stage B freezes the sensor and fits physiology (k_co2, gamma, per-hold " +
    "tau_washout/paco2_0) using weighted least squares with 3x weight on SpO\u2082 < 95%. " +
    "Regularization includes Gaussian priors on k_co2 and gamma, and per-type " +
    "IC regularization to shrink per-hold params toward their type mean.",
    "Differential evolution (popsize=40 for Stage A, 25 for Stage B) evolves " +
    "candidate solutions with mutation and crossover, followed by L-BFGS-B polish.",
    "All holds of all types are fitted simultaneously in each stage, sharing " +
    "global parameters while allowing per-hold variation.",
  ],
};
