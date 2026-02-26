# SpO2 Desaturation Model: Design, Critique, and Roadmap

This document captures the current model state, known issues, external review feedback,
and the planned v2 architecture. It serves as stable context for future development.

Last updated: 2026-02-26

---

## 1. What We're Modelling

When a freediver holds their breath, blood oxygen saturation (SpO2) — measured by a
finger pulse oximeter — follows a characteristic pattern: a **flat plateau** at ~98-100%
for some time, then a **steep sigmoidal drop** into the 50-70% range. The shape depends
on lung volume at the start of the hold:

- **FL (Full Lungs):** Largest O2 reserve, longest plateau, slowest desaturation
- **FRC (Functional Residual Capacity):** Normal resting volume, moderate speed
- **RV (Residual Volume):** Minimal O2 reserve, fastest desaturation

The goal is a mechanistic model that:
1. Fits the SpO2 curve accurately (especially the clinically important drop region)
2. Has parameters that map to identifiable physiological quantities
3. Works across hold types with per-type parameter bounds

---

## 2. Current Model (v1.2 — Severinghaus + gamma, post-lag-removal)

### Architecture

```
Input: t (seconds since hold start)
                    |
                    v
            +-------------------------------+
        1.  |  Exponential O2 Washout        |  PAO2(t) = pvo2 + (pao2_0 - pvo2) * e^(-t / tau_washout)
            +---------------+---------------+
                            v
            +-------------------------------+
        2.  |  Saturating Bohr Effect        |  P50_eff(t) = 26.6 + bohr_max * (1 - e^(-t / tau_bohr))
            +---------------+---------------+
                            v
            +-----------------------------------------------+
        3.  |  Virtual PO2 Transform                        |
            |  a) Bohr shift:  PO2_virtual = PAO2 * (26.6 / P50_eff)    |
            |  b) Steepness:   PO2_adj = 26.6 * (PO2_virtual / 26.6)^g |
            +------------------------+----------------------+
                                     v
            +----------------------------------------------+
        4.  |  Severinghaus (1979) ODC                      |
            |  SpO2 = r_offset + 100 / (1 + 23400/(PO2_adj^3 + 150*PO2_adj)) |
            +----------------------------------------------+
                                     |
                                     v
                Output: clip(SpO2, 0, 100)
```

### 7 Fitted Parameters

| Parameter     | Meaning                                      | FL Bounds  |
|---------------|----------------------------------------------|------------|
| `pao2_0`      | Initial alveolar PO2 (mmHg)                  | 100-250    |
| `pvo2`        | Asymptotic PO2 floor (mmHg)                  | 20-50      |
| `tau_washout`  | O2 depletion time constant (s)               | 50-250     |
| `gamma`       | Steepness exponent (1.0 = standard Sev.)     | 0.8-2.0    |
| `bohr_max`    | Maximum Bohr P50 shift (mmHg)                | 2-15       |
| `tau_bohr`    | CO2 saturation time constant (s)             | 40-250     |
| `r_offset`    | Sensor calibration bias (%)                  | -3 to +3   |

**Fixed constant:** `P50_BASE = 26.6 mmHg` (standard haemoglobin half-saturation).

### Fitting

- **Optimizer:** `scipy.optimize.differential_evolution`, popsize=60, maxiter=5000,
  tol=1e-12, polish=True (L-BFGS-B refinement at end)
- **Objective:** Unweighted sum of squared residuals across all holds of the same type
- **Bounds:** Per hold type (FRC, RV, FL)
- **Best fit on FL hold 6:** R^2 = 0.9956, RMSE = 0.858%, 3/8 params at bounds
  (before lag removal — needs re-benchmarking)

### Key Files

| File                                    | Role                                    |
|-----------------------------------------|-----------------------------------------|
| `backend/app/services/hill_model.py`    | Model equations, ApneaModelParams       |
| `backend/app/services/fitter.py`        | Differential evolution fitting engine   |
| `backend/app/models/db_models.py`       | SQLAlchemy ORM (ModelVersion table)     |
| `backend/app/models/schemas.py`         | Pydantic API schemas                    |
| `backend/app/routers/fit.py`            | Fit preview/save endpoints              |
| `backend/app/services/model_manager.py` | Model version CRUD                      |
| `backend/scripts/compare_odc.py`        | Standalone comparison script (Hill/Kelman/Sev) |

### Data Available

Per-second (1 Hz) from finger pulse oximeter CSV:
- **SpO2** (%) — used in fitting
- **HR** (bpm) — stored in DB but **not currently used by the model**
- **Elapsed time** (s) — relative to hold start

---

## 3. Model History

### v1.0 — Hill equation + linear Bohr
- Hill ODC: `SpO2 = 100 * PO2^n / (PO2^n + P50^n)` — symmetric on log scale
- Linear Bohr effect: `P50_eff = P50_BASE + bohr_rate * t`

### v1.1 — Hill + saturating Bohr
- Replaced linear Bohr with saturating exponential: `bohr_max * (1 - e^(-t/tau_bohr))`
- Fixed the unphysical unbounded P50 growth at long durations

### v1.2 — Severinghaus + gamma (current)
- Replaced Hill with Severinghaus (1979): `100 / (1 + 23400/(PO2^3 + 150*PO2))`
- Added gamma steepness exponent (power transform on PO2 axis)
- Added virtual PO2 Bohr shift
- Comparison results (FL hold 6):
  - Severinghaus+gamma: R^2=0.9956, 3/8 at bounds, RMSE=0.858
  - Hill (original):    R^2=0.9915, 6/8 at bounds
  - Kelman:             R^2=0.9947, 4/7 at bounds (rejected)
- Removed lag parameter (confounded with tau_washout and ODC plateau flatness)

---

## 4. Strengths

1. **Right macro-shape.** ODC nonlinearity + falling alveolar PO2 naturally produces the
   plateau-then-collapse pattern without arbitrary sigmoid fitting in time.

2. **Severinghaus captures asymmetry.** The real ODC is steeper in the 40-80 mmHg range
   than above 80. Severinghaus models this; Hill cannot.

3. **Compact.** 7 parameters, each with at least a rough physiological interpretation.

4. **Fits well.** R^2 > 0.995 on FL holds with reasonable parameter values.

5. **Robust fitting.** Differential evolution + polish avoids local minima that
   gradient-only methods get stuck in with this nonlinear model.

---

## 5. Known Issues

### 5.1 Parameter Identifiability

**Gamma and Bohr are mathematically entangled.** The full transform chain collapses to:
```
PO2_adj = P50_BASE * (PAO2 / P50_eff)^gamma
```
Gamma exponentiates the Bohr-shifted PO2, so it amplifies or dampens the Bohr shift.
A small change in `bohr_max` has drastically different effects at gamma=0.8 vs gamma=1.5.
The optimizer can trade between them freely, making both parameters less interpretable.

**pao2_0 and tau_washout are correlated.** Higher initial PO2 with longer time constant
produces the same PO2 at the time the ODC starts to bend (~60-80 mmHg). The plateau
region is information-free for distinguishing these two.

**3/8 params at bounds** (before lag removal). The optimizer wants to go further but
can't. The fitted parameters may be distorted to compensate.

### 5.2 Physics We're Getting Wrong

**pvo2 is a fixed floor, but real mixed venous PO2 declines during apnea.** Tissues
continue extracting O2 from an ever-depleting blood supply. Making pvo2 a constant
forces the model curve to level off at `severinghaus(pvo2)`, which may not match
reality. This likely causes pvo2 to hit its lower bound.

**Time-based Bohr can't represent hypocapnic starts.** After hyperventilation (common
before FL holds), PaCO2 is low (~25-30 mmHg) and P50 starts *below* 26.6 (left-shifted).
Our model can only shift P50 rightward from baseline. The real trajectory is:
P50: ~24 -> 26.6 -> 30+ (left -> baseline -> right-shift). This creates systematic
error early in FL holds that other parameters must absorb.

**The saturating exponential for CO2 is less physiological than linear.** CO2 production
is roughly constant (~200 mL/min), buffered with roughly constant capacitance. PaCO2
rises at ~3-5 mmHg/min. It doesn't saturate during typical apnea durations (3-7 min).

**The virtual PO2 Bohr shift is an approximation, not an equivalence.** For Severinghaus,
the hardcoded coefficients (23400, 150) were calibrated for P50=26.6. Scaling the input
PO2 linearly is only exact at the P50 point itself; at the tails it diverges from a
true P50-shifted curve. The error is small for modest shifts (5-10 mmHg).

### 5.3 Missing Measurement Model

**No sensor dynamics.** Finger pulse oximeters use moving-average filters (typically
5-15 seconds) that smooth and delay the signal. We model none of this — r_offset is
just a constant bias. All sensor dynamics are absorbed into physiology parameters
(tau_washout, gamma), contaminating their interpretation.

**HR data is available but unused.** We have per-second heart rate from the same device,
stored in the database. Heart rate changes during apnea (dive reflex bradycardia)
directly affect lung-to-finger transit time and sensor behavior. This is wasted
information.

### 5.4 Objective Function Bias

**Unweighted SSE biases toward the plateau.** A 372-point FL hold has ~250 points in
the flat plateau (SpO2 96-99%) and ~120 in the steep drop. The plateau contributes 2x
more to the objective, so the optimizer prioritizes getting those uninteresting flat
points right over nailing the clinically important desaturation curve.

---

## 6. External Review Feedback

The model was reviewed by Gemini 3 Pro and GPT-5.2-pro. Below is a synthesis of their
feedback with our assessment of each point.

### Agreed (will address)

| Suggestion                           | Source     | Our Assessment                      |
|--------------------------------------|------------|-------------------------------------|
| Weighted/region-based loss function  | Both       | Clear win, easy to implement        |
| Decouple steepness from Bohr shift   | Both       | GPT's logit-slope approach is best  |
| Add sensor low-pass filter           | Both       | Beat-based (using HR) is even better|
| Replace time-Bohr with CO2-based     | GPT        | Handles hypocapnic starts, more interpretable |
| Make pvo2 time-varying               | Both + us  | Fixed floor is wrong                |
| Shunt/mixing for RV/FRC             | GPT        | Defer to multi-type fitting phase   |
| Descriptive sigmoid as benchmark     | GPT        | Good sanity check                   |

### Rejected or deferred

| Suggestion                          | Source  | Why we disagree/defer                     |
|-------------------------------------|---------|-------------------------------------------|
| Lag with Bayesian prior             | Gemini  | Doesn't solve the fundamental confounding; beat-based delay is better |
| Kelman equations for Bohr           | Gemini  | We tested Kelman — it lost to Severinghaus. Also requires PCO2/pH/temp inputs we don't have |
| Linear PAO2 trajectory              | GPT     | Not clearly more physiological than exponential (see Section 7.5); worth testing as ablation |
| Full mass-balance ODE               | Gemini  | Over-engineering for our data; gives linear decline which may be worse |
| Gradient matching in loss           | Gemini  | Numerically noisy on 1 Hz data; needs smoothing which introduces bias |

---

## 7. Planned v2 Model Architecture

### 7.1 Design Principles

1. Every parameter should be identifiable from the data
2. Every parameter should map to a distinct physiological or measurement quantity
3. No parameter should be confounded with another
4. Physiology parameters should not absorb sensor dynamics

### 7.2 v2 Architecture

```
Input: t (seconds), HR(t) (beats/min from pulse oximeter)
                    |
                    v
            +-------------------------------+
        1.  |  Alveolar O2 Depletion        |  PAO2(t) = pvo2 + (pao2_0 - pvo2) * e^(-t/tau)
            |  [keep exponential; test       |  [or linear: PAO2(t) = pao2_0 - k_O2 * t]
            |   linear as ablation]          |
            +---------------+---------------+
                            v
            +-------------------------------+
        2.  |  CO2-Driven Bohr Effect        |  PaCO2(t) = paco2_0 + k_co2 * t
            |  [replaces time-based]         |  P50_eff = 26.6 + 0.48 * (PaCO2 - 40)
            +---------------+---------------+  [beta=0.48 fixed from literature]
                            v
            +-------------------------------+
        3.  |  Severinghaus ODC              |  S_base = Sev(PAO2 * 26.6 / P50_eff)
            |  + Logit-slope steepness       |  S_true = 100 * sigma(kappa * logit(S_base / 100))
            |  [replaces gamma power]        |  [kappa decoupled from Bohr]
            +---------------+---------------+
                            v
            +-------------------------------+
        4.  |  Beat-Based Sensor Model       |  delay(t) = B_delay / (HR(t)/60)
            |  [uses HR data]                |  tau_sens(t) = B_avg / (HR(t)/60)
            +---------------+---------------+  [IIR low-pass on delayed signal]
                            v
                Output: SpO2_pred + r_offset
```

### 7.3 v2 Parameters

| Param      | Meaning                                       | Identifiable from               |
|------------|-----------------------------------------------|---------------------------------|
| `pao2_0`   | Initial alveolar PO2 (mmHg)                   | Plateau duration + drop onset   |
| `pvo2`     | PO2 floor (mmHg)                              | Late-hold SpO2 nadir            |
| `tau_washout` | O2 depletion time constant (s)             | Transition steepness in time    |
| `paco2_0`  | Initial PaCO2 (mmHg; low after hypervent.)    | Early-hold plateau shape        |
| `k_co2`    | CO2 rise rate (mmHg/s)                        | Late-hold acceleration          |
| `kappa`    | Logit steepness in saturation space            | Drop sharpness (Bohr-independent) |
| `B_delay`  | Beat-based transport delay (beats)             | Phase shift that varies with HR |
| `B_avg`    | Beat-based sensor averaging (beats)            | Smoothing that varies with HR   |
| `r_offset` | Calibration bias (%)                           | Constant vertical shift         |

**9 total fitted params**, but:
- `beta = 0.48` is fixed (replaces free `bohr_max`)
- `tau_bohr` is eliminated entirely (CO2 rise is linear, not saturating)
- `B_delay` and `B_avg` could be fixed initially from recovery data -> 7 free params

**Key improvement:** Beat-based sensor parameters have HR-dependent signatures that
physiology parameters cannot mimic. During apnea, HR drops from ~60 to ~40 bpm. If
transit is ~12 heartbeats, delay stretches from 12s to 18s — a pattern that tau_washout
cannot reproduce. This breaks the confounding that made the old lag parameter
non-identifiable.

### 7.4 Why Logit-Slope is Better Than Gamma Power

Current gamma: `PO2_adj = P50 * (PO2_virtual / P50)^gamma`
- Operates in PO2 space, before the ODC
- Entangled with Bohr: gamma exponentiates the Bohr-shifted value
- "Power-law warping of the PO2 axis" — hard to map to Hb biochemistry

Proposed kappa: `S_true = 100 * sigma(kappa * logit(S_base / 100))`
- Operates in saturation space, after the ODC
- Completely decoupled from Bohr (applied to the final saturation value)
- Preserves the 50% crossing: logit(0.5) = 0, so kappa has no effect at P50
- kappa > 1 steepens (higher plateau, deeper nadir), kappa < 1 flattens
- More interpretable as "effective cooperativity" without contamination

### 7.5 Why Keep Exponential PAO2 (For Now)

GPT-5.2-pro argues for linear PAO2 based on constant-VO2 mass balance. However:

1. **Constant VO2 gives constant O2 content depletion, not constant PO2 depletion.**
   On the flat ODC shoulder (PAO2 > 80), a given VO2 causes a large PO2 drop but
   small content change. On the steep part, the same VO2 causes a small PO2 drop
   but large content change. So even with constant VO2, the PAO2 trajectory is
   nonlinear — fast PO2 drop early, slowing later.

2. **The exponential captures this qualitative shape** (fast early, slowing later)
   for roughly the right reasons.

3. **Linear decline has no natural floor** — it goes negative eventually, requiring
   arbitrary clamping.

4. **The real trajectory is an ODE coupled to the ODC,** which is neither linear
   nor exponential. Both are approximations.

Testing linear as an ablation is cheap and we should do it. But we expect the
exponential to win or tie.

### 7.6 Why CO2-Based Bohr is Better Than Time-Based

Current: `P50_eff = 26.6 + bohr_max * (1 - exp(-t / tau_bohr))`
- Detached from the actual driver (CO2/pH)
- Can only shift P50 rightward from 26.6
- Acts as an "extra sigmoid-maker" whether or not CO2 behaves that way
- Saturating exponential is less physical — CO2 rise is roughly linear

Proposed: `P50_eff = 26.6 + 0.48 * (PaCO2(t) - 40)`  where `PaCO2(t) = paco2_0 + k_co2 * t`
- Grounded in the actual Bohr mechanism (CO2 -> pH -> P50)
- `paco2_0` can be < 40 (hyperventilation -> left-shifted P50 start)
- Linear CO2 rise is well-supported: ~3-5 mmHg/min during apnea
- `beta = 0.48` mmHg P50 shift per mmHg CO2 from literature
- Eliminates `tau_bohr` (no saturation needed for typical apnea durations)

### 7.7 Beat-Based Sensor Model Details

Finger pulse oximeters introduce two measurement distortions:

1. **Transport delay:** Blood travels from lungs to finger. Duration depends on
   cardiac output, which varies with HR during apnea.

2. **Averaging window:** Device applies a moving-average filter (5-15s) to reduce
   motion artifacts. Effective window widens when perfusion drops (low HR).

Both are better expressed in heartbeats than seconds:

```python
# Beat-based delay
delay_s = B_delay / (hr / 60.0)  # seconds, stretches as HR drops
sao2_delayed = interp(t - delay_s, t, sao2_true)

# Beat-based IIR low-pass filter
tau_sensor = B_avg / (hr / 60.0)  # seconds, stretches as HR drops
alpha = dt / (tau_sensor + dt)
spo2_pred[i] = spo2_pred[i-1] + alpha * (sao2_delayed[i] - spo2_pred[i-1])
```

**Typical values:** B_delay ~10-15 beats, B_avg ~5-12 beats.

**Identification strategy:** If recovery data (post-hold breathing) is available, the
sharp SpO2 rise constrains B_delay and B_avg with less confounding than during the hold.
Fix these from recovery, then fit physiology parameters with sensor dynamics locked.

---

## 8. Implementation Plan

Same proven workflow as v1.2: comparison script first, then integrate the winner.

### Phase 1: Baseline + Benchmark (comparison script)
- Current v1.2 model (7 params) as baseline
- Descriptive sigmoid (4 params) as sanity-check floor
- Current model with weighted loss (no structural change)
- **Metric:** R^2 overall, R^2 in drop region (SpO2 < 95%), params at bounds, RMSE

### Phase 2: Incremental Ablations (each vs baseline)
- **A:** Replace gamma with logit-slope kappa (same param count)
- **B:** Replace time-Bohr with CO2-Bohr (same param count)
- **C:** Add fixed sensor LPF (tau_sensor=8s, no new fitted params)
- **D:** Add beat-based sensor model using HR data (1-2 new params)
- **E:** Linear PAO2 vs exponential

### Phase 3: Combine Winners
- Take whichever ablations improved R^2 in the drop region
- Test full v2 combination
- Compare against descriptive sigmoid

### Phase 4: Stress Tests
- Bootstrap parameter stability (are correlations reduced?)
- Cross-hold validation (fit on N-1 holds, predict Nth)
- Multi-seed fitting (are results consistent?)

### Phase 5: Integrate into App
- Update hill_model.py, fitter.py, db_models.py, schemas.py, routers, model_manager
- DB migration for new param columns
- Update frontend labels, charts, descriptions, About page
- Update tests

### Future (post-v2)
- **Shunt fraction for RV/FRC:** `S_arterial = (1-s) * S_endcap + s * S_venous`
- **Time-varying pvo2:** Declining floor as tissue extraction continues
- **HR-scaled VO2:** `k_O2(t) = k_inf + (k_0 - k_inf) * e^(-t/tau_reflex)`
- **Cross-type shared parameters:** kappa, CO2 beta, B_delay shared across hold types;
  only pao2_0, tau_washout, paco2_0, shunt vary by type
- **Bayesian fitting:** Posterior distributions on parameters to quantify uncertainty

---

## 9. Available Data

### From Pulse Oximeter CSV
- SpO2 (%) at 1 Hz
- HR (bpm) at 1 Hz
- Interval type labels (Rest, Apnea, Cooldown)
- Session date and round structure

### In Database
- `hold_data` table: elapsed_s, spo2, hr per data point per hold
- `holds` table: hold_type (FRC/RV/FL), duration, min_spo2, min_hr
- `model_versions` table: fitted params, R^2, per hold type
- `sessions` table: session metadata, CSV filename

### What We Don't Have
- End-tidal CO2 (ETCO2) — would directly constrain paco2_0 and k_co2
- Arterial blood gas — would give true SaO2, PaO2, PaCO2
- Multiple sensor sites — would constrain transport delay
- Perfusion index — would flag low-quality readings
- Core temperature — affects P50 via temperature Bohr effect

---

## 10. References

- Severinghaus, J.W. (1979). Simple, accurate equations for human blood O2 dissociation
  computations. J Appl Physiol. 46(3):599-602.
- Kelman, G.R. (1966). Digital computer subroutine for the conversion of oxygen tension
  into saturation. J Appl Physiol. 21(4):1375-1376.
- Hill, A.V. (1910). The possible effects of the aggregation of the molecules of
  haemoglobin on its dissociation curves. J Physiol. 40:iv-vii.
- Storn, R. & Price, K. (1997). Differential Evolution — A Simple and Efficient
  Heuristic for Global Optimization over Continuous Spaces. J Global Optimization. 11:341-359.
