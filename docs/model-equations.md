# SpO2 Desaturation Models: Equations and Data

Three models are retained for validation on future data. Two are mechanistic
(Production, CO2-Bohr) and one is a descriptive benchmark (Richards).

Last updated: 2026-02-26

---

## 1. Production Model (7 fitted parameters)

The model currently deployed in the application. Uses exponential alveolar O2
washout, a saturating Bohr effect, and the Severinghaus ODC with a gamma
steepness exponent.

### Equations

**Alveolar O2 decay** (exponential washout to venous floor):

    PAO2(t) = pvo2 + (pao2_0 - pvo2) * exp(-t / tau_washout)

**Bohr effect** (saturating exponential P50 shift):

    P50_eff(t) = 26.6 + bohr_max * (1 - exp(-t / tau_bohr))

**Oxygen-haemoglobin dissociation** (Severinghaus 1979 + virtual PO2 + gamma):

    PO2_virtual = PAO2(t) * 26.6 / P50_eff(t)
    PO2_adj     = 26.6 * (PO2_virtual / 26.6) ^ gamma
    SpO2(t)     = clip(r_offset + 100 / (1 + 23400 / (PO2_adj^3 + 150 * PO2_adj)), 0, 100)

### Parameters

| Parameter     | Unit   | Meaning                                     | FL bounds  | FRC bounds | RV bounds  |
|---------------|--------|---------------------------------------------|------------|------------|------------|
| `pao2_0`      | mmHg   | Initial alveolar PO2                        | 100-250    | 80-120     | 70-110     |
| `pvo2`        | mmHg   | Mixed venous PO2 floor                      | 20-50      | 20-50      | 20-50      |
| `tau_washout` | s      | O2 depletion time constant                  | 50-250     | 20-100     | 10-80      |
| `gamma`       | -      | ODC steepness (1.0 = standard Severinghaus) | 0.8-2.0    | 0.8-2.0    | 0.8-2.0    |
| `bohr_max`    | mmHg   | Maximum Bohr P50 shift                      | 2-15       | 2-15       | 2-15       |
| `tau_bohr`    | s      | CO2 saturation time constant                | 40-250     | 40-250     | 40-250     |
| `r_offset`    | %      | Sensor calibration bias                     | -3 to +3   | -3 to +3   | -3 to +3   |

**Fixed constant:** P50_BASE = 26.6 mmHg (standard haemoglobin half-saturation).

### Implementation

- `backend/app/services/hill_model.py` — `predict_spo2()`, `ApneaModelParams`
- `backend/app/services/fitter.py` — `fit_holds()`, `DEFAULT_BOUNDS`

---

## 2. CO2-Bohr Model (7 fitted parameters)

Identical to Production except the Bohr effect is driven by a linear CO2 rise
instead of a saturating exponential. This replaces `bohr_max` and `tau_bohr`
with `paco2_0` and `k_co2`.

### Equations

**Alveolar O2 decay** (identical to Production):

    PAO2(t) = pvo2 + (pao2_0 - pvo2) * exp(-t / tau_washout)

**Bohr effect** (linear CO2 accumulation drives P50):

    PaCO2(t) = paco2_0 + k_co2 * t
    P50_eff(t) = 26.6 + 0.48 * (PaCO2(t) - 40)

The coefficient 0.48 mmHg P50 shift per mmHg CO2 is fixed from literature, not
fitted. `paco2_0` can be below 40 mmHg after hyperventilation, giving an initial
left-shifted (lower) P50.

Expanding the P50 equation:

    P50_eff(t) = 26.6 + 0.48 * (paco2_0 - 40) + 0.48 * k_co2 * t

So P50 starts at `26.6 + 0.48*(paco2_0 - 40)` and rises linearly at rate
`0.48 * k_co2` mmHg/s. For a typical FL hold with paco2_0 = 30 (post-hyperventilation),
P50 starts at 21.8 mmHg (left-shifted) and rises through 26.6 as CO2 normalises.

**Oxygen-haemoglobin dissociation** (identical to Production):

    PO2_virtual = PAO2(t) * 26.6 / P50_eff(t)
    PO2_adj     = 26.6 * (PO2_virtual / 26.6) ^ gamma
    SpO2(t)     = clip(r_offset + 100 / (1 + 23400 / (PO2_adj^3 + 150 * PO2_adj)), 0, 100)

### Parameters

| Parameter     | Unit    | Meaning                                     | FL bounds  | FRC bounds | RV bounds  |
|---------------|---------|---------------------------------------------|------------|------------|------------|
| `pao2_0`      | mmHg    | Initial alveolar PO2                        | 100-250    | 80-120     | 70-110     |
| `pvo2`        | mmHg    | Mixed venous PO2 floor                      | 20-50      | 20-50      | 20-50      |
| `tau_washout` | s       | O2 depletion time constant                  | 50-250     | 20-100     | 10-80      |
| `gamma`       | -       | ODC steepness (1.0 = standard Severinghaus) | 0.8-2.0    | 0.8-2.0    | 0.8-2.0    |
| `paco2_0`     | mmHg    | Initial arterial CO2 partial pressure       | 25-45      | 30-45      | 35-50      |
| `k_co2`       | mmHg/s  | CO2 rise rate                               | 0.02-0.15  | 0.02-0.15  | 0.02-0.15  |
| `r_offset`    | %       | Sensor calibration bias                     | -3 to +3   | -3 to +3   | -3 to +3   |

**Fixed constants:** P50_BASE = 26.6 mmHg, beta = 0.48 mmHg P50 / mmHg CO2.

### Motivation vs Production

- The saturating exponential Bohr (`bohr_max * (1 - exp(-t/tau_bohr))`) is
  less physiological: CO2 production is roughly constant (~200 mL/min) and
  PaCO2 rises approximately linearly at 3-5 mmHg/min during apnea.
- The CO2-based formulation can represent hypocapnic starts (paco2_0 < 40 after
  hyperventilation), where P50 begins *below* baseline. The Production model
  can only shift P50 rightward from 26.6.
- In practice, both models produce nearly identical R^2 on the current dataset
  (differences in the 4th decimal place).

### Implementation

- `backend/scripts/compare_new_model.py` — `predict_baseline()`

---

## 3. Richards Sigmoid (5 fitted parameters)

A purely descriptive model with no physiological interpretation. Serves as a
benchmark ceiling: if a mechanistic model cannot match or beat a 5-parameter
sigmoid, it has a structural problem.

### Equation

    SpO2(t) = clip(s_min + (s_max - s_min) / (1 + nu * exp((t - t50) / k)) ^ (1/nu), 0, 100)

This is a generalised logistic (Richards curve) parameterised as a decreasing
sigmoid:

- As t -> 0:   SpO2 -> s_max  (plateau)
- As t -> inf:  SpO2 -> s_min  (nadir)
- t50:          Time at which SpO2 passes through the inflection region
- k:            Time scale of the sigmoid transition (larger = more gradual)
- nu:           Asymmetry parameter:
  - nu = 1: standard logistic (symmetric transition)
  - nu < 1: sharper onset of drop, gradual approach to nadir
  - nu > 1: gradual onset, sharper collapse to nadir

### Parameters

| Parameter | Unit | Meaning                    | FL bounds | FRC bounds | RV bounds |
|-----------|------|----------------------------|-----------|------------|-----------|
| `s_max`   | %    | Upper asymptote (plateau)  | 96-101    | 96-101     | 96-101    |
| `s_min`   | %    | Lower asymptote (nadir)    | 0-96      | 0-96       | 0-96      |
| `t50`     | s    | Inflection time            | 50-500    | 20-300     | 10-250    |
| `k`       | s    | Transition time scale      | 5-80      | 3-60       | 3-60      |
| `nu`      | -    | Asymmetry                  | 0.1-10    | 0.1-10     | 0.1-10    |

### Implementation

- `backend/scripts/compare_new_model.py` — `predict_richards()`

---

## 4. Loss Function

### Weighted SSE (Production and CO2-Bohr)

    L = sum_i  w_i * (y_i - y_hat_i)^2

    w_i = 3  if y_i < 95%
    w_i = 1  otherwise

The 3x weighting on the drop region compensates for the plateau containing
~60-70% of data points but carrying little shape information. Without weighting,
the optimizer over-fits the flat plateau at the expense of the clinically
important desaturation curve.

### Unweighted SSE (Richards)

    L = sum_i (y_i - y_hat_i)^2

Richards uses unweighted SSE since it has no physiological structure that could
be biased by the plateau/drop imbalance.

### Optimizer

All models are fitted with `scipy.optimize.differential_evolution`:
- Population size: 40
- Max iterations: 3000
- Tolerance: 1e-10
- Polish: True (L-BFGS-B refinement at convergence)
- Mutation: (0.5, 1.5)
- Recombination: 0.9
- Seed: 42 (reproducible)

---

## 5. Dataset

### Source

Single session recorded on 2026-02-21 using a finger pulse oximeter during a
freediving apnea training session ("Piramid" protocol). The CSV file
(`Saturday, February 21, 2026.csv`) contains per-second biometric readings
with interval type labels.

### Holds

| ID | Type | Duration (s) | Data points | SpO2 range (%) | HR range (bpm) | Notes |
|----|------|:------------:|:-----------:|:--------------:|:--------------:|-------|
| 1  | FL   | 179          | 178         | 97-99          | 61-102         | Barely desaturates; low signal |
| 2  | FRC  | 134          | 132         | 88-99          | 47-89          | Mild drop |
| 3  | RV   | 129          | 126         | 61-100         | 51-104         | Fast deep drop |
| 4  | RV   | 193          | 192         | 40-100         | 53-104         | Deepest; trimmed at t=194 (post-blackout recovery removed) |
| 5  | FRC  | 187          | 185         | 54-100         | 46-88          | Deep drop |
| 6  | FL   | 372          | 366         | 57-100         | 50-106         | Primary FL hold; longest |

**Hold types:**
- **FL (Full Lungs):** Maximum inspiration before hold. Largest O2 reserve,
  longest plateau before desaturation begins. pao2_0 is highest (100-250 mmHg
  due to hyperventilation).
- **FRC (Functional Residual Capacity):** Normal resting expiration level.
  Moderate O2 reserve, faster desaturation than FL.
- **RV (Residual Volume):** Maximum expiration before hold. Minimal O2 reserve,
  fastest desaturation. Lowest starting pao2_0 (70-110 mmHg).

### Measurements

- **SpO2 (%):** Peripheral oxygen saturation from finger pulse oximeter, 1 Hz.
  Integer values. Subject to transport delay (lung -> finger ~10-20s) and
  device averaging (5-15s moving window). Values at 40% may be censored
  (device floor).
- **HR (bpm):** Heart rate from the same device, 1 Hz. Integer values. Shows
  dive reflex bradycardia during holds (HR drops from ~60-100 to ~45-55 bpm).
  Currently stored but not used by any model.
- **Elapsed time (s):** Seconds since hold start, derived from interval
  timestamps in the CSV.

### Preprocessing

- **Ischaemic dip removal:** At the start of some holds, SpO2 briefly dips
  2-5% due to venous blood pooling before the real desaturation begins. The
  CSV parser detects and flattens these artifacts (see `csv_parser.py:
  _remove_ischaemic_dip`).
- **Post-blackout trimming:** Hold 4 (RV) was trimmed at t=194s where SpO2
  rises from 41% to 52% due to involuntary breathing after loss of
  consciousness. Only the desaturation portion is retained.
- **No smoothing** is applied to SpO2 or HR data. The models fit raw
  integer-valued readings.

### Data not available

- End-tidal CO2 (ETCO2) — would directly constrain paco2_0 and k_co2
- Arterial blood gas — would give true SaO2, PaO2, PaCO2
- Multiple sensor sites — would constrain transport delay
- Perfusion index — would flag low-quality oximeter readings
- Core temperature — affects P50 via temperature coefficient

### Storage

Data is stored in SQLite (`data/spo2.db`):
- `sessions` table: session metadata, CSV filename
- `holds` table: hold_type (FL/FRC/RV), duration, min_spo2, min_hr
- `hold_data` table: elapsed_s, spo2, hr per data point per hold
- `model_versions` table: fitted parameters, R^2, per hold type

---

## 6. Comparison Results (2026-02-26)

Fitted per-hold using differential evolution. All three models use the same
optimizer settings.

### R^2 (overall)

| Hold           | Production (7p) | CO2-Bohr (7p) | Richards (5p) |
|----------------|:---------------:|:-------------:|:-------------:|
| FL #1          | 0.7239          | 0.7394        | **0.7644**    |
| FL #6          | **0.9953**      | **0.9953**    | 0.9952        |
| FRC #2         | 0.9431          | 0.9487        | **0.9925**    |
| FRC #5         | 0.9760          | 0.9758        | **0.9833**    |
| RV #3          | 0.9838          | 0.9837        | **0.9876**    |
| RV #4          | 0.9589          | 0.9587        | **0.9710**    |

### R^2 (drop region, SpO2 < 95% only)

| Hold           | Production (7p) | CO2-Bohr (7p) | Richards (5p) |
|----------------|:---------------:|:-------------:|:-------------:|
| FL #1          | N/A             | N/A           | N/A           |
| FL #6          | **0.9902**      | **0.9901**    | 0.9889        |
| FRC #2         | 0.7789          | 0.7942        | **0.9543**    |
| FRC #5         | 0.9449          | **0.9505**    | 0.9618        |
| RV #3          | 0.9706          | **0.9710**    | 0.9571        |
| RV #4          | 0.9197          | 0.9198        | **0.9427**    |

### Params at bounds

| Hold           | Production | CO2-Bohr | Richards |
|----------------|:----------:|:--------:|:--------:|
| FL #1          | 4/7        | 4/7      | 2/5      |
| FL #6          | 2/7        | 3/7      | 1/5      |
| FRC #2         | 4/7        | 4/7      | 1/5      |
| FRC #5         | 5/7        | 4/7      | 2/5      |
| RV #3          | 3/7        | 4/7      | 2/5      |
| RV #4          | 3/7        | 3/7      | 1/5      |

### Key findings

1. **Production and CO2-Bohr are tied** on R^2 (differences in 4th decimal).
   The CO2-linear Bohr replacement does not improve fit quality on this dataset.
2. **Richards (5 params) beats both mechanistic models (7 params) on 5/6
   holds.** It also has consistently fewer parameters at bounds (1-2 vs 3-5).
3. **gamma is pinned at 2.0** (upper bound) on most holds for both mechanistic
   models, indicating the ODC steepness is under-parameterised or confounded
   with the Bohr effect.
4. **FL #1 is low-information** (2% total SpO2 variation). All models achieve
   R^2 ~ 0.72-0.76 — acceptable given minimal signal.
5. **RV #4 is the hardest hold** (deepest desaturation, nadir 40%). Richards
   achieves R^2 = 0.971 vs mechanistic 0.959. The gap is largest here.

---

## 7. References

- Severinghaus, J.W. (1979). Simple, accurate equations for human blood O2
  dissociation computations. J Appl Physiol. 46(3):599-602.
- Hill, A.V. (1910). The possible effects of the aggregation of the molecules
  of haemoglobin on its dissociation curves. J Physiol. 40:iv-vii.
- Richards, F.J. (1959). A flexible growth function for empirical use.
  J Exp Bot. 10(2):290-301.
- Storn, R. & Price, K. (1997). Differential Evolution — A Simple and Efficient
  Heuristic for Global Optimization over Continuous Spaces. J Global Optim.
  11:341-359.
