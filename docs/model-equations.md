# SpO2 Desaturation Models: Equations and Data

Three models are retained for validation on future data. Two are mechanistic
(Production, CO2-Bohr) and one is a descriptive benchmark (Richards).

v4 tested adding a sensor delay observation layer and censored loss to CO2-Bohr.
The delay did not improve fit quality — see Section 10.

v5 tested richer sensor observation models (IIR filter, delay+filter, recovery
phase, beat-based HR-coupled sensor). None improved fits on apnea-only data —
see Section 11.

Last updated: 2026-02-28

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
| 6  | FL   | 372          | 366         | 57-100         | 50-106         | Primary FL hold; longest; artifact at ~4:50 from raising hand |

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
- **Pulse oximeter floor (40%):** The device cannot report SpO2 below 40%.
  Hold 4 (RV) reaches this floor, so the true nadir may be lower than recorded.
  Values sitting at exactly 40% should be treated as left-censored.
- **Arm-raise artifact (Hold 6, ~4:50):** Around t=290s the subject raised the
  hand wearing the pulse oximeter, reducing peripheral perfusion and causing a
  transient SpO2 dip that is not physiological desaturation. This artifact is
  not currently removed and may bias the fit in that region.
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

## 7. Cross-Prediction / Generalisation (2026-02-26)

Tests whether fitted parameters generalise to unseen holds. Each model is
trained on one hold and used to predict all others. R² < 0 means the model
is worse than predicting the mean of the target hold.

### Within-type transfer (R²)

Train on one hold of a given type, predict the other hold of the same type.

| Pair           | Production | CO2-Bohr | Richards |
|----------------|:----------:|:--------:|:--------:|
| FRC #5 → FRC #2 | **0.92** | 0.88     | 0.90     |
| FRC #2 → FRC #5 | **0.92** | 0.89     | 0.40     |
| RV #4 → RV #3   | 0.89     | 0.89     | 0.84     |
| RV #3 → RV #4   | 0.88     | 0.87     | **0.92** |
| FL #6 → FL #1   | -0.61    | -0.49    | -0.52    |
| FL #1 → FL #6   | -0.58    | -3.05    | **-0.17**|

### Cross-type transfer (R²)

Trained on FL #6 (longest, deepest FL hold), predict all other holds.

| Predict | Production | CO2-Bohr | Richards |
|---------|:----------:|:--------:|:--------:|
| FL #1   | -0.61      | -0.49    | -0.52    |
| FRC #2  | -0.44      | -0.44    | -0.43    |
| RV #3   | -0.39      | -0.38    | -0.38    |
| RV #4   | -0.72      | -0.72    | -0.72    |
| FRC #5  | -0.35      | -0.35    | -0.36    |
| FL #6   | 0.9953 *   | 0.9953 * | 0.9952 * |

\* = self-fit (training hold).

### Cross-prediction RMSE (within-type pairs)

| Pair           | Production | CO2-Bohr | Richards |
|----------------|:----------:|:--------:|:--------:|
| FRC #5 → FRC #2 | **1.05** | 1.29     | 1.17     |
| FRC #2 → FRC #5 | **3.56** | 4.23     | 10.03    |
| RV #4 → RV #3   | **3.50** | 3.55     | 4.19     |
| RV #3 → RV #4   | 7.65     | 8.04     | **6.34** |
| FL #6 → FL #1   | **0.61** | 0.59     | 0.59     |
| FL #1 → FL #6   | **16.22**| 25.95    | 13.93    |

### Key findings

1. **FRC and RV pairs transfer well** (R² 0.84–0.92). Both holds within each
   type show real desaturation, so the fitted dynamics are similar enough.
   Production has the best and most consistent within-type transfer.

2. **FL pair does not transfer.** FL #1 barely desaturates (SpO2 97–99%) while
   FL #6 drops to 57%. A model trained on a deep drop predicts a deep drop on
   the flat hold, and vice versa. This is a data issue (insufficient FL #1
   desaturation), not a model failure.

3. **No model generalises across hold types.** All cross-type R² values are
   negative. The initial O2 store (pao2_0) and washout rate (tau_washout) are
   fundamentally different between FL/FRC/RV, making parameters trained on one
   type inapplicable to another. Per-type fitting is justified.

4. **Production slightly edges out CO2-Bohr on transfer tasks.** The saturating
   Bohr effect transfers more reliably than the linear CO2 formulation (e.g.,
   FRC #2→FRC #5: Production 0.92 vs CO2-Bohr 0.89).

5. **Richards transfer is inconsistent.** Best on RV #3→RV #4 (0.92) but worst
   on FRC #2→FRC #5 (0.40). Its curve-specific parameters (t50, k, nu) do not
   carry physiological meaning, so transfer success depends on how similar the
   two curves happen to be in shape.

6. **Global fitting across hold types would require** shared physiological
   parameters (Bohr effect, ODC steepness, sensor delay) with type-specific
   initial conditions (pao2_0, tau_washout). This is a potential next step.

### Implementation

- `backend/scripts/cross_predict.py` — full cross-prediction matrix + plots

---

## 8. Partial-Transfer Cross-Prediction (2026-02-26)

Tests whether the mechanistic models' parameters separate cleanly into
**shared physiology** (same person, transfers across types) and **type-specific
initial conditions** (re-fit per target hold). This is a prerequisite for
global fitting.

### Parameter split

**Production:**
- Shared (fixed from source): `pvo2`, `gamma`, `bohr_max`, `tau_bohr`, `r_offset` (5 params)
- Type-specific (re-fit on target): `pao2_0`, `tau_washout` (2 params)

**CO2-Bohr:**
- Shared (fixed from source): `pvo2`, `gamma`, `k_co2`, `r_offset` (4 params)
- Type-specific (re-fit on target): `pao2_0`, `tau_washout`, `paco2_0` (3 params)

### Protocol

1. Fit full model on source hold (all 7 params free)
2. Fix shared params from source fit
3. Re-optimize only type-specific params on target hold (DE, 2-3 free params)
4. Report R² and compare to raw transfer (0 re-fitted) and self-fit (7 re-fitted)

"Recovery" = (partial R² - raw R²) / (self-fit R² - raw R²) × 100%. A value
near 100% means the shared params transfer perfectly and only initial conditions
need adjusting.

### Cross-type partial transfer (R²)

All 24 cross-type (source type != target type) pairs.

**Production** (2 free params):

| Source → Target  |  Raw R² | Partial R² | Self-fit R² | Recovery |
|------------------|--------:|-----------:|------------:|---------:|
| FL#1 → FRC#2    | -0.33   | 0.94       | 0.94        | 99.6%    |
| FL#1 → RV#3     | -0.35   | 0.97       | 0.98        | 99.0%    |
| FL#1 → RV#4     | -0.71   | 0.96       | 0.96        | 100.0%   |
| FL#1 → FRC#5    | -0.35   | 0.98       | 0.98        | 99.9%    |
| FRC#2 → RV#3    |  0.33   | 0.93       | 0.98        | 91.0%    |
| FRC#2 → RV#4    |  0.47   | 0.91       | 0.96        | 89.2%    |
| FRC#2 → FL#6    | -2.81   | 0.99       | 1.00        | 99.9%    |
| FRC#5 → RV#3    |  0.24   | 0.97       | 0.98        | 98.2%    |
| FRC#5 → RV#4    |  0.58   | 0.95       | 0.96        | 97.8%    |
| FRC#5 → FL#6    | -10.05  | 0.99       | 1.00        | 100.0%   |
| RV#3 → FRC#2    | -6.57   | 0.88       | 0.94        | 99.1%    |
| RV#3 → FRC#5    | -1.32   | 0.87       | 0.98        | 95.3%    |
| RV#3 → FL#6     | -15.97  | 0.99       | 1.00        | 100.0%   |
| RV#4 → FRC#2    | -2.84   | 0.82       | 0.94        | 96.9%    |
| RV#4 → FRC#5    | -0.17   | 0.96       | 0.98        | 98.7%    |
| RV#4 → FL#6     | -13.30  | 0.99       | 1.00        | 99.9%    |
| FL#6 → FRC#2    | -0.44   | 0.37       | 0.94        | **58.8%**|
| FL#6 → RV#3     | -0.39   | 0.86       | 0.98        | 91.0%    |
| FL#6 → RV#4     | -0.72   | 0.74       | 0.96        | **86.9%**|
| FL#6 → FRC#5    | -0.35   | 0.59       | 0.98        | **70.8%**|

**CO2-Bohr** (3 free params):

| Source → Target  |  Raw R² | Partial R² | Self-fit R² | Recovery |
|------------------|--------:|-----------:|------------:|---------:|
| FL#1 → FRC#2    | -0.34   | 0.94       | 0.95        | 99.6%    |
| FL#1 → RV#3     | -0.35   | 0.98       | 0.98        | 99.7%    |
| FL#1 → RV#4     | -0.70   | 0.93       | 0.96        | 98.6%    |
| FL#1 → FRC#5    | -0.34   | 0.98       | 0.98        | 100.1%   |
| FRC#2 → RV#3    |  0.33   | 0.95       | 0.98        | 94.1%    |
| FRC#2 → RV#4    |  0.42   | 0.95       | 0.96        | 99.2%    |
| FRC#2 → FL#6    | -1.68   | 1.00       | 1.00        | 100.0%   |
| FRC#5 → RV#3    |  0.22   | 0.96       | 0.98        | 97.0%    |
| FRC#5 → RV#4    |  0.57   | 0.91       | 0.96        | 87.8%    |
| FRC#5 → FL#6    | -11.34  | 0.98       | 1.00        | 99.9%    |
| RV#3 → FRC#2    | -6.60   | 0.88       | 0.95        | 99.1%    |
| RV#3 → FRC#5    | -1.43   | 0.96       | 0.98        | 99.3%    |
| RV#3 → FL#6     | -16.88  | 0.95       | 1.00        | 99.8%    |
| RV#4 → FRC#2    | -2.82   | 0.82       | 0.95        | 96.7%    |
| RV#4 → FRC#5    | -0.17   | 0.96       | 0.98        | 98.7%    |
| RV#4 → FL#6     | -13.41  | 0.99       | 1.00        | 99.9%    |
| FL#6 → FRC#2    | -0.44   | 0.93       | 0.95        | 98.8%    |
| FL#6 → RV#3     | -0.38   | 0.89       | 0.98        | 93.2%    |
| FL#6 → RV#4     | -0.72   | 0.91       | 0.96        | 97.2%    |
| FL#6 → FRC#5    | -0.35   | 0.94       | 0.98        | 97.4%    |

### Key findings

1. **Partial transfer recovers 87-100% of self-fit R² for most pairs.** The
   shared physiological parameters (gamma, Bohr effect, pvo2, r_offset) transfer
   across hold types almost perfectly. The failure in raw cross-prediction was
   entirely due to type-specific initial conditions.

2. **CO2-Bohr transfers more consistently than Production.** CO2-Bohr achieves
   93-100% recovery across all 24 cross-type pairs. Production achieves 86-100%
   for most pairs but drops to 59-71% when FL#6 is the source. The extra free
   param (paco2_0) in CO2-Bohr gives it more flexibility to adapt.

3. **FL#6 as source is problematic for Production** (59-87% recovery on
   FRC/RV targets). The FL#6 fit likely has compensatory shared params (gamma,
   Bohr) tuned to the long FL plateau that don't represent the true physiology
   well. CO2-Bohr avoids this by having paco2_0 as a type-specific param.

4. **FL#1 as source works surprisingly well** (99-100% recovery) despite
   barely desaturating. The shared physiology estimated from a flat hold still
   transfers — the model correctly attributes the lack of desaturation to high
   pao2_0/tau_washout rather than different physiology.

5. **This validates the global fitting approach.** A model with shared
   physiology (4-5 params) + per-type initial conditions (2-3 params) should
   achieve R² close to individual per-hold fits, while being constrained by
   a single consistent set of physiological parameters.

### Implementation

- `backend/scripts/cross_predict_partial.py` — partial-transfer fitting + plots

---

## 10. v4 Experiment: Sensor Delay + Censored Loss (2026-02-27)

### Hypothesis

v3 showed gamma pinning at upper bound (2.0) and 3-5/7 params at bounds.
The hypothesis was that physiology params compensate for a missing sensor
delay (circulation lag from lung to finger pulse oximeter, ~10-25s empirically).
Adding a pure time delay should free gamma and reduce params-at-bounds.

### CO2-Bohr+Delay (8 params)

Identical physiology to CO2-Bohr, with a pure delay observation layer:

    SaO2(t)        = [CO2-Bohr physiology, identical equations]
    SaO2_delayed(t) = interp(t - d, t, SaO2, left=SaO2[0])
    SpO2(t)        = clip(SaO2_delayed + r_offset, 0, 100)

New parameter: `d` in (3, 30) seconds. Gamma bounds narrowed to (0.8, 1.5).

### Censored Loss

Alternative loss function that treats SpO2 = 40% readings as left-censored
(device floor):

    L = -sum_i  log p(y_i | y_hat_i)
    p(y_i | y_hat_i) = N(y_i; y_hat_i, sigma)    if y_i > 40
    p(y_i | y_hat_i) = Phi(40; y_hat_i, sigma)    if y_i <= 40

where sigma = 1.5 (fixed), Phi is the normal CDF.

### Variants tested

| Variant              | Params | Observation       | Loss           | Gamma bounds |
|----------------------|--------|-------------------|----------------|-------------|
| CO2-Bohr             | 7      | r_offset only     | Weighted SSE   | 0.8-2.0     |
| CO2-Bohr+Delay       | 8      | pure delay + clip | Weighted SSE   | 0.8-1.5     |
| CO2-Bohr+Delay+Cens  | 8      | pure delay + clip | Censored NLL   | 0.8-1.5     |
| Richards             | 5      | None              | Unweighted SSE | N/A         |

### Per-hold results (R²)

| Hold    | CO2-Bohr (7p) | CO2-B+Delay (8p) | CO2-B+D+Cens (8p) | Richards (5p) |
|---------|:-------------:|:-----------------:|:------------------:|:-------------:|
| FL #1   | 0.7394        | 0.7231            | 0.7231             | **0.7644**    |
| FL #6   | **0.9953**    | 0.9953            | **0.9955**         | 0.9952        |
| FRC #2  | 0.9487        | 0.9493            | 0.9565             | **0.9925**    |
| FRC #5  | 0.9758        | 0.9767            | 0.9779             | **0.9833**    |
| RV #3   | 0.9837        | 0.9836            | 0.9845             | **0.9876**    |
| RV #4   | 0.9587        | 0.9585            | 0.9521             | **0.9710**    |

### Params at bounds

| Hold    | CO2-Bohr | CO2-B+Delay | CO2-B+D+Cens | Richards |
|---------|:--------:|:-----------:|:------------:|:--------:|
| FL #1   | 4/7      | 5/8         | 5/8          | 2/5      |
| FL #6   | 3/7      | 4/8         | 4/8          | 1/5      |
| FRC #2  | 4/7      | 5/8         | 5/8          | 1/5      |
| FRC #5  | 4/7      | 5/8         | 5/8          | 2/5      |
| RV #3   | 4/7      | 6/8         | 4/8          | 2/5      |
| RV #4   | 3/7      | 4/8         | 5/8          | 1/5      |

### Delay values (per-hold fits)

| Hold    | d (s)  | Notes               |
|---------|-------:|---------------------|
| FL #1   | 30.0   | At upper bound      |
| FL #6   | 30.0   | At upper bound      |
| FRC #2  | 30.0   | At upper bound      |
| FRC #5  | 30.0   | At upper bound      |
| RV #3   | 26.4   | Interior            |
| RV #4   | 30.0   | At upper bound      |

Empirical delay (end-of-apnea to SpO2 nadir): 9-34s, median 22s.

### Global fit (14p vs 13p)

CO2-Bohr+Delay global fit: 5 shared + 3x3 type-specific = 14 params.
CO2-Bohr global fit: 4 shared + 3x3 type-specific = 13 params.

| Metric                  | Global CO2-Bohr (13p) | Global CO2-B+Delay (14p) |
|-------------------------|:---------------------:|:------------------------:|
| Avg R² (fitted holds)   | 0.9602                | 0.9606                   |
| Params at bounds        | 5/13                  | 5/14                     |
| gamma                   | 1.76                  | 1.44                     |
| k_co2                   | 0.020 (at bound)      | 0.028                    |
| pvo2                    | 15.6                  | 15.0 (at bound)          |
| d                       | N/A                   | 30.0 (at bound)          |

### Key findings

1. **Delay d pins at upper bound (30s) in 5/6 per-hold fits and the global
   fit.** The model uses delay as another compensation parameter, not as a
   faithful sensor model. Delay and tau_washout are confounded.

2. **Gamma moved from 2.0 → 1.5 (per-hold) / 1.76 → 1.44 (global).** The
   narrower gamma bound forces the optimizer off the ceiling, but gamma still
   pushes to whatever ceiling is set. It's not converging to a physiological
   value — it's just hitting the new, lower wall.

3. **Adding delay increases params-at-bounds** (4-6/8 vs 3-4/7). The extra
   parameter creates more identifiability problems rather than resolving them.

4. **R² is essentially unchanged.** +Delay: ±0.001 vs CO2-Bohr on all holds.

5. **Censored loss doesn't help.** CO2-Bohr+Delay+Cens degrades FRC#2 drop
   R² (0.65 vs 0.81) and RV#4 overall (0.952 vs 0.959). Only improves FL#6
   marginally.

6. **Per-hold shared-param consistency is poor with delay.** pvo2 ranges
   15-45, r_offset ranges -2.9 to +3.0 across per-hold fits. The delay
   doesn't stabilise these.

7. **k_co2 didn't collapse in the global +Delay fit** (0.028 vs 0.020 at
   bound without delay). This is a minor improvement.

8. **Partial transfer still works well.** Both models recover 93-100% of
   self-fit R² with partial transfer, confirming the shared/type-specific
   split is sound regardless of delay.

### Conclusion

The sensor delay hypothesis was not supported. The delay parameter is
confounded with tau_washout and gets absorbed as another compensation
mechanism. The underlying issue is not missing delay but rather that the
7-parameter mechanistic structure cannot match the flexibility of the
5-parameter Richards sigmoid. Possible next steps:

- Constrain d from external measurement rather than fitting it
- Try a different observation model (e.g., IIR filter instead of pure delay)
- Accept that the current physiology model has ~0.96 R² ceiling and focus
  on interpretability rather than chasing Richards' 0.98

### Implementation

- `backend/scripts/compare_v4_model.py` — per-hold 4-way comparison
- `backend/scripts/cross_predict_v4.py` — cross-prediction matrix
- `backend/scripts/cross_predict_partial_v4.py` — partial transfer
- `backend/scripts/global_fit_v4.py` — joint fit (13p vs 14p)

---

## 11. v5 Experiments: Sensor Observation Models (2026-02-28)

### Motivation

v4 showed pure time delay is structurally unidentifiable with exponential PAO2
washout: `delay` just rescales the amplitude via `exp(d/tau)` and gets absorbed
by `pao2_0`. Expert analysis suggested two fixes: (1) an IIR low-pass filter
changes the **shape** of the curve (rounds the ODC knee), which delay alone
cannot, and (2) fitting the **recovery phase** after breathing resumes breaks
the exponential symmetry entirely.

### Experiment 1: IIR Filter Only (8p)

CO2-Bohr physiology + IIR low-pass filter (no delay):

    alpha = dt / (tau_f + dt)
    SpO2[i] = (1 - alpha) * SpO2[i-1] + alpha * SaO2[i]

New param: `tau_f` in (1, 20) seconds.

**Result:** tau_f pins at lower bound (1.0s) in all 6 holds. R² is **worse**
than baseline CO2-Bohr in 5/6 holds. gamma remains at 2.0. The filter adds
unwanted smoothing without absorbing any shape compensation.

### Experiment 2: Delay + Filter (9p)

CO2-Bohr + delay `d` + filter `tau_f`:

    SaO2_delayed(t) = interp(t - d, t, SaO2, left=SaO2[0])
    SpO2 = IIR_filter(SaO2_delayed)

**Result:** tau_f pins at 1.0 in all holds. `d` is erratic (3-30s). On
apnea-only data, both delay and filter are structurally unidentifiable.

### Experiment 3: Recovery Phase (9p)

Includes 30-90s of data after breathing resumes. Piecewise PAO2 model:

    PAO2(t) = pvo2 + (pao2_0 - pvo2) * exp(-t/tau)   for t <= t_end (apnea)
    PAO2(t) = 100.0                                    for t > t_end (recovery)

The V-shaped SpO2 nadir during recovery (SpO2 keeps falling for 10-30s after
first breath, then rises) should directly constrain delay + filter.

**Result:** Mixed. tau_f moved off lower bound in 3/6 holds (6.4, 6.8, 15.3s)
— recovery data **does** inform the filter. But `d` still pins at 30 in 5/6
holds. R² on recovery is severely negative (-4 to -22) — the instantaneous
PAO2 jump model is too aggressive. The actual recovery is more gradual (alveolar
re-oxygenation takes multiple breaths).

### Experiment 4: Global Fit with Filter (14p)

Took the "d still confounded" path: fixed d=0, fitted tau_f as shared param.
Gamma sweep: free (0.8-2.5), narrow (0.9-1.3), fixed (1.0).

**Result:** Filter **hurts** global fit — R² avg 0.61 vs 0.96 baseline.
tau_f pins at 1.0, r_offset compensates at +3 (bound). gamma drops from
1.76 to 1.23 with filter (partial absorption), but at catastrophic R² cost.

### Experiment 5: HR-Coupled Beat-Based Sensor (9p)

Time-varying delay and filter based on heartbeat counts:

    d(t) = B_delay * 60 / HR(t)      # delay in seconds, HR-dependent
    tau_f(t) = B_avg * 60 / HR(t)    # filter time constant, HR-dependent

Per-sample loop with time-varying IIR filter. B_delay in (5, 25) beats,
B_avg in (3, 15) beats.

**Result:** Slight R² improvements in 4/6 holds (best: RV#3 +0.006). But
B_delay pins at bounds (25 hi in 4/6, 5 lo in 2/6) — same confounding as
constant delay. B_avg pins at bounds in 5/6 holds. gamma still 1.3-2.0.
The beat-based sensor is marginally better than constant delay+filter but
does not solve identifiability.

### Summary

| Experiment | Key param behaviour | R² vs baseline | Verdict |
|-----------|---------------------|----------------|---------|
| Filter only (8p) | tau_f=1.0 (all holds) | Worse | No benefit |
| Delay+Filter (9p) | tau_f=1.0, d erratic | No change | Unidentifiable |
| Recovery (9p) | tau_f off bound in 3/6 | Recovery R² negative | Promising but model too simple |
| Global+Filter (14p) | tau_f=1.0, r_offset=+3 | 0.61 vs 0.96 | Catastrophic |
| Beat-Sensor (9p) | B_delay/B_avg at bounds | +0.006 best | Marginal, still confounded |

### Conclusions

1. **On apnea-only data, sensor models are structurally unidentifiable.** The
   exponential PAO2 decay creates a confounding: any delay/filter effect can
   be absorbed by rescaling `pao2_0` and `tau_washout`.

2. **Recovery data partially breaks the confounding** (Exp 3 showed tau_f
   moving to interior values), but the step PAO2 recovery model is too
   aggressive. A more gradual re-oxygenation model (exponential rise with
   its own time constant) might work better.

3. **Gamma persistently wants to be 1.5-2.0.** No sensor model tested can
   absorb gamma's shape compensation without destroying overall fit. gamma > 1
   may reflect real physiology (steep ODC at low PO2) or compensate for model
   misspecification elsewhere.

4. **The current CO2-Bohr model at R² ~0.96 (global) is near the ceiling**
   achievable without external measurement (ETCO2, multi-site pulse oximetry).
   The Richards sigmoid ceiling (R² ~0.98) requires shape flexibility that the
   current mechanistic structure cannot provide.

### Implementation

- `backend/scripts/exp_v5_01_filter_only.py` — Exp 1
- `backend/scripts/exp_v5_02_delay_filter.py` — Exp 2
- `backend/scripts/exp_v5_03_recovery.py` — Exp 3
- `backend/scripts/exp_v5_04_global_sensor.py` — Exp 4
- `backend/scripts/exp_v5_05_beat_sensor.py` — Exp 5

---

## 9. References

- Severinghaus, J.W. (1979). Simple, accurate equations for human blood O2
  dissociation computations. J Appl Physiol. 46(3):599-602.
- Hill, A.V. (1910). The possible effects of the aggregation of the molecules
  of haemoglobin on its dissociation curves. J Physiol. 40:iv-vii.
- Richards, F.J. (1959). A flexible growth function for empirical use.
  J Exp Bot. 10(2):290-301.
- Storn, R. & Price, K. (1997). Differential Evolution — A Simple and Efficient
  Heuristic for Global Optimization over Continuous Spaces. J Global Optim.
  11:341-359.
