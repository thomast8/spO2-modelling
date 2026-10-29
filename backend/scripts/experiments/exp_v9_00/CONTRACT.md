# Nadir learning framework: Swift ⇄ Python contract

Status, 28 October 2026: filed under thomast8/nadir#697 (post-v1 intelligence). The Python side is
this experiment (exp_v9_00). The Swift side is parked, written but never compiled, on thomast8/nadir
branch `feature/learning-framework`; implementation is paused by decision. Sections changed on 28
October are marked (v3).

Source of truth for both implementations. The approved plan (revision 6, 22 October) set the design;
this file pins the exact definitions the plan leaves open so that the Swift core (Nadir `Sources/ApneaCore/Learning`)
and the Python harness (spo2-modelling `backend/scripts/experiments/exp_v9_00`) compute identical
numbers. Where this file and the plan disagree, this file wins and the difference is listed in §15.

Conventions: seconds are integers where the wire stores `TimeSpan` (whole seconds), everything
else `Double`. Logs are natural logs. "Trusted sample" = `VitalsSample.contactDetected == true`
with the named field non-nil. Lung-mode keys in JSON are the `HoldStyle` raw values `"full"`,
`"frc"`, `"rv"` (packed holds map to `"full"` everywhere in learning); Swift APIs use `LungMode`
(`.full`, `.frc`, `.rv`; note `LungMode.rv.rawValue == "empty"`, so never serialise LungMode raw
values into learning JSON). Sorting rows: by `(session.startedAt, session.id.rawValue, stepIndex)`.
Every public entry point first keeps sessions with `endedAt != nil` and deduplicates by `id`
(first occurrence after sorting by `(startedAt, id)` wins).

## 1. Wire keys (all optional, omitted when nil)

- `SessionRecording.oximeter: String?`: the roster entry id (`RememberedSensor.id.uuidString`) of the
  oximeter that delivered usable SpO₂.
- `RecordedStep.recoveryTrace: RecoveryTrace?` with
  `public struct RecoveryTrace: Hashable, Sendable, Codable { public var onsetSeconds: Int?; public var traceSeconds: Int; public var version: Int }`.
- `RecordedStep.styleSource: String?`, values `"imported" | "estimated" | "confirmed"`; nil = the
  plan recorded the style. Typed accessor `RecordedStep.styleSourceKind: StyleSource?` with
  `public enum StyleSource: String { case imported, estimated, confirmed }`.

## 2. Oximeter key

`Analytics.oximeterKey(of session: RecordedSession, roster: [RememberedSensor]) -> String?`:
1. `"stamina"` when `session.isImportedFromStamina`.
2. else `session.recording?.oximeter` when non-nil.
3. else rule B: `session.recording == nil`, `session.startedAt >= Learning.pm100PeriodStart`
   (2026-09-02T00:00:00Z), at least one trusted SpO₂ sample in the session, and the roster has
   exactly one entry with `kind == .oximeter` → that entry's `id.uuidString`; else nil.
Harness: the export has no roster; it passes a synthetic roster with one oximeter id
`"pm100"` (constant `HARNESS_PM100_KEY`) so rule B yields `"pm100"`.

## 3. Recovery trace (task 1 observation, version 1)

`Analytics.recoveryTrace(after holdIndex: Int, in session: RecordedSession) -> RecoveryTrace?`
1. nil unless `steps[holdIndex].isPerformedHold`.
2. Walk steps `holdIndex+1 ..< nextPerformedHold` (index of the next step with `isPerformedHold`
   after `holdIndex`, or `steps.endIndex`). `offset = 0`; for each step, for each trusted SpO₂
   sample append `(t: offset + sample.at.seconds, spo2)`; then `offset += step.actualDuration.seconds`.
3. nil when no readings.
4. `traceSeconds = readings.last.t` (readings are in walk order; do not sort).
5. `m = min(spo2)`; `k = last index where spo2 == m`; onset = first reading at index > k with
   `spo2 >= m + 2`; `onsetSeconds = that.t` or nil when none.
6. `version = 1`.
Censored (for display and counts): `onsetSeconds == nil && traceSeconds < 90`.
`RecordedSession.stampingRecoveryTraces() -> RecordedSession` fills `recoveryTrace` only where it is
nil and the hold or any following step up to the next hold has vitals; idempotent.
The fold reads `step.recoveryTrace ?? Analytics.recoveryTrace(after:in:)`.

## 4. Task 1: post-hold delay per oximeter

Rows: per finished session whose `oximeterKey == key`, `x_s = mean(onsetSeconds)` over its performed
holds with a non-nil onset (Double); sessions with no onset contribute nothing. `n` = number of such
sessions; `censored` = number of performed holds in the matching sessions with
`onset == nil && traceSeconds < 90`.
Normal-Inverse-Gamma with prior `(mu0, kappa0, alpha0, beta0)` = `Priors.postHoldDelay`:
`xbar = mean(x)`, `S = Σ(x − xbar)²`, `kn = k0 + n`, `mun = (k0·mu0 + n·xbar)/kn`,
`an = a0 + n/2`, `bn = b0 + S/2 + k0·n·(xbar − mu0)²/(2·kn)`.
Posterior on the mean: Student-t, df `2·an`, location `mun`, scale `sqrt(bn/(an·kn))`.
80 % interval: `mun ± t90(2·an)·scale`. Prior interval: same with `(mu0, k0, a0, b0)`, df `2·a0`.
`Estimate(lower, value: mun, upper, confidence: 0.8, basis: .personal(sessions: n, holds: Σ holds with onset), censored)`.
Narrowing `= 1 − (upper−lower)/(prior upper − prior lower)`.
`rowsToReach`: with `s2 = bn/an`, smallest integer `m ≥ 0` (search 0…200) such that
`2·t90(2·an + m)·sqrt(s2/(kn + m)) ≤ (1 − rhoEnough)·w_prior`; nil if none ≤ 200.
Gate key `"delay.<key>"`, unit `.sessionsWithOximeter(key:)`, floor/keep from `Priors.evidence.delay`.

## 5. Student-t 0.9 quantile

`Fit.t90(df: Double)`: table `Priors.t90` holds `scipy.stats.t.ppf(0.9, df)` for integer df 1…30;
for non-integer df interpolate linearly between floor and ceil; df > 30 → 1.2815515655446004
(normal 0.9 quantile); df < 1 → value at 1. Python uses the same table and interpolation, not scipy,
at runtime, so the two agree to the last bit.

## 6. Task 2 features and model

`HoldFacts` exactly as `HoldVolumeClassifier.facts(of:)` in the rule-based prototype on the local
Nadir branch `feat/stamina-hold-volume` (nadir#930): its `HoldFacts` and `facts(of:)` logic unchanged,
with every rule, threshold and `classify` dropped.
Feature vector per hold, in this order (index: name = definition):
0 `logPlanned = ln(max(planned, 1))`
1 `plannedShareOfMax = planned / max(planned over the session's holds)` (1 if max is 0)
2 `position = i / n` with `i` the hold's 0-based index among `facts(of:)` rows, `n` = row count
3 `isFirst` (i == 0), 4 `isLast` (i == n−1), as 0/1
5 `logRestBefore = ln(1 + restBefore)`
6 `logRatioPrevious = ln(max(planned,1)) − ln(max(previous planned,1))`, 0 for the first
7 `hasRVToken`, 8 `hasFullToken`, 9 `hasFRCToken`: planName lowercased, split on non-letters, whole
  tokens; RV ∈ {rv, empty, exhale}, full ∈ {fl, full, max}, FRC ∈ {frc}
10 `hasTrace = spo2Seconds >= 60 ? 1 : 0` (nil → 0)
11 `baseline`, 12 `fallBy120`, 13 `fallBy150`, 14 `minBy180`: value when `hasTrace == 1` and non-nil, else 0.
Standardise `z_j = (f_j − mean_j)/scale_j` (scale 0 → use 1), append intercept `z_15 = 1`.
Prior literal `Priors.holdVolume: HoldVolumePrior?` with `featureMeans[15]`, `featureScales[15]`,
`mean[16]`, `covariance[16][16]`, `lowerBand`, `upperBand`, `labelled: [String: Int]` (keys "full",
"rv", "frc"). nil literal = no adopted model.
Predictive `p(full) = σ(m / sqrt(1 + π·v/8))`, `m = meanᵀz`, `v = zᵀΣz`.
Verdict: `.full` if p ≥ upperBand, `.rv` if p ≤ lowerBand, else nil. FRC never emitted.
Labels for fitting (harness): Nadir-recorded (`!isImportedFromStamina`) holds with style full or packed → 1,
rv → 0, frc counted and excluded; imported holds only with `styleSource == "confirmed"` (the harness
treats the 6 August 2026 overrides as confirmed). Laplace fit: prior N(0, 1) on every weight
including the intercept, Newton/IRLS to convergence (|Δ| < 1e-10, ≤ 100 iterations), covariance = inverse
negative Hessian at the mode.

## 7. Hold markers (measurements; always shown)

`Analytics.holdMarkers(of session:) -> [HoldMarkers]`, one per performed hold, keyed by `stepID`
and `stepIndex`. Hold-local samples are the hold's own `vitals`; `t = sample.at.seconds`.
- `firstContractionSeconds = min(contractions).seconds`; nil if none.
- `firstContractionShare = first / actualDuration` (Double); nil if either missing or actual 0.
- `struggleSeconds = actual − first`; `contractionRatePerMin = count(contractions with t ≥ first) / (struggleSeconds/60)` when `struggleSeconds ≥ 10`, else nil.
- `firstUrgeSeconds = min(urges)`; nil if none.
- `spo2Baseline = max trusted SpO₂ with t ≤ 30`; nil if none.
- `spo2End = last trusted SpO₂ sample of the hold` (walk order).
- `spo2Low = HoldRecords.lowestSpO2(ofHoldsAt: [stepIndex], in: session)` (existing function; hold through to the next hold).
- `secondsAtOrAbove95 = count of trusted hold samples with spo2 ≥ 95`.
- (v3) `desaturationOnsetSeconds` and `desaturationRatePerMin` are Nadir's existing `HoldOxygenDrop.measure`
  (one owner, so the History chart and the session detail never disagree): the baseline is
  `RecordedSession.reoxygenationTarget(before:in:)` over the cleaned pre-hold steps (at least three trusted
  seconds); per-second readings are `VitalsSummary`'s oxygen mean; onset = the first of two consecutive
  seconds at or below baseline − 2, with no gap before confirmation; rate = points per minute from onset to
  the first hold low, only over four or more continuous seconds (positive = falling).
- `hrBaseline = median HR` of trusted HR samples in the 30 s before the hold: walk the steps before `stepIndex` backwards (stop at the previous performed hold); a sample at `at` in a step whose end is `E` seconds before the hold start lies `E + (duration − at)` seconds before; keep those ≤ 30. Median of an even count = mean of the two middle values. nil if none.
- `hrMin = min trusted HR in the hold`; `hrDropShare = (hrBaseline − hrMin)/hrBaseline`; nil if either missing or baseline ≤ 0.
- `secondsToHrDrop = first t` with HR ≤ `hrBaseline − 0.9·(hrBaseline − hrMin)`; nil when the drop is ≤ 0.
- `reoxygenationSeconds = session.timeToReoxygenation(afterHoldAt: stepIndex)?.seconds`.
- `recoveryOnsetSeconds = (step.recoveryTrace ?? recoveryTrace(after:in:))?.onsetSeconds`.

## 8. Diver axes

`Analytics.diverAxes(in sessions:, asOf: Date, halfLifeDays: Double?) -> [String: Gated<AxisValue>]`.
Holds: performed holds with `actualDuration ≥ 60` in finished sessions with `startedAt < asOf`,
excluding imported holds unless `styleSource == "confirmed"`. Weight `w = halfLife == nil ? 1 : 2^(−Δdays/h)`
with `Δdays = (asOf − session.startedAt)/86400`. Weighted median: sort values ascending (ties by
row order), smallest value whose cumulative weight ≥ half the total weight.
Axes (key: marker), pooled across lung modes: `co2Tolerance`: firstContractionShare;
`urgeOnset`: firstContractionSeconds; `desaturationRate`: desaturationRatePerMin; `diveReflex`:
hrDropShare; `recovery`: reoxygenationSeconds. (v3) There is no axis for how low SpO₂ went: nadir#703
and #706 record the low but never score it as a strength, so nothing rewards desaturating further. The
earlier `hypoxicTolerance` axis is removed.
`AxisValue { value: Double; count: Int; effectiveCount: Double }`. Gate key `"axis.<name>"`,
unit `.holdsWithSpO2` for desaturationRate/recovery, `.holdsWithContractions` for
co2Tolerance/urgeOnset, `.holdsWithSpO2` for diveReflex (HR comes with the oximeter), floors from
`Priors.evidence.axis`.

## 9. Hold context (pre-hold covariates)

`HoldContext.init(session:, stepIndex:, previousSessions: [RecordedSession], timeZone: TimeZone)`;
every field `Double?`:
`holdIndex` (0-based count of performed holds before it in the session), `logRestBefore`
(`ln(1 + Σ actualDuration of non-hold steps since the previous performed hold or session start)`),
`warmUp` (1 if any earlier performed hold in the session, else 0), `breatheHyperventilation` and
`breathePacking` (1/0 from `HoldRecords.breatheUp(before:in:)?.style`), `openEnded` (plannedDuration nil → 1),
`logDaysSinceLastSession` (`ln(1 + days)` to the newest finished session in `previousSessions` with
`startedAt` before this one; nil if none), `logDaysSinceLastMax` (same, to the newest one containing an
uncensored §10 row (v2)), `timeOfDaySin`/`timeOfDayCos` (`2π·h/24` with `h` the hour of
`startedAt + elapsed-before-hold` in `timeZone`, fractional), `preHoldHeartRate` (= `hrBaseline` of §7).
Never derived from the hold's own vitals, contractions, urges, outcome or duration, nor from feedback.
Covariate names are exactly these field names. Not used yet (listed): `effort` (rated after the hold;
history only), `breatheUpPacing`, `readiness`, `feedback` (outcome, never a predictor), `firstUrge`.

## 10. Hold model (tasks 4a, 4b, 5)

(v2, 22 Oct: the v1 rule `looksLikeMaxAttempt` selected 3 holds on the 17 Oct export and censored in
the wrong direction. The model's unit is a session's best attempt at a lung mode; it is a lower bound
unless the diver ended it.)
Rows: one per (finished session, lung-mode slot): the session's longest performed hold in that slot,
exactly `BestDetection.candidateHolds(in:)` (packed → full slot; first of equal holds), plus every
other performed hold in the session with `maxAttempt == true`. Hold eligibility before choosing the
longest: skip sessions whose `feedback.symptoms` contains `.hypoxia`; include a hold iff
(`!isImportedFromStamina` and `styleSource ∉ {estimated, imported}`) or `styleSource == "confirmed"`;
skip unknown style. Lung mode = `HoldStyle(style).bestLungMode` (packed → full). `y = ln(actualDuration)`.
`looksLikeMaxAttempt` is not used; `maxAttempt == false` is the engine's default for an unmarked plan
hold (SessionEngine writes `markedMaxAttempt`, default false), not a diver statement, and means nothing
here beyond "not marked".
`inSeries` = `iterationCount != nil && iterationCount! > 1 && maxAttempt != true`.
`diverEnded` (the diver chose the end):
- if `endTrigger` is present: `endTrigger ∈ {tap, skip}`;
- else: (`plannedDuration == nil` and `outcome == .completed`) or (`plannedDuration != nil` and
  `outcome ∈ {.skipped, .endedEarly(.tappedOut)}` and `actual < planned − 1`) or (`plannedDuration != nil`
  and `continuePastTarget == true` and `outcome == .completed` and `actual > planned + 1`).
Uncensored iff `diverEnded && !inSeries`. Everything else is right-censored at `actual`: a timer or
target end (incl. timed completed), the safety floor, an abort, and every hold inside a series (the
diver stops those on purpose). The gate's `.maxAttempts(v)` count is the number of uncensored rows.
`tNow` = newest finished session `startedAt`. Weight `w = h == nil ? 1 : 2^(−(tNow − t)/86400/h)`.
Covariates: admitted names `Priors.holdModel.covariates` (from §9) and `.axes` (from §8, as of the
row's session `startedAt`); missing value → the centre (contributes 0). Centre = weighted median of
that covariate over all rows (missing excluded). Adjusted `y' = y − β·(x − x̄) − γ·(a − ā)` with β, γ the
literal means.
Prior on μ_v: physiology `N(ln T_phys,v, σ_phys²)` (§12) when available, else placeholder
`N(ln(best_full × ratio_v), placeholderLogSD²)` with `best_full` = longest `actualDuration` among full
rows (any outcome) and `ratio = placeholderRatios[v]` (full 1, frc 1/2, rv 1/3); if neither exists the
volume is `.waiting` with requirement `.maxAttempts(.full)`.
σ²: prior inverse-gamma `(sigmaA0, sigmaB0)`. Alternate 6 times, starting at `σ² = b0/(a0 − 1)`:
(i) per volume, maximise `log p(μ) = −(μ − m0)²/(2 s0²) + Σ_unc w·(−(y' − μ)²/(2σ²)) + Σ_cens w·log(1 − Φ((y' − μ)/σ))`
with damped Newton from `m0` (≤ 50 steps, stop when |step| < 1e-12; halve the step up to 30 times
until log p does not decrease; fail → volume `.waiting`); `V_v = −1/(d²log p/dμ²)` at the mode;
(ii) `an = a0 + ½·Σ_unc w`, `bn = b0 + ½·Σ_unc w·(y' − μ_v)²` pooled over volumes, `σ² = bn/(an − 1)`.
`n_eff,v = Σ w over the volume's rows`, `n_unc,v` = count of uncensored rows (unweighted).
Level (80 %, `holdRange`): `exp(μ_v ± t90(2·an)·sqrt(V_v))`.
PB estimate: `x_PB` = componentwise median of `x` over the volume's 3 longest uncensored rows (fewer
if fewer), `a_PB` = axes as of `tNow + 1 s`; `μ* = μ_v + β·(x_PB − x̄) + γ·(a_PB − ā)`;
`var* = V_v + σ² + dxᵀΣ_β dx + daᵀΣ_γ da`; interval `exp(μ* ± t90(2·an)·sqrt(var*))`.
Prior predictive for narrowing: `exp(m0 ± t90(2·a0)·sqrt(s0² + b0/(a0−1)))`; narrowing compares
log-widths: `1 − (ln upper − ln lower)/(prior ln width)`. Level narrowing compares `sqrt(V_v)` to `s0`
(t df `2·an` vs normal): `1 − (t90(2an)·sqrt(V_v))/(1.2815515655446004·s0)`.
`rowsToReach` (pb): with current σ², smallest m (0…200) with
`2·t90(2an + m)·sqrt(1/(1/s0² + (n_unc + m)/σ²) + σ²) ≤ (1 − rhoEnough)·w_prior_log`.
Headroom: `percentile = Φ((ln best − ln T_phys,v)/σ_phys)` with `best` the PersonalBest passed in.
Gate keys `"pb.<v>"`, `"level.<v>"`; counts `uncensored`, `effective` (n_eff), `censored`.

## 11. Lung volumes (task 3)

Spirometry: per sex and measure (fvc, fev1): `ln Y[L] = a0 + a1·ln(height_cm) + a2·age + a3·age²`,
`ln σ = c0 + c1·age`; limits `exp(μ ∓ 1.6448536269514722·σ)` (5th/95th), `confidence 0.9`; nil outside
`ageRange`/`heightRange[sex]`. `compare(measured:to:)` → `percent = 100·measured/exp(μ)`,
`z = (ln measured − μ)/σ`. Static volumes (route 2): anchor = measured VC, else measured FVC, else
predicted FVC median; `TLC = anchor/ρ_VC(sex, age)`, `FRC = ρ_FRC·TLC`, `RV = ρ_RV·TLC` with
`ρ_x = r0 + r1·age` per sex from `Priors.staticVolumes`; log-σ per volume `sqrt(σ_anchor² + ratioLogSD_x²)`
(`σ_anchor = 0` when measured). A measured TLC/FRC/RV wins for that volume.

## 12. Oxygen store (task 5 physiology)

Constants `Priors.oxygenStore` (values from the verified literature table): lung `btpsToStpd`,
`fAO2Start`, `fAO2End`; blood per the verified decomposition (arterial fraction, start/end saturations);
`hufner` mL O₂/g; Hb default per sex g/dL; Nadler coefficients; muscle fraction per sex, myoglobin
g/kg muscle, usable fraction; `restingVO2PerKg` mL/kg/min; `diveReflexFactor`. Stores in mL:
`L_v = 1000·V_v·btpsToStpd·(fAO2Start − fAO2End)`, `B` per the decomposition, `M` per §literature,
`VO2 = weight·restingVO2PerKg·diveReflexFactor` mL/min, `T_v = 60·(L_v + B + M)/VO2` seconds.
`σ_phys = max(sigmaMC, sigmaFloor)`, constants from the harness. (v3) The physiology is never
personalised by the diver's own SpO₂ lows: a deeper low would lengthen the predicted PB, which rewards
desaturating. `personalisePhysiology` stays false and `sigmaMCPersonalised` is unused. Consumption
personalisation is deferred.

## 13. Evidence gate

`EvidenceGate.evaluate(key:, floors:, counts:, narrowing:, previouslyShown:)`. `floors` per task:
`enough: [Unit: Int]`, `keep: [Unit: Int]`, `rhoEnough`, `rhoKeep` (nil when the task has no prior).
Enough iff every unit count ≥ its floor (keep floor if previouslyShown) and (no narrowing or
realised ≥ rho (keep rho if previouslyShown)). Requirements: one per failing floor
`(unit, have: count, need: floor, .floor)`; narrowing failure → `(unit of the task's primary count, have,
need: have + rowsToReach, .narrowing)` or, when rowsToReach is nil, `(.narrowing, 0, 0, .narrowing)`.
The formatter merges requirements of the same unit by taking the largest `need − have`.
English summary (core, used for parity): `"Not enough data yet: " + phrases.joined(", ") + "."` with
phrases: maxAttempts(v) → `"N more max attempt(s) at <full lung|FRC|RV>"`; sessionsWithOximeter →
`"N more session(s) with this oximeter"`; holdsWithSpO2 → `"N more hold(s) with SpO₂"`;
holdsWithContractions → `"N more hold(s) with contractions marked"`; confirmedHolds(v) →
`"N more confirmed <full lung|FRC|RV> hold(s)"`; profileField(names) → `"needs your <a>, <b> and <c>"`;
lungVolume(name) → `"needs a <name>"`; readinessDays → `"wait until N days of readiness data with one source"`;
narrowing-unreachable → `"these max attempts vary too much to narrow the estimate yet"`. Singular when N == 1.

## 14. Priors literal and fixtures

Swift types (hand-written, `Learning/PriorTypes.swift`): `PostHoldDelayPrior`, `HoldVolumePrior`,
`SpirometryReference`, `StaticVolumeRatios`, `OxygenStoreConstants`, `HoldModelPrior`, `EvidenceFloors`.
`Priors.generated.swift` (written only by `emit_swift.py`) instantiates them plus `t90: [Double]`
(index 0 = df 1). Until the harness runs, a placeholder generated file with conservative defaults
exists and says so in its header.
Fixtures under Nadir `fixtures/learning/` are JSON objects `{ "generator": {...}, "cases": [...] }`;
each case has `"sessions"` in the exact Nadir session wire encoding (ISO-8601 dates, as in the
export; notes removed; `rrIntervals` removed) plus task inputs and `"expected"`. Files:
`features.json` (recoveryTrace, holdFacts feature vectors, holdMarkers, holdContext per hold),
`post-hold-delay.json`, `hold-volume.json`, `spirometry.json`, `static-volumes.json`,
`oxygen-store.json`, `diver-axes.json`, `hold-model.json`, `headroom.json`, `evidence.json`
(gate state + English summary per key). Tolerances: 1e-9 relative for closed forms, 1e-6 for Newton.
Fixtures carry a subset of real sessions (≤ 8) chosen to cover: a Masimo STAmina import, the 6 August
RV table, a PM100 RV table, a full-lung max with a short trace, the 2 October alternating session, a
session without SpO₂, a censored hold.

## 15. Deviations from the plan text

- (v2) Hold-model rows are session bests per lung-mode slot, uncensored only when the diver ended
  the hold outside a series (§10); `looksLikeMaxAttempt` and `maxAttempt == false` no longer select rows.

- Recovery onset: after the last occurrence of the post-hold minimum (fixes the mid-descent bounce
  Codex flagged on spO2-modelling #6), not "first rise above the running minimum".
- Effort is never a covariate of the same hold (it is rated after it); listed as history-only.
- Axes are pooled across lung modes; the per-oximeter split is reported by the harness only.
- (v3) No physiology personalisation by SpO₂ lows; consumption personalisation deferred.
- (v3) No axis scores how low SpO₂ went; dip onset and rate reuse `HoldOxygenDrop`.

## 16. Pinned by the Swift chunk 1 (22 Oct; the Python port must match)

- Unit names: one string per unit, `kind` or `kind.parameter`: `maxAttempts.rv|frc|full`,
  `sessionsWithOximeter.<key>`, `holdsWithSpO2`, `holdsWithContractions`, `confirmedHolds.<v>`,
  `profileField.<name>`, `lungVolume.<name>`, `readinessDays`, `narrowing`. Lung modes use the
  hold-style spelling (`rv`), never `empty`. The same strings key `EvidenceGate.counts`.
- Floors are keyed by unit kind (`Priors.evidence.<task>.enough[kind]`); a count matches a floor by
  its unit's kind.
- Requirement order: by kind declaration order (the order listed above), then by name; the narrowing
  requirement after the floor ones. Merging: a unit keeps the position of its first requirement and
  the largest `need − have`; all missing profile fields share one phrase at the first one's position.
- Every failing requirement shows at least 1 (`need − have` clamped to ≥ 1, in the gate).
- Readiness phrase uses N = need (the whole baseline), reason `.baseline`.
- NaN realised narrowing counts as not enough. Newton "does not decrease" allows 4 ulp of slack.
- No effective-holds floor: narrowing already fails when recency weights shrink the evidence.
- t90 values come from `scipy.stats.t.ppf(0.9, df)` in `emit_swift.py`, never a pasted list.
- Task 1 and the backfill read `step.recoveryTrace ?? Analytics.recoveryTrace(after:in:)`, so a stamp
  lost to an older build re-saving a session changes nothing.
- `OxygenStoreConstants` fields: `btpsToStpd`, `fAO2Start`, `fAO2End`, `arterialFraction`,
  `arterialSaturationStart`, `arterialSaturationEnd`, `venousSaturationStart`, `venousSaturationEnd`,
  `hufner`; per sex `male`/`female`: `haemoglobinGdL`, `nadlerK1`, `nadlerK2`, `nadlerK3`,
  `muscleFraction`; then `myoglobinPerKgMuscle`, `myoglobinO2PerGram`, `myoglobinUsableFraction`,
  `restingVO2PerKg`, `diveReflexFactor`, `sigmaMC`, `sigmaMCPersonalised`, `sigmaFloor`.
- A nil population prior (spirometry, static volumes, oxygen store, hold volume) means the task returns
  nothing; the shipped build always carries the harness literal.
- `CoachShare.redacted` clears `recoveryTrace` whenever it drops vitals.
- (harness, 22 Oct) Oxygen-store blood term: `arterialSaturationStart` and `arterialSaturationEnd` from the
  Severinghaus ODC in spo2-modelling (`backend/app/services/hill_model.py`) at PAO2 = 0.1444·713 and
  PETO2 end 26.9 mmHg (Lindholm 2006), status derived. The arterial/venous split is uncited, so the
  Monte Carlo samples `arterialFraction` uniform on [0.25, 1.0] with `venousSaturationStart` 0.75
  (textbook resting mixed-venous value, status unverified) and `venousSaturationEnd` =
  `arterialSaturationEnd`; the emitted point values are the Monte Carlo medians of the store, and
  `arterialFraction` is emitted as the value that reproduces the median. The results file prints two
  sourced cross-checks: the reference diver's usable share of the total store against Schagatay 2009's
  0.5 to 0.7, and the blood store against Rahn's 880 mL (70 kg).
- `StaticVolumeRatios` carries `ageFloor` (25: ECSC enters ages 18 to 25 as 25) and `ageRange`
  (18...70); `ρ_x = r0 + r1·max(age, ageFloor)`; outside `ageRange` the static volumes are nil (the
  spirometry may still show).
- Censored likelihood numerics, identical in Swift and Python: `logSurvival(z) = z < 0 ?
  log1p(−0.5·erfc(−z/√2)) : z < 30 ? log(0.5·erfc(z/√2)) : −z²/2 − log(z) − ½·log(2π) + log(1 − 1/z² + 3/z⁴)`
  (the z < 0 branch added 23 Oct: log of a number within 1e-9 of 1 lost the Newton step's precision); inverse Mills ratio
  `λ(z) = exp(−z²/2 − ½·log(2π) − logSurvival(z))`. Gradient and Hessian of `log p(μ)` use `λ`:
  censored term derivative `w·λ(z)/σ`, second derivative `−w·λ(z)·(λ(z) − z)/σ²`.
- (23 Oct) Profile-field units: parameter = the `DiverProfile` field identifier (`sex`, `dateOfBirth`,
  `heightCm`, `weightKg`); phrase display names "sex", "date of birth", "height", "weight"; ordered by
  identifier, so "needs your date of birth, height, sex and weight".
- `oxygenStore` gate: profile fields `sex`, `dateOfBirth`, `heightCm`, `weightKg`; the TLC comes from the
  static-volume estimate, so `lungVolume.tlc` is required only when every profile field is present and
  no TLC can be measured or estimated (outside the reference ranges).
- Physiology ratios are never fed to `DryBestEstimate` or anything else in Planning: they would raise
  RV and FRC table targets above what the diver has demonstrated (framing rule). Deviation from plan
  step 12; the Lung volumes section shows usable stores in litres only.
- Fixture layout: `fixtures/learning/sessions.json` = `{generator, sessions: [wire sessions]}` holds the
  real sessions once; every other file's cases carry `"sessionIDs": [...]` (resolved against
  sessions.json) or inline `"sessions"` for hand-built cases. Never both.
- Spirometry spread: final fit on all three cycles; per sex and measure a σ scale `k` from
  leave-one-cycle-out so the pooled held-out 5 to 95 % coverage is 90 %; `ln σ` is emitted with `ln k`
  folded into `c0`.
- English strings for the app live in the app's formatter (localised); the core `EvidenceGate.summary`
  is the English parity string only.
- (23 Oct, harness review) Static-volume spreads: when TLC is measured, `σ_FRC = ratioLogSD_FRC` and
  `σ_RV = ratioLogSD_RV` (no anchor term, no TLC-ratio term); when TLC is estimated from the anchor,
  `σ_TLC = sqrt(σ_anchor² + ratioLogSD_VC²)` and `σ_x = sqrt(σ_TLC² + ratioLogSD_x²)` for FRC and RV.
- Oxygen store: `restingVO2PerKg` is emitted as the Monte Carlo median of the sampled consumption
  (status derived), so `ln T_phys` is centred on the same distribution `σ_phys` describes.
- Fixtures record the floors they were built with and use the emitted `Priors.evidence` floors.

## 17. Packing (v4, 29 October)

Lung packing adds air above a full inhale. It is measured, never assumed.
- Measurement: `LungMeasurement.Kind.packedVC` with `packs: Int`, the vital capacity blown after that
  many packs. Packing efficiency `e = (packedVC − baseVC) / packs` litres per pack, where `baseVC` is the
  measured VC, else measured FVC, taken nearest in time to the packed measurement. No packed measurement
  means no efficiency: packing adds nothing anywhere, and packed estimates wait with
  `lungVolume.packedVC` ("needs a packed vital capacity"). There is no population default.
- Extra volume of a packed hold: `ΔV = e · min(packCount, packs)`, capped at the measured packed gain, so
  it never extrapolates beyond what was measured. A packed hold with no recorded `packCount` gets
  `ΔV = 0` (a lower bound on its volume). Recorded `packCount` is the only per-hold input.
- Oxygen store at a packed volume: the lung term uses `TLC + ΔV` (or the measured TLC plus ΔV); `fAO2Start`
  and `fAO2End` stay as for full lung (the literature's higher packed FETO2, 0.16, is not used, which
  understates the gain). Blood and muscle terms are unchanged.
- Hold model: packed holds stay in the full-lung slot (as #840 files their bests), with a fixed physiology
  offset instead of a new parameter: `y' = y − ln(U(V_full + ΔV) / U(V_full))`, where `U` is the usable store
  at the diver's profile. The offset is 0 without a profile or without an efficiency. So a packed hold
  tells the model about full-lung capacity after removing what the extra air should buy.
- Packed PB estimate: shown only next to a ready full-lung estimate, at the diver's most-used pack count
  among uncensored packed rows (else the measured `packs`), as `μ_full + ln(U(V_full + ΔV)/U(V_full))` with
  the full-lung interval; its gate is the full-lung gate plus the packed measurement.
- Packing efficiency itself is a measurement, always shown with its date, never scored and never a target.

## 18. Capacity from every hold (v5, 29 October)

Thomas's objection to §10: whether the diver pressed stop says little about where the limit was. A
one-minute hold the diver ended is not a max, and a 7:02 packed hold the diver ended is far more
informative than a count of attempts. The limit is read from the trace, not from the flag.
- Per-diver oxygen consumption `VO2_d` replaces the population `restingVO2PerKg · diveReflexFactor ·
  weight` in §12 when the hold evidence supports it. It is fitted from every performed hold with at
  least 60 s of trusted SpO₂, maximal or not: for each hold, the usable store at its lung volume
  (§11, §12, packing §17) and the oxygen the fall implies (store × fraction of the usable
  desaturation reached, from `spo2Baseline` to `spo2End` against the population end saturation)
  give a consumption estimate `store_used / duration`. A log-normal prior centred on the population
  value with sd 0.25 (so the population value stays when evidence is thin); posterior by weighted
  mean of log consumption over holds, weights = seconds of trusted SpO₂ in the hold, Student-t
  interval over the between-hold spread. Hypoxia-symptom sessions contribute to consumption like any
  other hold: consumption is a measurement of how fast O₂ went, not a reward.
- Distance to the limit per hold, `d ∈ [0, 1]`: the larger of (a) the SpO₂ fall reached as a fraction
  of the diver's own deepest fall (robust: 90th percentile of falls over holds with a trace) and
  (b) the struggle share `(duration − firstContraction) / duration` against the diver's 90th
  percentile of struggle shares; 0 when neither is measurable. `d` is descriptive and is never shown.
- Capacity rows (replaces §10's row rule): every performed hold with a trace is a row with weight
  `w_d = d²` (a one-minute easy hold adds almost nothing; a hold near the limit adds a lot), observed
  value `y = ln(actualDuration)`, and censoring: a hold is a lower bound unless `d ≥ 0.8`, in which
  case it is treated as an observation of the limit regardless of who ended it. The prior on the
  level is the physiology prior at `VO2_d`. Everything else in §10 (Newton, σ², intervals, narrowing,
  packing offset, context and axes admission by forward chain) is unchanged.
- Gate: `pb.<v>` floors become holds with a trace (`holdsWithSpO2`, 5/4) and effective weight
  `Σ w_d ≥ 2.0/1.5` (new unit `effectiveHolds`, Double), plus the narrowing test as before. The
  sentence reads "N more holds with SpO₂ at <volume>". The diver-ended flag and `maxAttempt` are no
  longer inputs; §10's `.maxAttempts` unit stays in the gate for the delay task's counts only.
- Symptoms (#704): the model uses every hold, always. No symptom report excludes a hold, in §18 or
  §10 v2 (v2's hypoxia skip is withdrawn). A reported symptom (hypoxia feeling today; blackout, LMC
  or samba once #704 adds them) is kept as information on the hold it belongs to, the session's last
  performed hold: a blackout marks that hold as past the limit, so it is an observation of the limit
  with `d = 1`, and the symptom rides as a marker on the fingerprint. What the app does with a
  blackout (hold a suggestion back, show a caution, refuse to file a best per nadir#873) is an
  application rule on top of the model, never a change to what the model learns from. Reported in
  the results file as counts.
- Forward chain: the new row rule is admitted only if it beats §10 v2 on log score at each next
  session's holds with `d ≥ 0.8`, with coverage 70 to 90 %. Both are printed. If it does not win,
  the results file says so and §10 v2 stays the shipped rule.
