"""
SpO₂ Desaturation Model — Hill Equation ODC
=============================================
Fits a physiological O₂ depletion model to finger pulse oximeter data
during a full-lung (TLC) static apnea hold.

Model structure:
    O₂(t)     = O₂_start - (VO₂ / 60) × max(t - lag, 0)
    PaO₂_eff  = O₂(t) / scale
    SpO₂(t)   = 100 × PaO₂_eff^n / (PaO₂_eff^n + P50^n)    [Hill equation]

Two phases:
    Phase 1 (arm down): SpO₂_finger = SpO₂(t)
    Phase 2 (arm up):   SpO₂_finger = SpO₂(t) - arm_offset × (100 - SpO₂(t)) / 100

Parameters:
    O₂_start    Total O₂ stores at hold start (blood + tissue + lung), mL
    VO₂         O₂ consumption rate, mL/min (constant)
    scale       Converts mL O₂ remaining → PaO₂-equivalent units
    P50         PaO₂_eff at which SpO₂ = 50% (cf. Hb P50 ≈ 26.6 mmHg)
    n           Hill coefficient (Hb cooperativity; textbook ≈ 2.7)
    arm_offset  Additional SpO₂ penalty when arm is raised, scaled by desaturation
    lag         Finger-to-arterial delay, seconds (fixed at 19s from empirical data)

Fitted values (Hold 6, TLC 7.0L, CO₂ pyramid session 2026-02-21):
    O₂_start  = 1966 mL
    VO₂       = 220 mL/min
    scale     = 12.8
    P50       = 50.7
    n         = 4.00
    arm_offset = 5.0%
    lag       = 19s

Result: BO (finger = 40%) at 6:34 arm-down. Hold ended at 6:12.
        Margin ≈ +22s (range +5 to +40s depending on VO₂ assumption).
"""

import numpy as np
from scipy.optimize import differential_evolution


# =============================================================================
# MODEL
# =============================================================================

def hill_spo2(pao2_eff: np.ndarray, p50: float, n: float) -> np.ndarray:
    """Oxygen-haemoglobin dissociation curve (Hill equation).
    
    Returns SpO₂ in % given effective PaO₂ units.
    Asymmetric: flat plateau at high PaO₂, steep in mid-range.
    """
    pao2_eff = np.maximum(pao2_eff, 0.01)
    return 100.0 * (pao2_eff ** n) / (pao2_eff ** n + p50 ** n)


def predict_spo2(
    t: np.ndarray,
    o2_start: float,
    vo2: float,
    scale: float,
    p50: float,
    n: float,
    lag: float,
    arm_up: bool = False,
    arm_offset: float = 0.0,
) -> np.ndarray:
    """Predict finger SpO₂ at times t (seconds from hold start).
    
    Args:
        t:           Time array in seconds
        o2_start:    Total O₂ at t=0, mL
        vo2:         O₂ consumption, mL/min
        scale:       mL O₂ → PaO₂-equivalent conversion
        p50:         PaO₂_eff at 50% SpO₂
        n:           Hill coefficient
        lag:         Finger delay, seconds
        arm_up:      Whether arm is raised (applies penalty)
        arm_offset:  Arm-raise penalty parameter (%)
    
    Returns:
        Predicted SpO₂ (%) array, same shape as t
    """
    t_eff = np.maximum(t - lag, 0.0)
    o2_remaining = o2_start - (vo2 / 60.0) * t_eff
    o2_remaining = np.maximum(o2_remaining, 0.01)

    pao2_eff = o2_remaining / scale
    spo2 = hill_spo2(pao2_eff, p50, n)

    if arm_up and arm_offset > 0:
        # Penalty increases as desaturation increases
        penalty = arm_offset * (100.0 - spo2) / 100.0
        spo2 = np.maximum(spo2 - penalty, 0.0)

    return spo2


# =============================================================================
# DATA LOADING
# =============================================================================

def load_apnea_data(csv_path: str) -> dict:
    """Load apnea session from exported CSV.
    
    Expected format: rows under 'Biometrics' header with columns:
        Time, elapsed (MM:SS), interval_type, HR, SpO₂
    
    Returns:
        dict mapping hold_number (1-6) to {
            't': seconds array,
            'spo2': SpO₂ array,
            'hr': HR array,
        }
    """
    import csv

    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        in_bio = False
        for row in reader:
            if row and row[0] == 'Biometrics':
                in_bio = True
                continue
            if in_bio and len(row) >= 5 and row[0] != 'Time':
                rows.append(row)

    # Split into intervals by type
    intervals = []
    current_block, current_type = [], None
    for r in rows:
        itype = r[2]
        if itype != current_type:
            if current_block:
                intervals.append((current_type, current_block))
            current_block, current_type = [], itype
        current_block.append(r)
    if current_block:
        intervals.append((current_type, current_block))

    # Apnea intervals are at indices 1, 3, 5, 7, 9, 11
    apnea_indices = [1, 3, 5, 7, 9, 11]
    apneas = {}

    for hold_num, ai in enumerate(apnea_indices, start=1):
        if ai >= len(intervals):
            break
        block = intervals[ai][1]
        t = np.array([
            int(r[1].split(':')[0]) * 60 + int(r[1].split(':')[1])
            for r in block
        ])
        spo2 = np.array([int(r[4]) for r in block], dtype=float)
        hr = np.array([int(r[3]) for r in block], dtype=float)
        apneas[hold_num] = {'t': t, 'spo2': spo2, 'hr': hr}

    return apneas


# =============================================================================
# FITTING
# =============================================================================

def fit_hold(
    t_phase1: np.ndarray,
    s_phase1: np.ndarray,
    t_phase2: np.ndarray = None,
    s_phase2: np.ndarray = None,
    lag: float = 19.0,
    n_range: tuple = (2.0, 4.0),
    seed: int = 42,
) -> dict:
    """Fit the Hill model to one hold's data.
    
    Args:
        t_phase1:  Time array (seconds), arm-down phase
        s_phase1:  SpO₂ array, arm-down phase
        t_phase2:  Time array, arm-up phase (optional)
        s_phase2:  SpO₂ array, arm-up phase (optional)
        lag:       Fixed finger lag in seconds
        n_range:   Bounds for Hill coefficient
        seed:      Random seed for differential evolution
    
    Returns:
        dict with fitted parameters and diagnostics
    """
    has_p2 = t_phase2 is not None and s_phase2 is not None

    def objective(params):
        if has_p2:
            o2_start, vo2, scale, p50, n_val, arm_off = params
        else:
            o2_start, vo2, scale, p50, n_val = params
            arm_off = 0.0

        pred1 = predict_spo2(t_phase1, o2_start, vo2, scale, p50, n_val, lag)
        err = np.sum((s_phase1 - pred1) ** 2)

        if has_p2:
            pred2 = predict_spo2(
                t_phase2, o2_start, vo2, scale, p50, n_val, lag,
                arm_up=True, arm_offset=arm_off,
            )
            err += np.sum((s_phase2 - pred2) ** 2)

        return err

    bounds = [
        (1800, 2800),       # o2_start
        (100, 300),         # vo2
        (5, 50),            # scale
        (15, 60),           # p50
        n_range,            # n
    ]
    if has_p2:
        bounds.append((5, 50))  # arm_offset

    result = differential_evolution(
        objective, bounds,
        maxiter=10000, seed=seed, tol=1e-14,
        polish=True, popsize=80,
        mutation=(0.5, 1.5), recombination=0.9,
    )

    if has_p2:
        o2_start, vo2, scale, p50, n_val, arm_off = result.x
    else:
        o2_start, vo2, scale, p50, n_val = result.x
        arm_off = 0.0

    # Goodness of fit
    pred1 = predict_spo2(t_phase1, o2_start, vo2, scale, p50, n_val, lag)
    ss_res1 = np.sum((s_phase1 - pred1) ** 2)
    ss_tot1 = np.sum((s_phase1 - np.mean(s_phase1)) ** 2)
    r2_p1 = 1 - ss_res1 / ss_tot1 if ss_tot1 > 0 else 0

    r2_p2 = None
    if has_p2:
        pred2 = predict_spo2(
            t_phase2, o2_start, vo2, scale, p50, n_val, lag,
            arm_up=True, arm_offset=arm_off,
        )
        ss_res2 = np.sum((s_phase2 - pred2) ** 2)
        ss_tot2 = np.sum((s_phase2 - np.mean(s_phase2)) ** 2)
        r2_p2 = 1 - ss_res2 / ss_tot2 if ss_tot2 > 0 else 0

    return {
        'o2_start': o2_start,
        'vo2': vo2,
        'scale': scale,
        'p50': p50,
        'n': n_val,
        'arm_offset': arm_off,
        'lag': lag,
        'r2_phase1': r2_p1,
        'r2_phase2': r2_p2,
        'converged': result.success,
        'objective': result.fun,
    }


# =============================================================================
# ANALYSIS UTILITIES
# =============================================================================

def find_threshold_time(
    params: dict,
    threshold: float = 40.0,
    arm_up: bool = False,
    t_max: float = 800.0,
    dt: float = 0.5,
) -> float | None:
    """Find time (seconds) at which SpO₂ crosses below threshold.
    
    Returns None if threshold is never reached within t_max.
    """
    t = np.arange(0, t_max, dt)
    spo2 = predict_spo2(
        t, params['o2_start'], params['vo2'],
        params['scale'], params['p50'], params['n'], params['lag'],
        arm_up=arm_up, arm_offset=params['arm_offset'],
    )
    idx = np.where(spo2 <= threshold)[0]
    return float(t[idx[0]]) if len(idx) > 0 else None


def sensitivity_vo2(
    params: dict,
    hold_end: float = 372.0,
    pct_range: range = range(-15, 16, 5),
    threshold: float = 40.0,
) -> list[dict]:
    """VO₂ sensitivity analysis.
    
    Returns list of dicts with vo2, pct_change, bo_time, margin, spo2_at_end.
    """
    results = []
    t = np.arange(0, 800, 0.5)

    for pct in pct_range:
        vo2_test = params['vo2'] * (1 + pct / 100)
        spo2 = predict_spo2(
            t, params['o2_start'], vo2_test,
            params['scale'], params['p50'], params['n'], params['lag'],
        )
        idx_end = int(hold_end * 2)
        spo2_end = float(spo2[idx_end]) if idx_end < len(spo2) else 0.0

        cross = np.where(spo2 <= threshold)[0]
        bo_time = float(t[cross[0]]) if len(cross) > 0 else None
        margin = bo_time - hold_end if bo_time else None

        results.append({
            'vo2': vo2_test,
            'pct_change': pct,
            'bo_time': bo_time,
            'margin': margin,
            'spo2_at_end': spo2_end,
        })

    return results


def desaturation_rate(
    params: dict,
    t_points: list[float],
    dt: float = 0.5,
) -> list[dict]:
    """Compute desaturation rate at specified times.
    
    Returns list of dicts with time, rate_per_min.
    """
    t = np.arange(0, 800, dt)
    spo2 = predict_spo2(
        t, params['o2_start'], params['vo2'],
        params['scale'], params['p50'], params['n'], params['lag'],
    )
    gradient = np.gradient(spo2, dt) * 60  # %/min

    results = []
    for tp in t_points:
        idx = int(tp / dt)
        if idx < len(gradient):
            results.append({'time': tp, 'rate_per_min': float(gradient[idx])})
    return results


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == '__main__':
    import sys

    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'session.csv'

    # Load
    apneas = load_apnea_data(csv_path)
    t6 = apneas[6]['t']
    s6 = apneas[6]['spo2']

    # Split phases (adjust times to your session)
    ISCHAEMIC_END = 40      # seconds — end of initial transient
    ARM_RAISE_TIME = 290    # seconds — when arm was raised
    HOLD_END = 372          # seconds — when hold ended

    mask_p1 = (t6 >= ISCHAEMIC_END) & (t6 <= ARM_RAISE_TIME)
    mask_p2 = (t6 >= ARM_RAISE_TIME + 10) & (t6 <= HOLD_END)

    # Fit
    params = fit_hold(
        t_phase1=t6[mask_p1], s_phase1=s6[mask_p1],
        t_phase2=t6[mask_p2], s_phase2=s6[mask_p2],
        lag=19.0,
        n_range=(2.0, 4.0),
    )

    print("Fitted parameters:")
    for k, v in params.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.1f}")
        else:
            print(f"  {k}: {v}")

    # BO prediction
    bo = find_threshold_time(params, threshold=40.0, arm_up=False)
    margin = bo - HOLD_END if bo else None
    print(f"\nBO (arm down, 40%): {bo:.0f}s = {int(bo//60)}:{int(bo%60):02d}")
    print(f"Margin from hold end: {margin:+.0f}s")

    # Sensitivity
    print("\nVO₂ sensitivity:")
    for r in sensitivity_vo2(params, hold_end=HOLD_END):
        if r['margin'] is not None:
            print(f"  VO₂={r['vo2']:.0f} ({r['pct_change']:+d}%): "
                  f"margin={r['margin']:+.0f}s, SpO₂@end={r['spo2_at_end']:.0f}%")
