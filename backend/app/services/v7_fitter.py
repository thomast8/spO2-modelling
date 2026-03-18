"""
v7 Two-Stage Fitting Engine
============================

Orchestrates the two-stage fitting process:
- Stage A: Sensor calibration (power-law latent + gamma kernel)
- Stage B: Physiology (exponential washout + CO2 + Severinghaus ODC)

The fitting is CPU-intensive (minutes), so it runs in a thread via asyncio.to_thread().
"""

import numpy as np
from loguru import logger
from scipy.optimize import differential_evolution

from app.services.v7_model import (
    BS_FIXED,
    CV_FIXED,
    DELTA_BOUNDS,
    EFF_LAG_MIN,
    GAMMA_BOUNDS,
    GAMMA_PRIOR_CENTER,
    K_CO2_BOUNDS,
    K_CO2_PRIOR_CENTER,
    LAMBDA_DELTA,
    LAMBDA_GAMMA,
    LAMBDA_K_CO2,
    LAMBDA_NADIR,
    LAMBDA_P,
    LAMBDA_PACO2,
    LAMBDA_REG,
    LAMBDA_TAU0,
    LAMBDA_ZEROSUM,
    N_PH,
    P_BOUNDS,
    P_PRIOR_CENTER,
    PERHOLD_BOUNDS,
    PVO2_FIXED,
    S_MIN_BOUNDS,
    TAU0_BOUNDS,
    TAU0_PRIOR_CENTER,
    V_UP_BOUNDS,
    FrozenSensor,
    StageAGlobalParams,
    StageAPerHold,
    StageBGlobalParams,
    StageBPerHold,
    StageResult,
    V7FitResult,
    V7ModelParams,
    apply_gamma_kernel,
    build_powerlaw_latent,
    compute_nadir_info,
    compute_r2,
    nadir_timing_penalty_huber,
    nadir_window_mask,
    predict_stage_b_hold,
    student_t_nll,
)


# -- Stage A: Sensor calibration ---------------------------------------------


def run_stage_a(
    fit_holds: list[dict],
    nadir_info: dict[int, dict],
    s_base_values: dict[int, float],
    seed: int = 42,
) -> tuple[np.ndarray, bool, float]:
    """Stage A: Sensor calibration with b_s=1 fixed, CV=0.15.

    Returns:
        (flat_a, converged, objective_val)
        flat_a layout: [tau_0, b_s, p, deltas..., (S_min, v_up) per hold...]
    """
    n_holds = len(fit_holds)

    # Build bounds: tau_0, p, deltas, (S_min, v_up) per hold
    bounds: list[tuple[float, float]] = [TAU0_BOUNDS, P_BOUNDS]
    n_global = 2
    delta_offset = n_global

    for _ in fit_holds:
        bounds.append(DELTA_BOUNDS)

    latent_offset = delta_offset + n_holds
    for _ in fit_holds:
        bounds.append(S_MIN_BOUNDS)
        bounds.append(V_UP_BOUNDS)

    logger.info(
        f"Stage A: {len(bounds)} params ({n_global} global + {n_holds} delta + "
        f"2x{n_holds} latent), seed={seed}"
    )

    masks = [nadir_window_mask(h["t"], h["t_end"]) for h in fit_holds]
    nadir_ts = [nadir_info[h["id"]]["t_nadir"] for h in fit_holds]
    t_grids = [np.arange(0, h["t"][-1] + 1, 1.0) for h in fit_holds]

    def objective(flat: np.ndarray) -> float:
        tau_0, p = flat[:n_global]
        deltas = flat[delta_offset:delta_offset + n_holds]
        total = 0.0

        for i, h in enumerate(fit_holds):
            lp_start = latent_offset + i * 2
            s_min_val, v_up = flat[lp_start:lp_start + 2]
            s_start = s_base_values[h["id"]]
            t_1hz = t_grids[i]
            latent = build_powerlaw_latent(t_1hz, h["t_end"], s_start, s_min_val, v_up, p)
            eff_lag = max(tau_0 + deltas[i], EFF_LAG_MIN)
            filtered = apply_gamma_kernel(latent, eff_lag, CV_FIXED)
            pred_at_obs = np.interp(h["t"], t_1hz, filtered)
            pred_display = np.clip(pred_at_obs, 0.0, 100.0)
            m = masks[i]
            residuals = h["spo2"][m] - pred_at_obs[m]
            total += student_t_nll(residuals)
            total += nadir_timing_penalty_huber(h["t"][m], pred_display[m], nadir_ts[i])

        # Priors
        total += LAMBDA_TAU0 * (np.log(max(tau_0, 1.0)) - np.log(TAU0_PRIOR_CENTER)) ** 2
        total += LAMBDA_DELTA * np.sum(np.log1p(deltas**2 / 9.0))
        total += LAMBDA_ZEROSUM * np.sum(deltas) ** 2
        total += LAMBDA_P * (np.log(max(p, 0.1)) - np.log(P_PRIOR_CENTER)) ** 2

        return total

    maxiter_a = 4000
    res = differential_evolution(
        objective, bounds, maxiter=maxiter_a, seed=seed, tol=1e-10,
        polish=True, popsize=40, mutation=(0.5, 1.5), recombination=0.9,
    )
    logger.info(f"Stage A converged: {res.success}, fun={res.fun:.2f}")

    # Build full vector with b_s=1.0 inserted at index 1
    full = np.empty(len(res.x) + 1)
    full[0] = res.x[0]       # tau_0
    full[1] = BS_FIXED        # b_s (fixed)
    full[2:] = res.x[1:]     # p, deltas, latent params

    return full, res.success, float(res.fun)


def evaluate_stage_a(
    flat_a: np.ndarray,
    fit_holds: list[dict],
    nadir_info: dict[int, dict],
    s_base_values: dict[int, float],
) -> tuple[dict[int, float], dict[int, dict], dict[int, dict]]:
    """Evaluate Stage A results.

    Returns:
        (per_hold_r2, per_hold_predictions, latent_curves)
    """
    n_holds = len(fit_holds)
    n_global = 3  # tau_0, b_s, p
    delta_offset = n_global
    latent_offset = delta_offset + n_holds

    tau_0, _b_s, p = flat_a[:n_global]
    deltas = flat_a[delta_offset:delta_offset + n_holds]

    per_hold_r2: dict[int, float] = {}
    per_hold_predictions: dict[int, dict] = {}
    latent_curves: dict[int, dict] = {}

    for i, h in enumerate(fit_holds):
        lp_start = latent_offset + i * 2
        s_min_val, v_up = flat_a[lp_start:lp_start + 2]
        s_start = s_base_values[h["id"]]

        t_1hz = np.arange(0, h["t"][-1] + 1, 1.0)
        latent = build_powerlaw_latent(t_1hz, h["t_end"], s_start, s_min_val, v_up, p)
        eff_lag = max(tau_0 + deltas[i], EFF_LAG_MIN)
        filtered = apply_gamma_kernel(latent, eff_lag, CV_FIXED)
        pred_at_obs = np.interp(h["t"], t_1hz, filtered)
        pred_at_obs = np.clip(pred_at_obs, 0.0, 100.0)

        r2 = compute_r2(h["spo2"], pred_at_obs)
        per_hold_r2[h["id"]] = r2
        per_hold_predictions[h["id"]] = {
            "elapsed_s": h["t"].tolist(),
            "observed": h["spo2"].tolist(),
            "predicted": pred_at_obs.tolist(),
        }
        latent_curves[h["id"]] = {
            "t_1hz": t_1hz.tolist(),
            "latent": latent.tolist(),
            "filtered": filtered.tolist(),
            "s_start": s_start,
            "s_min": s_min_val,
            "v_up": v_up,
        }

    return per_hold_r2, per_hold_predictions, latent_curves


def extract_frozen_sensor(flat_a: np.ndarray, fit_holds: list[dict]) -> FrozenSensor:
    """Extract frozen sensor params from Stage A result."""
    n_holds = len(fit_holds)
    n_global = 3
    delta_offset = n_global
    tau_0, b_s, p = flat_a[:n_global]
    deltas = flat_a[delta_offset:delta_offset + n_holds]
    return FrozenSensor(
        tau_0=float(tau_0),
        b_s=float(b_s),
        p=float(p),
        deltas=[float(d) for d in deltas],
        cv=CV_FIXED,
    )


# -- Stage B: Physiology (frozen sensor) --------------------------------------


def run_stage_b(
    fit_holds: list[dict],
    nadir_info: dict[int, dict],
    frozen_sensor: FrozenSensor,
    s_base_values: dict[int, float],
    seed: int = 42,
) -> tuple[np.ndarray, bool, float]:
    """Stage B: Physiology fitting with frozen sensor.

    Returns:
        (flat_b, converged, objective_val)
        flat_b layout: [k_co2, gamma, deltas_frozen..., (tau_washout, paco2_0) per hold...]
    """
    tau_0_frozen = frozen_sensor.tau_0
    b_s_frozen = frozen_sensor.b_s
    cv_frozen = frozen_sensor.cv
    deltas_frozen = np.array(frozen_sensor.deltas)

    n_holds = len(fit_holds)

    bounds: list[tuple[float, float]] = [K_CO2_BOUNDS, GAMMA_BOUNDS]
    n_phys = 2

    ic_offset = n_phys
    for h in fit_holds:
        ht = h["type"]
        if ht not in PERHOLD_BOUNDS:
            logger.warning(f"No bounds for hold type {ht!r}, using FL bounds")
            ht = "FL"
        bounds.extend(PERHOLD_BOUNDS[ht])

    type_groups: dict[str, list[int]] = {}
    for i, h in enumerate(fit_holds):
        type_groups.setdefault(h["type"], []).append(i)

    apnea_window = 5
    masks = [h["t"] <= h["t_end"] + apnea_window for h in fit_holds]
    weights = [
        np.where(h["spo2"][m] < 95, 3.0, 1.0)
        for h, m in zip(fit_holds, masks)
    ]

    logger.info(f"Stage B: {len(bounds)} params, seed={seed}")

    def objective(flat: np.ndarray) -> float:
        k_co2, gamma_val = flat[:n_phys]
        total = 0.0

        for i, h in enumerate(fit_holds):
            ph_offset = ic_offset + i * N_PH
            tau_washout, paco2_0 = flat[ph_offset:ph_offset + N_PH]

            pred = predict_stage_b_hold(
                h["t"], tau_washout, gamma_val,
                paco2_0, k_co2, b_s_frozen,
                tau_0_frozen, cv_frozen, h["t_end"],
                s_base=s_base_values[h["id"]],
                shift=deltas_frozen[i],
            )
            m = masks[i]
            total += np.sum(weights[i] * (h["spo2"][m] - pred[m]) ** 2)
            total += LAMBDA_PACO2 * (paco2_0 - 40.0) ** 2

        total += LAMBDA_K_CO2 * (k_co2 - K_CO2_PRIOR_CENTER) ** 2
        total += LAMBDA_GAMMA * (gamma_val - GAMMA_PRIOR_CENTER) ** 2

        for _ht, indices in type_groups.items():
            if len(indices) < 2:
                continue
            for p_off in range(N_PH):
                values = [flat[ic_offset + idx * N_PH + p_off] for idx in indices]
                mean_val = np.mean(values)
                total += LAMBDA_REG * sum((v - mean_val) ** 2 for v in values)

        return total

    maxiter_b = 2000
    res = differential_evolution(
        objective, bounds, maxiter=maxiter_b, seed=seed, tol=1e-10,
        polish=True, popsize=25, mutation=(0.5, 1.5), recombination=0.9,
    )
    logger.info(f"Stage B converged: {res.success}, fun={res.fun:.2f}")

    k_co2, gamma_val = res.x[:n_phys]
    ics = res.x[n_phys:]
    # Insert frozen deltas for evaluate compatibility
    full_b = np.concatenate([
        [k_co2, gamma_val],
        deltas_frozen,
        ics,
    ])
    return full_b, res.success, float(res.fun)


def evaluate_stage_b(
    flat_b: np.ndarray,
    fit_holds: list[dict],
    frozen_sensor: FrozenSensor,
    s_base_values: dict[int, float],
) -> tuple[dict[int, float], dict[int, dict]]:
    """Evaluate Stage B results.

    Returns:
        (per_hold_r2, per_hold_predictions)
    """
    n_holds = len(fit_holds)
    n_phys = 2
    delta_offset = n_phys
    ic_offset = delta_offset + n_holds

    k_co2, gamma_val = flat_b[:n_phys]
    deltas = flat_b[delta_offset:delta_offset + n_holds]

    per_hold_r2: dict[int, float] = {}
    per_hold_predictions: dict[int, dict] = {}

    for i, h in enumerate(fit_holds):
        ph_start = ic_offset + i * N_PH
        tau_washout, paco2_0 = flat_b[ph_start:ph_start + N_PH]

        pred = predict_stage_b_hold(
            h["t"], tau_washout, gamma_val,
            paco2_0, k_co2, frozen_sensor.b_s,
            frozen_sensor.tau_0, frozen_sensor.cv, h["t_end"],
            s_base=s_base_values[h["id"]],
            shift=deltas[i],
        )

        r2 = compute_r2(h["spo2"], pred)
        per_hold_r2[h["id"]] = r2
        per_hold_predictions[h["id"]] = {
            "elapsed_s": h["t"].tolist(),
            "observed": h["spo2"].tolist(),
            "predicted": pred.tolist(),
        }

    return per_hold_r2, per_hold_predictions


# -- Orchestrator -------------------------------------------------------------


def _extract_params(
    flat_a: np.ndarray,
    flat_b: np.ndarray,
    fit_holds: list[dict],
    s_base_values: dict[int, float],
) -> V7ModelParams:
    """Extract structured params from flat optimization vectors."""
    n_holds = len(fit_holds)
    n_global_a = 3
    delta_offset_a = n_global_a
    latent_offset_a = delta_offset_a + n_holds

    tau_0, _b_s, p = flat_a[:n_global_a]
    deltas_a = flat_a[delta_offset_a:delta_offset_a + n_holds]

    n_phys_b = 2
    delta_offset_b = n_phys_b
    ic_offset_b = delta_offset_b + n_holds
    k_co2, gamma_val = flat_b[:n_phys_b]

    stage_a_global = StageAGlobalParams(tau_0=float(tau_0), p=float(p))
    stage_a_per_hold = []
    for i, h in enumerate(fit_holds):
        lp_start = latent_offset_a + i * 2
        s_min_val, v_up = flat_a[lp_start:lp_start + 2]
        stage_a_per_hold.append(StageAPerHold(
            hold_id=h["id"],
            hold_type=h["type"],
            delta=float(deltas_a[i]),
            s_min=float(s_min_val),
            v_up=float(v_up),
            eff_lag=float(max(tau_0 + deltas_a[i], EFF_LAG_MIN)),
        ))

    stage_b_global = StageBGlobalParams(k_co2=float(k_co2), gamma=float(gamma_val))
    stage_b_per_hold = []
    for i, h in enumerate(fit_holds):
        ph_start = ic_offset_b + i * N_PH
        tau_washout, paco2_0 = flat_b[ph_start:ph_start + N_PH]
        stage_b_per_hold.append(StageBPerHold(
            hold_id=h["id"],
            hold_type=h["type"],
            tau_washout=float(tau_washout),
            paco2_0=float(paco2_0),
        ))

    return V7ModelParams(
        stage_a_global=stage_a_global,
        stage_a_per_hold=stage_a_per_hold,
        stage_b_global=stage_b_global,
        stage_b_per_hold=stage_b_per_hold,
        s_base_values=s_base_values,
    )


def fit_v7(
    hold_data_list: list[dict],
    s_base_values: dict[int, float],
    seed: int = 42,
) -> V7FitResult:
    """Run the complete two-stage v7 fit.

    Args:
        hold_data_list: Enriched holds with recovery data (from extract_holds_with_recovery)
        s_base_values: {hold_id: baseline SpO2} (typically the first few seconds' mean)
        seed: Random seed for reproducibility

    Returns:
        V7FitResult with all params, diagnostics, and predictions
    """
    logger.info(f"Starting v7 fit on {len(hold_data_list)} holds, seed={seed}")

    # Compute nadir info for all holds
    nadir_info = {h["id"]: compute_nadir_info(h) for h in hold_data_list}

    # -- Stage A --
    flat_a, conv_a, obj_a = run_stage_a(hold_data_list, nadir_info, s_base_values, seed)
    r2_a, preds_a, latent_curves = evaluate_stage_a(
        flat_a, hold_data_list, nadir_info, s_base_values
    )

    stage_a_result = StageResult(
        converged=conv_a,
        objective_val=obj_a,
        per_hold_r2=r2_a,
        per_hold_predictions=preds_a,
    )

    # -- Extract frozen sensor --
    frozen_sensor = extract_frozen_sensor(flat_a, hold_data_list)

    # -- Stage B --
    flat_b, conv_b, obj_b = run_stage_b(
        hold_data_list, nadir_info, frozen_sensor, s_base_values, seed
    )
    r2_b, preds_b = evaluate_stage_b(flat_b, hold_data_list, frozen_sensor, s_base_values)

    stage_b_result = StageResult(
        converged=conv_b,
        objective_val=obj_b,
        per_hold_r2=r2_b,
        per_hold_predictions=preds_b,
    )

    # Overall R2 from Stage B predictions (the final model output)
    all_obs = np.concatenate([
        np.array(preds_b[h["id"]]["observed"]) for h in hold_data_list
    ])
    all_pred = np.concatenate([
        np.array(preds_b[h["id"]]["predicted"]) for h in hold_data_list
    ])
    overall_r2 = compute_r2(all_obs, all_pred)

    # Extract structured params
    params = _extract_params(flat_a, flat_b, hold_data_list, s_base_values)

    logger.info(
        f"v7 fit complete: R2={overall_r2:.4f}, "
        f"Stage A conv={conv_a}, Stage B conv={conv_b}"
    )

    return V7FitResult(
        params=params,
        stage_a_result=stage_a_result,
        stage_b_result=stage_b_result,
        overall_r2=overall_r2,
        hold_ids=[h["id"] for h in hold_data_list],
        frozen_sensor=frozen_sensor,
        stage_a_latent=latent_curves,
    )
