"""
Analysis Utilities
==================

Tools for analyzing fitted models: threshold crossing prediction,
parameter sensitivity analysis, and desaturation rate computation.
"""

from dataclasses import dataclass

import numpy as np
from loguru import logger

from app.services.hill_model import ApneaModelParams, predict_spo2


@dataclass
class ThresholdResult:
    """Result of a threshold crossing search."""

    threshold: float          # SpO2 threshold searched for
    crossing_time_s: float | None  # Time at which threshold is crossed
    crossing_time_fmt: str | None  # Formatted as M:SS
    spo2_at_end: float | None      # SpO2 at t_max if threshold not reached


@dataclass
class SensitivityPoint:
    """One point in a parameter sensitivity analysis."""

    param_value: float
    pct_change: float
    crossing_time_s: float | None
    margin_s: float | None       # crossing_time - reference_time
    spo2_at_ref: float           # SpO2 at reference time


@dataclass
class DesatRatePoint:
    """Desaturation rate at a specific time."""

    time_s: float
    rate_per_min: float  # SpO2 %/min (negative = desaturating)
    spo2: float          # SpO2 at this time


def format_time(seconds: float) -> str:
    """Format seconds as M:SS."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}:{s:02d}"


def find_threshold_time(
    params: ApneaModelParams,
    threshold: float = 40.0,
    t_max: float = 800.0,
    dt: float = 0.5,
) -> ThresholdResult:
    """Find time at which SpO2 crosses below a threshold.

    Args:
        params:    Fitted model parameters
        threshold: SpO2 level to search for (%)
        t_max:     Maximum time to search (seconds)
        dt:        Time step (seconds)

    Returns:
        ThresholdResult with crossing time or None if not reached
    """
    t = np.arange(0, t_max, dt)
    spo2 = predict_spo2(t, params)

    idx = np.where(spo2 <= threshold)[0]
    if len(idx) > 0:
        crossing = float(t[idx[0]])
        logger.debug(f"Threshold {threshold}% crossed at {format_time(crossing)}")
        return ThresholdResult(
            threshold=threshold,
            crossing_time_s=crossing,
            crossing_time_fmt=format_time(crossing),
            spo2_at_end=None,
        )
    else:
        spo2_end = float(spo2[-1]) if len(spo2) > 0 else None
        logger.debug(f"Threshold {threshold}% not reached within {t_max}s (SpO2 at end: {spo2_end})")
        return ThresholdResult(
            threshold=threshold,
            crossing_time_s=None,
            crossing_time_fmt=None,
            spo2_at_end=spo2_end,
        )


def sensitivity_analysis(
    params: ApneaModelParams,
    param_name: str = "tau_washout",
    reference_time_s: float = 372.0,
    pct_range: list[int] | None = None,
    threshold: float = 40.0,
    t_max: float = 800.0,
    dt: float = 0.5,
) -> list[SensitivityPoint]:
    """Parameter sensitivity analysis.

    Varies a named parameter by percentage and reports how crossing time
    and margin change.

    Args:
        params:          Base fitted parameters
        param_name:      Parameter to vary (must be a field of ApneaModelParams)
        reference_time_s: Reference time to compute margin from (e.g., hold end)
        pct_range:       Percentage changes to test (default: -15 to +15 in steps of 5)
        threshold:       SpO2 threshold for crossing
        t_max:           Max simulation time
        dt:              Time step

    Returns:
        List of SensitivityPoint results
    """
    if pct_range is None:
        pct_range = list(range(-15, 16, 5))

    if not hasattr(params, param_name):
        raise ValueError(f"Unknown parameter: {param_name!r}")

    base_value = getattr(params, param_name)
    t = np.arange(0, t_max, dt)
    results = []

    for pct in pct_range:
        test_value = base_value * (1 + pct / 100)
        test_params = ApneaModelParams.from_dict({**params.to_dict(), param_name: test_value})

        spo2 = predict_spo2(t, test_params)

        # SpO2 at reference time
        ref_idx = int(reference_time_s / dt)
        spo2_at_ref = float(spo2[ref_idx]) if ref_idx < len(spo2) else 0.0

        # Crossing time
        cross_idx = np.where(spo2 <= threshold)[0]
        crossing = float(t[cross_idx[0]]) if len(cross_idx) > 0 else None
        margin = crossing - reference_time_s if crossing else None

        results.append(SensitivityPoint(
            param_value=test_value,
            pct_change=float(pct),
            crossing_time_s=crossing,
            margin_s=margin,
            spo2_at_ref=spo2_at_ref,
        ))

    return results


def desaturation_rate(
    params: ApneaModelParams,
    time_points: list[float],
    dt: float = 0.5,
    t_max: float = 800.0,
) -> list[DesatRatePoint]:
    """Compute instantaneous desaturation rate at specified times.

    Args:
        params:      Fitted model parameters
        time_points: Times at which to compute the rate (seconds)
        dt:          Time step for gradient computation
        t_max:       Max simulation time

    Returns:
        List of DesatRatePoint with rate in %/min
    """
    t = np.arange(0, t_max, dt)
    spo2 = predict_spo2(t, params)
    gradient = np.gradient(spo2, dt) * 60  # Convert to %/min

    results = []
    for tp in time_points:
        idx = int(tp / dt)
        if idx < len(gradient):
            results.append(DesatRatePoint(
                time_s=tp,
                rate_per_min=float(gradient[idx]),
                spo2=float(spo2[idx]),
            ))
        else:
            logger.warning(f"Time point {tp}s exceeds simulation range")

    return results


def generate_prediction_curve(
    params: ApneaModelParams,
    t_max: float = 600.0,
    dt: float = 1.0,
) -> dict:
    """Generate a full prediction curve for visualization.

    Returns:
        Dict with 't', 'spo2', 'spo2_base', 'pao2', 'p50_eff' keys
    """
    from app.services.hill_model import predict_spo2_components

    t = np.arange(0, t_max, dt)
    components = predict_spo2_components(t, params)

    return {
        "t": t.tolist(),
        "spo2": components["total"].tolist(),
        "spo2_base": components["base"].tolist(),
        "pao2": components["pao2"].tolist(),
        "p50_eff": components["p50_eff"].tolist(),
    }
