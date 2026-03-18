"""
Analysis Utilities
==================

Tools for analyzing fitted models: threshold crossing prediction,
parameter sensitivity analysis, and desaturation rate computation.

v7: These functions accept a predict_fn callback instead of model params directly,
since v7 predictions are per-hold and require the full two-stage model context.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from loguru import logger


@dataclass
class ThresholdResult:
    """Result of a threshold crossing search."""

    threshold: float
    crossing_time_s: float | None
    crossing_time_fmt: str | None
    spo2_at_end: float | None


@dataclass
class SensitivityPoint:
    """One point in a parameter sensitivity analysis."""

    param_value: float
    pct_change: float
    crossing_time_s: float | None
    margin_s: float | None
    spo2_at_ref: float


@dataclass
class DesatRatePoint:
    """Desaturation rate at a specific time."""

    time_s: float
    rate_per_min: float
    spo2: float


def format_time(seconds: float) -> str:
    """Format seconds as M:SS."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}:{s:02d}"


def find_threshold_time(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    threshold: float = 40.0,
    t_max: float = 800.0,
    dt: float = 0.5,
) -> ThresholdResult:
    """Find time at which SpO2 crosses below a threshold.

    Args:
        predict_fn: Function that takes time array and returns SpO2 array
        threshold:  SpO2 level to search for (%)
        t_max:      Maximum time to search (seconds)
        dt:         Time step (seconds)
    """
    t = np.arange(0, t_max, dt)
    spo2 = predict_fn(t)

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
        logger.debug(f"Threshold {threshold}% not reached within {t_max}s")
        return ThresholdResult(
            threshold=threshold,
            crossing_time_s=None,
            crossing_time_fmt=None,
            spo2_at_end=spo2_end,
        )


def desaturation_rate(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    time_points: list[float],
    dt: float = 0.5,
    t_max: float = 800.0,
) -> list[DesatRatePoint]:
    """Compute instantaneous desaturation rate at specified times.

    Args:
        predict_fn:  Function that takes time array and returns SpO2 array
        time_points: Times at which to compute the rate (seconds)
        dt:          Time step for gradient computation
        t_max:       Max simulation time
    """
    t = np.arange(0, t_max, dt)
    spo2 = predict_fn(t)
    gradient = np.gradient(spo2, dt) * 60  # %/min

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
