"""
Model Fitting Engine
====================

Fits Hill model parameters to SpO2 data using scipy's differential evolution
optimizer. Supports fitting across multiple holds simultaneously with
per-hold-type parameter bounds.
"""

from dataclasses import dataclass, field

import numpy as np
from loguru import logger
from scipy.optimize import differential_evolution

from app.services.hill_model import HillParams, compute_r_squared, predict_spo2

# Default parameter bounds per hold type
# Format: {param_name: (lower, upper)}
DEFAULT_BOUNDS: dict[str, dict[str, tuple[float, float]]] = {
    "FRC": {
        "o2_start": (800, 1500),     # Smaller lung volume
        "vo2": (100, 300),
        "scale": (5, 50),
        "p50": (15, 60),
        "n": (2.0, 4.0),
        "r_offset": (-5.0, 5.0),
        "r_decay": (-10.0, 10.0),
        "tau_decay": (10, 60),
        "lag": (10, 30),
    },
    "RV": {
        "o2_start": (400, 1000),     # Minimal lung volume
        "vo2": (100, 300),
        "scale": (5, 50),
        "p50": (15, 60),
        "n": (2.0, 4.0),
        "r_offset": (-5.0, 5.0),
        "r_decay": (-10.0, 10.0),
        "tau_decay": (10, 60),
        "lag": (10, 30),
    },
    "FL": {
        "o2_start": (1800, 2800),    # Full lungs
        "vo2": (100, 300),
        "scale": (5, 50),
        "p50": (15, 60),
        "n": (2.0, 4.0),
        "r_offset": (-5.0, 5.0),
        "r_decay": (-10.0, 10.0),
        "tau_decay": (20, 90),
        "lag": (10, 30),
    },
}

# Parameter names in order (matches HillParams dataclass field order)
PARAM_NAMES = list(HillParams.__dataclass_fields__.keys())


@dataclass
class FitResult:
    """Result of a model fitting run."""

    params: HillParams
    r_squared: float
    r_squared_per_hold: list[float]
    objective_val: float
    converged: bool
    n_holds: int
    hold_ids: list[int]
    n_data_points: int
    # Predicted curves for each hold (for preview)
    predictions: list[dict] = field(default_factory=list)


def get_bounds(
    hold_type: str,
    overrides: dict[str, tuple[float, float]] | None = None,
) -> list[tuple[float, float]]:
    """Get parameter bounds for a hold type, with optional overrides.

    Args:
        hold_type: One of 'FRC', 'RV', 'FL'
        overrides: Optional dict of {param_name: (lower, upper)} to override defaults

    Returns:
        List of (lower, upper) tuples in parameter order
    """
    if hold_type not in DEFAULT_BOUNDS:
        raise ValueError(f"Unknown hold type: {hold_type!r}. Must be one of {list(DEFAULT_BOUNDS)}")

    bounds = dict(DEFAULT_BOUNDS[hold_type])
    if overrides:
        for name, (lo, hi) in overrides.items():
            if name not in bounds:
                logger.warning(f"Ignoring unknown parameter override: {name}")
                continue
            bounds[name] = (lo, hi)

    return [bounds[name] for name in PARAM_NAMES]


def fit_holds(
    hold_data: list[dict],
    hold_type: str,
    bounds_override: dict[str, tuple[float, float]] | None = None,
    seed: int = 42,
    maxiter: int = 5000,
    popsize: int = 60,
) -> FitResult:
    """Fit Hill model to one or more holds of the same type.

    Args:
        hold_data: List of dicts, each with keys:
            - 'id': hold database ID
            - 'elapsed_s': numpy array of time points
            - 'spo2': numpy array of SpO2 values
        hold_type: One of 'FRC', 'RV', 'FL'
        bounds_override: Optional parameter bound overrides
        seed: Random seed for reproducibility
        maxiter: Maximum optimizer iterations
        popsize: Population size for differential evolution

    Returns:
        FitResult with fitted parameters and diagnostics
    """
    if not hold_data:
        raise ValueError("No hold data provided for fitting")

    bounds = get_bounds(hold_type, bounds_override)
    hold_ids = [h["id"] for h in hold_data]
    total_points = sum(len(h["elapsed_s"]) for h in hold_data)

    logger.info(
        f"Fitting {hold_type} model to {len(hold_data)} holds "
        f"({total_points} data points), seed={seed}"
    )

    def objective(param_array: np.ndarray) -> float:
        params = HillParams.from_array(param_array)
        total_error = 0.0
        for hold in hold_data:
            pred = predict_spo2(hold["elapsed_s"], params)
            total_error += np.sum((hold["spo2"] - pred) ** 2)
        return total_error

    result = differential_evolution(
        objective,
        bounds,
        maxiter=maxiter,
        seed=seed,
        tol=1e-12,
        polish=True,
        popsize=popsize,
        mutation=(0.5, 1.5),
        recombination=0.9,
    )

    fitted_params = HillParams.from_array(result.x)

    # Compute R² for each hold and predictions
    r2_per_hold = []
    predictions = []
    for hold in hold_data:
        pred = predict_spo2(hold["elapsed_s"], fitted_params)
        r2 = compute_r_squared(hold["spo2"], pred)
        r2_per_hold.append(r2)
        predictions.append({
            "hold_id": hold["id"],
            "elapsed_s": hold["elapsed_s"].tolist(),
            "observed": hold["spo2"].tolist(),
            "predicted": pred.tolist(),
            "r_squared": r2,
        })

    # Overall R²
    all_observed = np.concatenate([h["spo2"] for h in hold_data])
    all_predicted = np.concatenate([
        predict_spo2(h["elapsed_s"], fitted_params) for h in hold_data
    ])
    overall_r2 = compute_r_squared(all_observed, all_predicted)

    logger.info(
        f"Fit complete: R²={overall_r2:.4f}, converged={result.success}, "
        f"objective={result.fun:.2f}"
    )
    logger.info(f"  Params: {fitted_params}")

    return FitResult(
        params=fitted_params,
        r_squared=overall_r2,
        r_squared_per_hold=r2_per_hold,
        objective_val=float(result.fun),
        converged=result.success,
        n_holds=len(hold_data),
        hold_ids=hold_ids,
        n_data_points=total_points,
        predictions=predictions,
    )
