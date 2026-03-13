"""
Model Fitting Engine
====================

Fits apnea model parameters to SpO2 data using scipy's differential evolution
optimizer. Supports fitting across multiple holds simultaneously with
per-hold-type parameter bounds.
"""

from dataclasses import dataclass, field, fields

import numpy as np
from loguru import logger
from scipy.optimize import differential_evolution

from app.services.hill_model import ApneaModelParams, compute_r_squared, predict_spo2

# Default parameter bounds per hold type
# Format: {param_name: (lower, upper)}
DEFAULT_BOUNDS: dict[str, dict[str, tuple[float, float]]] = {
    "FRC": {
        "pao2_0": (80, 120),        # Smaller lung volume -> lower initial PAO2
        "pvo2": (20, 50),           # Can drop below 30 in prolonged apnea
        "tau_washout": (20, 100),    # Faster washout with less O2 reserve
        "gamma": (0.8, 2.0),       # Steepness exponent; 1.0 = standard Severinghaus
        "bohr_max": (2.0, 15.0),   # Max Bohr P50 shift (mmHg); up to ~15 at respiratory acidosis
        "tau_bohr": (40, 250),      # CO2 time constant (s); ~80-150 physiological
        "r_offset": (-3.0, 3.0),
    },
    "RV": {
        "pao2_0": (70, 110),        # Minimal lung volume
        "pvo2": (20, 50),           # Can drop below 30 in prolonged apnea
        "tau_washout": (10, 80),     # Fastest washout
        "gamma": (0.8, 2.0),       # Steepness exponent; 1.0 = standard Severinghaus
        "bohr_max": (2.0, 15.0),   # Max Bohr P50 shift (mmHg); up to ~15 at respiratory acidosis
        "tau_bohr": (40, 250),      # CO2 time constant (s); ~80-150 physiological
        "r_offset": (-3.0, 3.0),
    },
    "FL": {
        "pao2_0": (100, 250),       # Full lungs; hyperventilation → PAO2 140-200+
        "pvo2": (20, 50),           # Asymptotic PvO2; drops below 30 in prolonged apnea
        "tau_washout": (50, 250),    # Slowest washout, largest O2 reserve
        "gamma": (0.8, 2.0),       # Steepness exponent; 1.0 = standard Severinghaus
        "bohr_max": (2.0, 15.0),   # Max Bohr P50 shift (mmHg); up to ~15 at respiratory acidosis
        "tau_bohr": (40, 250),      # CO2 time constant (s); ~80-150 physiological
        "r_offset": (-3.0, 3.0),
    },
}

# Parameter names in order (matches ApneaModelParams dataclass field order, excluding ClassVar)
PARAM_NAMES = [f.name for f in fields(ApneaModelParams)]


@dataclass
class FitResult:
    """Result of a model fitting run."""

    params: ApneaModelParams
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
    """Fit apnea model to one or more holds of the same type.

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
        params = ApneaModelParams.from_array(param_array)
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

    fitted_params = ApneaModelParams.from_array(result.x)

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
