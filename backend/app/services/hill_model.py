"""
Hill Equation Oxygen-Haemoglobin Dissociation Curve Model
=========================================================

Models SpO2 desaturation during breath-hold apnea using the Hill equation
with a residual correction term that subsumes sensor bias and transient effects.

Model structure:
    O2(t)      = O2_start - (VO2 / 60) * max(t - lag, 0)
    PaO2_eff   = O2(t) / scale
    SpO2_base  = 100 * PaO2_eff^n / (PaO2_eff^n + P50^n)    [Hill equation]
    SpO2(t)    = SpO2_base + r_offset + r_decay * exp(-t / tau_decay)

The residual correction replaces the previous arm_offset approach with:
    r_offset   - constant SpO2 bias (sensor placement, calibration drift)
    r_decay    - transient correction amplitude (initial ischemic response)
    tau_decay  - time constant for transient decay (seconds)
"""

from dataclasses import asdict, dataclass

import numpy as np
from loguru import logger


@dataclass
class HillParams:
    """Fitted parameters for the Hill desaturation model."""

    o2_start: float    # Total O2 stores at hold start (mL)
    vo2: float         # O2 consumption rate (mL/min)
    scale: float       # mL O2 -> PaO2-equivalent conversion factor
    p50: float         # PaO2_eff at which SpO2 = 50%
    n: float           # Hill coefficient (Hb cooperativity)
    r_offset: float    # Residual constant offset (SpO2 %)
    r_decay: float     # Residual transient amplitude (SpO2 %)
    tau_decay: float   # Residual decay time constant (seconds)
    lag: float         # Finger-to-arterial delay (seconds)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "HillParams":
        """Create from dictionary, ignoring extra keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid_keys})

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "HillParams":
        """Create from parameter array (order matches dataclass fields)."""
        fields = list(cls.__dataclass_fields__.keys())
        if len(arr) != len(fields):
            raise ValueError(f"Expected {len(fields)} params, got {len(arr)}")
        return cls(**dict(zip(fields, arr)))

    def to_array(self) -> np.ndarray:
        """Convert to parameter array (order matches dataclass fields)."""
        return np.array([getattr(self, f) for f in self.__dataclass_fields__])


def hill_spo2(pao2_eff: np.ndarray, p50: float, n: float) -> np.ndarray:
    """Oxygen-haemoglobin dissociation curve (Hill equation).

    Returns SpO2 in % given effective PaO2 units.
    Asymmetric: flat plateau at high PaO2, steep in mid-range.
    """
    pao2_eff = np.maximum(pao2_eff, 0.01)
    return 100.0 * (pao2_eff**n) / (pao2_eff**n + p50**n)


def predict_spo2(t: np.ndarray, params: HillParams) -> np.ndarray:
    """Predict finger SpO2 at times t (seconds from hold start).

    Args:
        t:      Time array in seconds
        params: Fitted model parameters

    Returns:
        Predicted SpO2 (%) array, same shape as t
    """
    # Effective time after finger lag
    t_eff = np.maximum(t - params.lag, 0.0)

    # O2 depletion
    o2_remaining = params.o2_start - (params.vo2 / 60.0) * t_eff
    o2_remaining = np.maximum(o2_remaining, 0.01)

    # Hill equation base prediction
    pao2_eff = o2_remaining / params.scale
    spo2_base = hill_spo2(pao2_eff, params.p50, params.n)

    # Residual correction: constant + decaying transient
    residual = params.r_offset + params.r_decay * np.exp(-t / max(params.tau_decay, 0.01))
    spo2 = np.clip(spo2_base + residual, 0.0, 100.0)

    return spo2


def predict_spo2_components(t: np.ndarray, params: HillParams) -> dict[str, np.ndarray]:
    """Predict SpO2 with decomposed components for visualization.

    Returns dict with keys: total, base, residual, o2_remaining, pao2_eff
    """
    t_eff = np.maximum(t - params.lag, 0.0)
    o2_remaining = params.o2_start - (params.vo2 / 60.0) * t_eff
    o2_remaining = np.maximum(o2_remaining, 0.01)

    pao2_eff = o2_remaining / params.scale
    spo2_base = hill_spo2(pao2_eff, params.p50, params.n)
    residual = params.r_offset + params.r_decay * np.exp(-t / max(params.tau_decay, 0.01))
    spo2_total = np.clip(spo2_base + residual, 0.0, 100.0)

    return {
        "total": spo2_total,
        "base": spo2_base,
        "residual": residual,
        "o2_remaining": o2_remaining,
        "pao2_eff": pao2_eff,
    }


def compute_r_squared(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Compute R-squared goodness of fit.

    Returns 0.0 if total sum of squares is zero (constant data).
    """
    ss_res = np.sum((observed - predicted) ** 2)
    ss_tot = np.sum((observed - np.mean(observed)) ** 2)
    if ss_tot == 0:
        logger.warning("R² undefined: observed data is constant")
        return 0.0
    return float(1.0 - ss_res / ss_tot)
