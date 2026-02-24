"""
Exponential Alveolar Washout + Bohr Effect Model
=================================================

Models SpO2 desaturation during breath-hold apnea using:
- Exponential PAO2 decline (alveolar-capillary O2 equilibration)
- Dynamic P50 via the Bohr effect (CO2 accumulation during apnea)
- Hill equation oxygen-haemoglobin dissociation curve

Model structure:
    t_eff      = max(t - lag, 0)
    PAO2(t)    = pvo2 + (pao2_0 - pvo2) * exp(-t_eff / tau_washout)
    P50_eff(t) = p50_base + bohr_coeff * t_eff
    SpO2(t)    = clip(r_offset + 100 * PAO2^n / (PAO2^n + P50_eff^n), 0, 100)

The exponential washout naturally produces a flat SpO2 plateau (while PAO2
remains on the upper shoulder of the ODC) followed by a steep desaturation
(as PAO2 enters the mid-range). The Bohr effect accelerates late desaturation
by progressively right-shifting the curve.
"""

from dataclasses import asdict, dataclass

import numpy as np
from loguru import logger


@dataclass
class ApneaModelParams:
    """Fitted parameters for the apnea desaturation model."""

    pao2_0: float       # Initial alveolar PO2 (mmHg)
    pvo2: float         # Mixed venous PO2, asymptotic floor (mmHg)
    tau_washout: float   # Exponential O2 washout time constant (seconds)
    p50_base: float      # Baseline P50 of the ODC (mmHg)
    n: float             # Hill coefficient (Hb cooperativity)
    bohr_coeff: float    # P50 shift rate during apnea (mmHg/s)
    lag: float           # Finger-to-arterial circulation delay (seconds)
    r_offset: float      # Constant SpO2 offset for sensor calibration (%)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ApneaModelParams":
        """Create from dictionary, ignoring extra keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid_keys})

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "ApneaModelParams":
        """Create from parameter array (order matches dataclass fields)."""
        fields = list(cls.__dataclass_fields__.keys())
        if len(arr) != len(fields):
            raise ValueError(f"Expected {len(fields)} params, got {len(arr)}")
        return cls(**dict(zip(fields, arr)))

    def to_array(self) -> np.ndarray:
        """Convert to parameter array (order matches dataclass fields)."""
        return np.array([getattr(self, f) for f in self.__dataclass_fields__])


def hill_spo2(
    pao2: np.ndarray,
    p50: np.ndarray | float,
    n: float,
) -> np.ndarray:
    """Oxygen-haemoglobin dissociation curve (Hill equation).

    Returns SpO2 in % given PaO2 (mmHg) and P50 (mmHg).
    Both pao2 and p50 can be arrays (numpy broadcasting applies).
    """
    pao2 = np.maximum(pao2, 0.01)
    p50 = np.maximum(p50, 0.01)
    return 100.0 * (pao2**n) / (pao2**n + p50**n)


def predict_spo2(t: np.ndarray, params: ApneaModelParams) -> np.ndarray:
    """Predict finger SpO2 at times t (seconds from hold start).

    Args:
        t:      Time array in seconds
        params: Fitted model parameters

    Returns:
        Predicted SpO2 (%) array, same shape as t
    """
    t_eff = np.maximum(t - params.lag, 0.0)

    # Exponential PAO2 decline (alveolar-capillary equilibration)
    pao2 = params.pvo2 + (params.pao2_0 - params.pvo2) * np.exp(
        -t_eff / max(params.tau_washout, 0.01)
    )

    # Bohr effect: CO2 accumulates -> P50 increases
    p50_eff = params.p50_base + params.bohr_coeff * t_eff

    # Hill equation with dynamic P50
    spo2 = params.r_offset + hill_spo2(pao2, p50_eff, params.n)
    return np.clip(spo2, 0.0, 100.0)


def predict_spo2_components(
    t: np.ndarray,
    params: ApneaModelParams,
) -> dict[str, np.ndarray]:
    """Predict SpO2 with decomposed components for visualization.

    Returns dict with keys: total, base, pao2, p50_eff
    """
    t_eff = np.maximum(t - params.lag, 0.0)

    pao2 = params.pvo2 + (params.pao2_0 - params.pvo2) * np.exp(
        -t_eff / max(params.tau_washout, 0.01)
    )
    p50_eff = params.p50_base + params.bohr_coeff * t_eff

    spo2_base = hill_spo2(pao2, p50_eff, params.n)
    spo2_total = np.clip(spo2_base + params.r_offset, 0.0, 100.0)

    return {
        "total": spo2_total,
        "base": spo2_base,
        "pao2": pao2,
        "p50_eff": p50_eff,
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
