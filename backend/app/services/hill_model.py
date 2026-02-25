"""
Exponential Alveolar Washout + Saturating Bohr Effect Model
============================================================

Models SpO2 desaturation during breath-hold apnea using:
- Exponential PAO2 decline (alveolar-capillary O2 equilibration)
- Saturating Bohr effect (CO2 accumulation with exponential saturation)
- Hill equation oxygen-haemoglobin dissociation curve

Model structure:
    t_eff      = max(t - lag, 0)
    PAO2(t)    = pvo2 + (pao2_0 - pvo2) * exp(-t_eff / tau_washout)
    P50_eff(t) = P50_BASE + bohr_max * (1 - exp(-t_eff / tau_bohr))
    SpO2(t)    = clip(r_offset + 100 * PAO2^n / (PAO2^n + P50_eff^n), 0, 100)

P50_BASE (26.6 mmHg) is a fixed haemoglobin biochemistry constant.
n (Hill coefficient) is fitted within a tight physiological range (2.6-3.0).
The saturating Bohr effect replaces the linear model to prevent unphysical
P50 growth at long apnea durations.
"""

from dataclasses import asdict, dataclass, fields
from typing import ClassVar

import numpy as np
from loguru import logger


@dataclass
class ApneaModelParams:
    """Fitted parameters for the apnea desaturation model.

    P50_BASE is a fixed haemoglobin constant (not fitted).
    """

    P50_BASE: ClassVar[float] = 26.6   # Baseline P50 of the ODC (mmHg)

    pao2_0: float       # Initial alveolar PO2 (mmHg)
    pvo2: float         # Mixed venous PO2, asymptotic floor (mmHg)
    tau_washout: float   # Exponential O2 washout time constant (seconds)
    n: float             # Hill coefficient (Hb cooperativity, ~2.6-3.0)
    bohr_max: float      # Maximum Bohr P50 shift (mmHg)
    tau_bohr: float      # CO2 accumulation time constant (seconds)
    lag: float           # Finger-to-arterial circulation delay (seconds)
    r_offset: float      # Constant SpO2 offset for sensor calibration (%)

    def to_dict(self) -> dict:
        """Convert to dictionary, including fixed constant for API transparency."""
        d = asdict(self)
        d["p50_base"] = self.P50_BASE
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ApneaModelParams":
        """Create from dictionary, ignoring extra keys."""
        valid_keys = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid_keys})

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "ApneaModelParams":
        """Create from parameter array (order matches dataclass fields)."""
        field_names = [f.name for f in fields(cls)]
        if len(arr) != len(field_names):
            raise ValueError(f"Expected {len(field_names)} params, got {len(arr)}")
        return cls(**dict(zip(field_names, arr, strict=True)))

    def to_array(self) -> np.ndarray:
        """Convert to parameter array (order matches dataclass fields)."""
        return np.array([getattr(self, f.name) for f in fields(self)])


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

    # Saturating Bohr effect: CO2 accumulates with exponential saturation
    p50_eff = ApneaModelParams.P50_BASE + params.bohr_max * (
        1.0 - np.exp(-t_eff / max(params.tau_bohr, 0.01))
    )

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
    p50_eff = ApneaModelParams.P50_BASE + params.bohr_max * (
        1.0 - np.exp(-t_eff / max(params.tau_bohr, 0.01))
    )

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
