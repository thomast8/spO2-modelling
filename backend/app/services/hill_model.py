"""
Exponential Alveolar Washout + CO2-Driven Bohr Effect Model
============================================================

Models SpO2 desaturation during breath-hold apnea using:
- Exponential PAO2 decline (alveolar-capillary O2 equilibration)
- CO2-driven Bohr effect (linear PaCO2 rise with physiological Bohr coefficient)
- Severinghaus (1979) oxygen-haemoglobin dissociation curve with gamma steepness

Model structure:
    PAO2(t)      = pvo2 + (pao2_0 - pvo2) * exp(-t / tau_washout)
    PaCO2(t)     = paco2_0 + k_co2 * t                  [linear CO2 rise]
    P50_eff(t)   = P50_BASE + BOHR_BETA * (PaCO2 - 40)  [Bohr shift]
    PAO2_virtual = PAO2 * (P50_BASE / P50_eff)           [virtual PO2]
    PAO2_adj     = P50_BASE * (PAO2_virtual / P50_BASE)^gamma  [steepness]
    SpO2(t)      = clip(r_offset + 100 / (1 + 23400/(PAO2_adj^3 + 150*PAO2_adj)), 0, 100)

P50_BASE (26.6 mmHg) is a fixed haemoglobin biochemistry constant.
BOHR_BETA (0.48 mmHg/mmHg) is the Bohr coefficient from literature.
gamma (steepness exponent) is fitted — 1.0 = standard Severinghaus, >1 = steeper.
"""

from dataclasses import asdict, dataclass, fields
from typing import ClassVar

import numpy as np
from loguru import logger


@dataclass
class ApneaModelParams:
    """Fitted parameters for the apnea desaturation model.

    P50_BASE and BOHR_BETA are fixed physiological constants (not fitted).
    """

    P50_BASE: ClassVar[float] = 26.6    # Baseline P50 of the ODC (mmHg)
    BOHR_BETA: ClassVar[float] = 0.48   # Bohr coefficient: mmHg P50 shift per mmHg CO2

    pao2_0: float       # Initial alveolar PO2 (mmHg)
    pvo2: float         # Mixed venous PO2, asymptotic floor (mmHg)
    tau_washout: float   # Exponential O2 washout time constant (seconds)
    gamma: float         # Steepness exponent (1.0 = standard Severinghaus, >1 steeper)
    paco2_0: float       # Initial arterial CO2 (mmHg); low after hyperventilation
    k_co2: float         # CO2 rise rate (mmHg/s); ~0.05 = 3 mmHg/min
    r_offset: float      # Constant SpO2 offset for sensor calibration (%)

    def to_dict(self) -> dict:
        """Convert to dictionary, including fixed constants for API transparency."""
        d = asdict(self)
        d["p50_base"] = self.P50_BASE
        d["bohr_beta"] = self.BOHR_BETA
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


def severinghaus_spo2(pao2: np.ndarray) -> np.ndarray:
    """Severinghaus (1979) ODC: PO2 → SpO2 (%).

    SO2 = 100 / (1 + 23400/(PO2^3 + 150*PO2))

    Standard empirical formula for the oxygen-haemoglobin dissociation curve.
    Naturally captures the asymmetric shape (steeper in 40-80 mmHg range).
    """
    x = np.maximum(pao2, 0.01)
    return np.clip(100.0 / (1.0 + 23400.0 / (x**3 + 150.0 * x)), 0.0, 100.0)


def predict_spo2(t: np.ndarray, params: ApneaModelParams) -> np.ndarray:
    """Predict finger SpO2 at times t (seconds from hold start).

    Args:
        t:      Time array in seconds
        params: Fitted model parameters

    Returns:
        Predicted SpO2 (%) array, same shape as t
    """
    # Exponential PAO2 decline (alveolar-capillary equilibration)
    pao2 = params.pvo2 + (params.pao2_0 - params.pvo2) * np.exp(
        -t / max(params.tau_washout, 0.01)
    )

    # CO2-driven Bohr effect: linear PaCO2 rise with physiological coefficient
    paco2 = params.paco2_0 + params.k_co2 * t
    p50_eff = ApneaModelParams.P50_BASE + ApneaModelParams.BOHR_BETA * (paco2 - 40.0)

    # Virtual PO2 for Bohr effect: shifts the ODC
    pao2_virtual = pao2 * (ApneaModelParams.P50_BASE / np.maximum(p50_eff, 1.0))

    # Power transform for steepness: preserves P50 crossing, adjusts slope
    pao2_adj = ApneaModelParams.P50_BASE * (
        (np.maximum(pao2_virtual, 0.01) / ApneaModelParams.P50_BASE) ** params.gamma
    )

    # Severinghaus equation with steepness-adjusted PO2
    spo2 = params.r_offset + severinghaus_spo2(pao2_adj)
    return np.clip(spo2, 0.0, 100.0)


def predict_spo2_components(
    t: np.ndarray,
    params: ApneaModelParams,
) -> dict[str, np.ndarray]:
    """Predict SpO2 with decomposed components for visualization.

    Returns dict with keys: total, base, pao2, p50_eff
    """
    pao2 = params.pvo2 + (params.pao2_0 - params.pvo2) * np.exp(
        -t / max(params.tau_washout, 0.01)
    )
    paco2 = params.paco2_0 + params.k_co2 * t
    p50_eff = ApneaModelParams.P50_BASE + ApneaModelParams.BOHR_BETA * (paco2 - 40.0)

    # Virtual PO2 for Bohr effect + power transform for steepness
    pao2_virtual = pao2 * (ApneaModelParams.P50_BASE / np.maximum(p50_eff, 1.0))
    pao2_adj = ApneaModelParams.P50_BASE * (
        (np.maximum(pao2_virtual, 0.01) / ApneaModelParams.P50_BASE) ** params.gamma
    )

    spo2_base = severinghaus_spo2(pao2_adj)
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
