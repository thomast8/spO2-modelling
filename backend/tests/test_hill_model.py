"""Tests for the apnea desaturation model."""

import numpy as np

from app.services.hill_model import (
    ApneaModelParams,
    compute_r_squared,
    hill_spo2,
    predict_spo2,
    predict_spo2_components,
)


def _default_params() -> ApneaModelParams:
    """Reasonable FL parameters for testing."""
    return ApneaModelParams(
        pao2_0=120.0,       # mmHg, full-lung pre-oxygenation
        pvo2=40.0,          # mmHg, typical mixed venous
        tau_washout=80.0,    # seconds
        n=2.7,               # Hill coefficient
        bohr_max=5.0,        # mmHg, moderate Bohr shift
        tau_bohr=120.0,      # seconds, CO2 time constant
        lag=19.0,            # seconds
        r_offset=0.0,        # no calibration bias
    )


class TestHillSpo2:
    def test_returns_50_at_p50(self):
        """SpO2 should be 50% when PaO2 equals P50."""
        result = hill_spo2(np.array([26.6]), p50=26.6, n=2.7)
        assert abs(result[0] - 50.0) < 0.01

    def test_high_pao2_near_100(self):
        """Very high PaO2 should give SpO2 near 100%."""
        result = hill_spo2(np.array([200.0]), p50=26.6, n=2.7)
        assert result[0] > 99.0

    def test_low_pao2_near_0(self):
        """Very low PaO2 should give SpO2 near 0%."""
        result = hill_spo2(np.array([0.1]), p50=26.6, n=2.7)
        assert result[0] < 1.0

    def test_monotonic_increasing(self):
        """SpO2 should increase monotonically with PaO2."""
        pao2 = np.linspace(1, 200, 100)
        spo2 = hill_spo2(pao2, p50=26.6, n=2.7)
        assert np.all(np.diff(spo2) > 0)

    def test_accepts_array_p50(self):
        """hill_spo2 should work with array p50 (for Bohr effect)."""
        pao2 = np.array([60.0, 60.0, 60.0])
        p50 = np.array([26.0, 30.0, 34.0])
        result = hill_spo2(pao2, p50=p50, n=2.7)
        # Higher P50 -> lower SpO2 at same PaO2
        assert result[0] > result[1] > result[2]


class TestPredictSpo2:
    def test_starts_near_100(self):
        """SpO2 at t=0 should be near 100% for FL holds."""
        params = _default_params()
        t = np.array([0.0])
        result = predict_spo2(t, params)
        assert result[0] > 95.0

    def test_decreases_over_time(self):
        """SpO2 should decrease as PAO2 declines."""
        params = _default_params()
        t = np.array([0.0, 120.0, 240.0, 360.0])
        result = predict_spo2(t, params)
        assert result[-1] < result[0]

    def test_plateau_then_drop(self):
        """SpO2 should stay high during plateau, then drop steeply.

        This is the key behavioral test: exponential washout + Hill sigmoid
        should naturally produce a flat plateau followed by steep desaturation.
        """
        params = _default_params()
        t = np.arange(0, 400, 1.0)
        spo2 = predict_spo2(t, params)

        # During lag + early plateau, SpO2 should stay very high
        plateau_mask = t < 60
        assert np.all(spo2[plateau_mask] > 95.0), "SpO2 should stay >95% during early plateau"

        # At late times, SpO2 should have dropped significantly
        assert spo2[-1] < 70.0, "SpO2 should drop well below 70% by t=400"

    def test_r_offset_shifts_curve(self):
        """Positive r_offset should shift entire curve up."""
        params_base = _default_params()
        params_shifted = ApneaModelParams.from_dict({**params_base.to_dict(), "r_offset": 2.0})

        t = np.array([60.0, 120.0, 180.0])
        base = predict_spo2(t, params_base)
        shifted = predict_spo2(t, params_shifted)

        assert np.all(shifted >= base - 0.01)

    def test_bohr_effect_accelerates_late_desaturation(self):
        """Positive bohr_max should cause lower SpO2 at late times."""
        base = _default_params().to_dict()
        params_no_bohr = ApneaModelParams.from_dict({**base, "bohr_max": 0.0})
        params_bohr = ApneaModelParams.from_dict({**base, "bohr_max": 8.0})

        t = np.array([200.0, 300.0])
        spo2_no_bohr = predict_spo2(t, params_no_bohr)
        spo2_bohr = predict_spo2(t, params_bohr)

        # Bohr effect should lower SpO2 at late times
        assert np.all(spo2_bohr < spo2_no_bohr)

    def test_clipped_to_0_100(self):
        """Output should always be between 0 and 100."""
        params = ApneaModelParams.from_dict({**_default_params().to_dict(), "r_offset": 50.0})
        t = np.linspace(0, 600, 100)
        result = predict_spo2(t, params)
        assert np.all(result >= 0.0)
        assert np.all(result <= 100.0)


class TestPredictComponents:
    def test_base_plus_offset_equals_total(self):
        """base + r_offset should approximate total (before clipping)."""
        params = _default_params()
        t = np.linspace(0, 300, 50)
        components = predict_spo2_components(t, params)

        expected = np.clip(components["base"] + params.r_offset, 0, 100)
        np.testing.assert_allclose(components["total"], expected, atol=0.01)

    def test_pao2_decreases(self):
        """PAO2 should decrease over time (after lag)."""
        params = _default_params()
        t = np.linspace(20, 300, 50)
        components = predict_spo2_components(t, params)
        assert np.all(np.diff(components["pao2"]) <= 0)

    def test_pao2_decays_toward_pvo2(self):
        """PAO2 should decay exponentially toward pvo2."""
        params = _default_params()
        t = np.linspace(20, 600, 100)
        components = predict_spo2_components(t, params)
        pao2 = components["pao2"]

        # All values should be >= pvo2
        assert np.all(pao2 >= params.pvo2 - 0.01)
        # Late values should approach pvo2
        assert pao2[-1] < params.pvo2 + 2.0

    def test_p50_eff_increases(self):
        """P50_eff should increase over time (Bohr effect)."""
        params = _default_params()
        t = np.linspace(20, 300, 50)
        components = predict_spo2_components(t, params)
        assert np.all(np.diff(components["p50_eff"]) >= 0)


class TestApneaModelParamsConversion:
    def test_roundtrip_dict(self):
        """to_dict -> from_dict should be lossless."""
        params = _default_params()
        reconstructed = ApneaModelParams.from_dict(params.to_dict())
        assert params == reconstructed

    def test_roundtrip_array(self):
        """to_array -> from_array should be lossless."""
        params = _default_params()
        arr = params.to_array()
        reconstructed = ApneaModelParams.from_array(arr)
        for field_name in [f.name for f in __import__("dataclasses").fields(ApneaModelParams)]:
            assert abs(getattr(params, field_name) - getattr(reconstructed, field_name)) < 1e-10

    def test_from_dict_ignores_extra_keys(self):
        """from_dict should ignore keys not in the dataclass."""
        d = {**_default_params().to_dict(), "extra_key": 42}
        params = ApneaModelParams.from_dict(d)
        assert params == _default_params()


class TestRSquared:
    def test_perfect_fit(self):
        """R² should be 1.0 for perfect fit."""
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        assert compute_r_squared(obs, obs) == 1.0

    def test_zero_for_mean_prediction(self):
        """R² should be 0 if prediction is the mean."""
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        pred = np.full_like(obs, np.mean(obs))
        assert abs(compute_r_squared(obs, pred)) < 0.01

    def test_constant_observed(self):
        """R² should be 0 for constant observed data."""
        obs = np.array([5.0, 5.0, 5.0])
        pred = np.array([4.0, 5.0, 6.0])
        assert compute_r_squared(obs, pred) == 0.0
