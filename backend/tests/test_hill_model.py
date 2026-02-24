"""Tests for the Hill model."""

import numpy as np

from app.services.hill_model import (
    HillParams,
    compute_r_squared,
    hill_spo2,
    predict_spo2,
    predict_spo2_components,
)


def _default_params() -> HillParams:
    """Reasonable FL parameters for testing."""
    return HillParams(
        o2_start=1966.0,
        vo2=220.0,
        scale=12.8,
        p50=50.7,
        n=4.0,
        r_offset=0.0,
        r_decay=0.0,
        tau_decay=30.0,
        lag=19.0,
    )


class TestHillSpo2:
    def test_returns_50_at_p50(self):
        """SpO2 should be 50% when PaO2_eff equals P50."""
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


class TestPredictSpo2:
    def test_starts_near_100(self):
        """SpO2 at t=0 should be near 100% for FL holds."""
        params = _default_params()
        t = np.array([0.0])
        result = predict_spo2(t, params)
        assert result[0] > 95.0

    def test_decreases_over_time(self):
        """SpO2 should decrease as O2 is consumed."""
        params = _default_params()
        t = np.array([0.0, 120.0, 240.0, 360.0])
        result = predict_spo2(t, params)
        # After lag, SpO2 should decrease
        assert result[-1] < result[0]

    def test_residual_offset_shifts_curve(self):
        """Positive r_offset should shift entire curve up."""
        params_base = _default_params()
        params_shifted = HillParams(**{**params_base.to_dict(), "r_offset": 2.0})

        t = np.array([60.0, 120.0, 180.0])
        base = predict_spo2(t, params_base)
        shifted = predict_spo2(t, params_shifted)

        # Shifted should be higher (within clip bounds)
        assert np.all(shifted >= base - 0.01)

    def test_residual_decay_affects_early_times(self):
        """r_decay should have more effect at early times."""
        params = HillParams(**{**_default_params().to_dict(), "r_decay": 5.0})
        t = np.array([1.0, 300.0])
        spo2 = predict_spo2(t, params)
        # The effect should be larger at t=1 than t=300
        base = predict_spo2(t, _default_params())
        diff_early = spo2[0] - base[0]
        diff_late = spo2[1] - base[1]
        assert diff_early > diff_late

    def test_clipped_to_0_100(self):
        """Output should always be between 0 and 100."""
        params = HillParams(**{**_default_params().to_dict(), "r_offset": 50.0})
        t = np.linspace(0, 600, 100)
        result = predict_spo2(t, params)
        assert np.all(result >= 0.0)
        assert np.all(result <= 100.0)


class TestPredictComponents:
    def test_components_sum_to_total(self):
        """Base + residual should approximate total (before clipping)."""
        params = _default_params()
        t = np.linspace(0, 300, 50)
        components = predict_spo2_components(t, params)

        # Total = clip(base + residual, 0, 100)
        expected = np.clip(components["base"] + components["residual"], 0, 100)
        np.testing.assert_allclose(components["total"], expected, atol=0.01)

    def test_o2_remaining_decreases(self):
        """O2 remaining should decrease over time (after lag)."""
        params = _default_params()
        t = np.linspace(20, 300, 50)
        components = predict_spo2_components(t, params)
        assert np.all(np.diff(components["o2_remaining"]) <= 0)


class TestHillParamsConversion:
    def test_roundtrip_dict(self):
        """to_dict -> from_dict should be lossless."""
        params = _default_params()
        reconstructed = HillParams.from_dict(params.to_dict())
        assert params == reconstructed

    def test_roundtrip_array(self):
        """to_array -> from_array should be lossless."""
        params = _default_params()
        arr = params.to_array()
        reconstructed = HillParams.from_array(arr)
        for field_name in HillParams.__dataclass_fields__:
            assert abs(getattr(params, field_name) - getattr(reconstructed, field_name)) < 1e-10

    def test_from_dict_ignores_extra_keys(self):
        """from_dict should ignore keys not in the dataclass."""
        d = {**_default_params().to_dict(), "extra_key": 42}
        params = HillParams.from_dict(d)
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
