"""Checks on the v8.01 lag measures. Run from backend/:

uv run --extra dev pytest scripts/experiments/exp_v8_01/test_exp_v8_01_sensor.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_v8_01_sensor import boot_median, recovery_coverage, recovery_onset, resample_sessions


def test_onset_follows_the_final_nadir():
    # A bounce to 82 at 2 s, then a lower nadir of 78 (last at 7 s), then the real rise.
    t = np.arange(1.0, 13)
    s = np.array([80, 82, 81, 80, 79, 78, 78, 79, 80, 81, 85, 90], float)
    assert recovery_onset(t, s, 0.0) == 9.0  # first reading >= 80 after t=7
    assert recovery_onset(t[:8], s[:8], 0.0) is None  # trace ends before the rise


def test_coverage_is_elapsed_time_not_sample_count():
    h = {"t_end": 100.0, "t_recovery": np.array([102.0, 110.0, 160.0])}
    assert recovery_coverage(h) == 60.0
    assert recovery_coverage({"t_end": 1.0, "t_recovery": np.array([])}) == 0.0


def test_bootstrap_resamples_whole_sessions():
    sessions = np.array(["a", "a", "a", "b"])
    for _ in range(50):
        idx = resample_sessions(sessions)
        # whole sessions only: "a" contributes 0, 3 or 6 holds, never 1 or 2
        assert (sessions[idx] == "a").sum() % 3 == 0
    med, lo, hi = boot_median([1, 2, 3, 4], sessions)
    assert np.isnan(lo) and np.isnan(hi)  # two sessions: no interval claimed
