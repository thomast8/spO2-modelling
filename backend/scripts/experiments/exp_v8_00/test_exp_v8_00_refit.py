"""Summary checks on synthetic results. Run from backend/:

uv run --extra dev pytest scripts/experiments/exp_v8_00/test_exp_v8_00_refit.py
"""

from exp_v8_00_refit import summarise


def _result(group, converged_a, tau_washout, globals_from=None):
    return {
        "group": group,
        "globals_from": globals_from,
        "n_holds": 1,
        "n_points": 100,
        "stage_a": {"converged": converged_a, "tau_0": 20.0, "p": 3.0},
        "stage_b": {"converged": True, "k_co2": 0.1, "gamma": 2.0},
        "holds": [
            {
                "id": f"{group}-RV1",
                "type": "RV",
                "tau_washout": tau_washout,
                "tau_washout_at_bound": tau_washout == 80,
                "paco2_0_at_bound": False,
                "r2a_stage_b": 0.9,
                "rmse_apnea_stage_b": 1.0,
                "nadir_spo2_err_stage_b": -2.0,
            }
        ],
    }


def test_summary_marks_unconverged_fits_and_pairs_transfers(capsys):
    summarise(
        [
            _result("2026-10-02", False, 45.0),
            _result("2026-10-02", False, 80.0, globals_from="2026-02-21"),
        ]
    )
    out = capsys.readouterr().out.splitlines()
    drift = [line for line in out if line.startswith("2026-10-02    ")]
    assert len(drift) == 2 and all("no (A)" in line for line in drift)
    (transfer,) = [line for line in out if line.startswith("2026-10-02  2026-02-21")]
    assert "45 -> 80* (RV)" in transfer
