import numpy as np

from cort.bootstrap import paired_calibration_test_bootstrap


def test_bootstrap_returns_requested_replicates() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1])
    calibration = {
        "tabular": np.array([0.05, 0.10, 0.20, 0.70, 0.80, 0.95]),
        "longitudinal": np.array([0.10, 0.15, 0.25, 0.65, 0.85, 0.90]),
    }
    test = {
        "tabular": np.array([0.05, 0.45, 0.95, 0.55]),
        "longitudinal": np.array([0.10, 0.60, 0.90, 0.50]),
    }
    rows = paired_calibration_test_bootstrap(
        labels,
        calibration,
        test,
        alpha=0.20,
        n_resamples=5,
        seed=7,
    )
    assert len(rows) == 5
    assert all(row["n_directional"] + row["n_review"] == 4 for row in rows)

