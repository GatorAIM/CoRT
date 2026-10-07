import numpy as np

from cort.conformal import MondrianCalibrator


def test_prediction_sets_and_pvalues() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1])
    probabilities = np.array([0.05, 0.10, 0.20, 0.70, 0.80, 0.95])
    calibrator = MondrianCalibrator(alpha=0.20).fit(labels, probabilities)
    sets = calibrator.predict_sets(np.array([0.05, 0.50, 0.95]))
    pvalues = calibrator.p_values(np.array([0.05, 0.50, 0.95]))
    assert sets[0] == frozenset({0})
    assert sets[2] == frozenset({1})
    assert pvalues.shape == (3, 2)
    assert ((pvalues > 0) & (pvalues <= 1)).all()

