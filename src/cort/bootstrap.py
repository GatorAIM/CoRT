"""Paired bootstrap utilities that preserve cross-predictor dependence."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .conformal import MondrianCalibrator
from .routing import route_prediction_sets


def evaluate_routing(
    calibration_labels: np.ndarray,
    calibration_probabilities: dict[str, np.ndarray],
    test_probabilities: dict[str, np.ndarray],
    alpha: float,
) -> dict[str, float | int]:
    """Refit both calibrators and calculate pathway allocation."""
    required = {"tabular", "longitudinal"}
    if set(calibration_probabilities) != required or set(test_probabilities) != required:
        raise ValueError(f"probability mappings must have keys {sorted(required)}")

    prediction_sets = {}
    for view in sorted(required):
        calibrator = MondrianCalibrator(alpha=alpha).fit(
            calibration_labels, calibration_probabilities[view]
        )
        prediction_sets[view] = calibrator.predict_sets(test_probabilities[view])

    decisions = [
        route_prediction_sets(tabular, longitudinal)
        for tabular, longitudinal in zip(
            prediction_sets["tabular"], prediction_sets["longitudinal"]
        )
    ]
    counts = {state: sum(decision.state.value == state for decision in decisions) for state in "ABCDEF"}
    directional = sum(counts[state] for state in "ABCD")
    return {
        **{f"state_{state}": count for state, count in counts.items()},
        "n_directional": directional,
        "n_review": len(decisions) - directional,
        "directional_coverage": directional / len(decisions),
    }


def paired_calibration_test_bootstrap(
    calibration_labels: np.ndarray,
    calibration_probabilities: dict[str, np.ndarray],
    test_probabilities: dict[str, np.ndarray],
    *,
    alpha: float = 0.10,
    n_resamples: int = 1000,
    seed: int = 0,
    evaluator: Callable[..., dict] = evaluate_routing,
) -> list[dict]:
    """Resample shared calibration and test indices for both predictors."""
    calibration_labels = np.asarray(calibration_labels, dtype=int)
    if np.unique(calibration_labels).size != 2:
        raise ValueError("calibration_labels must contain both outcome classes")
    n_calibration = calibration_labels.size
    n_test = len(next(iter(test_probabilities.values())))
    if any(len(value) != n_calibration for value in calibration_probabilities.values()):
        raise ValueError("calibration probabilities must align with calibration_labels")
    if any(len(value) != n_test for value in test_probabilities.values()):
        raise ValueError("test probability arrays must have the same length")
    rng = np.random.default_rng(seed)
    rows = []

    for replicate in range(n_resamples):
        for _ in range(1000):
            calibration_index = rng.integers(0, n_calibration, n_calibration)
            if np.unique(calibration_labels[calibration_index]).size == 2:
                break
        else:
            raise RuntimeError("could not draw a calibration resample containing both classes")
        test_index = rng.integers(0, n_test, n_test)
        rows.append(
            {
                "replicate": replicate + 1,
                **evaluator(
                    calibration_labels[calibration_index],
                    {key: np.asarray(value)[calibration_index] for key, value in calibration_probabilities.items()},
                    {key: np.asarray(value)[test_index] for key, value in test_probabilities.items()},
                    alpha,
                ),
            }
        )
    return rows
