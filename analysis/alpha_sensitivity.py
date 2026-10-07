"""Evaluate CoRT state composition over multiple significance levels."""

from __future__ import annotations

from cort.bootstrap import evaluate_routing


def evaluate_alpha_grid(
    calibration_labels,
    calibration_probabilities,
    test_probabilities,
    alphas=(0.05, 0.10, 0.15),
):
    return [
        {
            "alpha": float(alpha),
            **evaluate_routing(
                calibration_labels,
                calibration_probabilities,
                test_probabilities,
                float(alpha),
            ),
        }
        for alpha in alphas
    ]

