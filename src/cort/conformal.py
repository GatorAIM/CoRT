"""Class-conditional split conformal prediction for binary outcomes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Return the finite-sample split-conformal quantile."""
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("scores must be a nonempty one-dimensional array")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    level = min(np.ceil((values.size + 1) * (1 - alpha)) / values.size, 1.0)
    return float(np.quantile(values, level, method="higher"))


def conformal_pvalues(scores: np.ndarray, calibration_scores: np.ndarray) -> np.ndarray:
    """Calculate smoothed conformal p-values using greater-or-equal ties."""
    values = np.asarray(scores, dtype=float)
    reference = np.sort(np.asarray(calibration_scores, dtype=float))
    if reference.ndim != 1 or reference.size == 0:
        raise ValueError("calibration_scores must be nonempty")
    left = np.searchsorted(reference, values, side="left")
    return (1 + reference.size - left) / (reference.size + 1)


@dataclass
class MondrianCalibrator:
    """Binary class-conditional calibrator using probability nonconformity."""

    alpha: float = 0.10
    class0_scores_: np.ndarray | None = None
    class1_scores_: np.ndarray | None = None
    q_class0_: float | None = None
    q_class1_: float | None = None

    def fit(self, labels: np.ndarray, probabilities: np.ndarray) -> "MondrianCalibrator":
        labels = np.asarray(labels, dtype=int)
        probabilities = np.asarray(probabilities, dtype=float)
        if labels.shape != probabilities.shape or labels.ndim != 1:
            raise ValueError("labels and probabilities must be aligned vectors")
        if not np.isin(labels, [0, 1]).all():
            raise ValueError("labels must contain only 0 and 1")
        if ((probabilities < 0) | (probabilities > 1)).any():
            raise ValueError("probabilities must be in [0, 1]")

        self.class0_scores_ = np.sort(probabilities[labels == 0])
        self.class1_scores_ = np.sort(1.0 - probabilities[labels == 1])
        self.q_class0_ = conformal_quantile(self.class0_scores_, self.alpha)
        self.q_class1_ = conformal_quantile(self.class1_scores_, self.alpha)
        return self

    def _check_fitted(self) -> None:
        if self.class0_scores_ is None or self.class1_scores_ is None:
            raise RuntimeError("fit must be called before prediction")

    def p_values(self, probabilities: np.ndarray) -> np.ndarray:
        """Return class-0 and class-1 p-values in columns 0 and 1."""
        self._check_fitted()
        probabilities = np.asarray(probabilities, dtype=float)
        p0 = conformal_pvalues(probabilities, self.class0_scores_)
        p1 = conformal_pvalues(1.0 - probabilities, self.class1_scores_)
        return np.column_stack([p0, p1])

    def predict_sets(self, probabilities: np.ndarray) -> list[frozenset[int]]:
        """Construct prediction sets with the fitted finite-sample thresholds."""
        self._check_fitted()
        probabilities = np.asarray(probabilities, dtype=float)
        include0 = probabilities <= self.q_class0_
        include1 = (1.0 - probabilities) <= self.q_class1_
        return [
            frozenset(label for label, keep in ((0, in0), (1, in1)) if keep)
            for in0, in1 in zip(include0, include1)
        ]

