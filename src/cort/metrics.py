"""Performance summaries for directional and review pathways."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score


def directional_summary(
    labels: np.ndarray,
    directions: np.ndarray,
    scores: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Summarize only encounters that received direction 0 or 1."""
    labels = np.asarray(labels, dtype=int)
    directions = np.asarray(directions, dtype=object)
    mask = np.isin(directions, [0, 1])
    y = labels[mask]
    pred = directions[mask].astype(int)

    result: dict[str, float | int] = {
        "n_total": int(labels.size),
        "n_directional": int(mask.sum()),
        "n_review": int((~mask).sum()),
        "directional_coverage": float(mask.mean()),
    }
    if y.size == 0:
        return result

    result.update(
        tp=int(((pred == 1) & (y == 1)).sum()),
        tn=int(((pred == 0) & (y == 0)).sum()),
        fp=int(((pred == 1) & (y == 0)).sum()),
        fn=int(((pred == 0) & (y == 1)).sum()),
        balanced_accuracy=float(balanced_accuracy_score(y, pred)),
    )
    if scores is not None and np.unique(y).size == 2:
        selected_scores = np.asarray(scores, dtype=float)[mask]
        result["auroc"] = float(roc_auc_score(y, selected_scores))
        result["auprc"] = float(average_precision_score(y, selected_scores))
    return result


def class_conditional_coverage(
    labels: np.ndarray,
    prediction_sets: list[frozenset[int]],
) -> dict[str, float]:
    """Calculate empirical prediction-set coverage by outcome class."""
    labels = np.asarray(labels, dtype=int)
    covered = np.array([label in pred_set for label, pred_set in zip(labels, prediction_sets)])
    return {
        "coverage": float(covered.mean()),
        "coverage_class0": float(covered[labels == 0].mean()),
        "coverage_class1": float(covered[labels == 1].mean()),
    }

