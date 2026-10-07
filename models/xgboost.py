"""Reference XGBoost construction used as the tabular CoRT interface."""

from __future__ import annotations

from typing import Any

import numpy as np
import xgboost as xgb


def build_xgboost(config: dict[str, Any], *, positive_weight: float = 1.0) -> xgb.XGBClassifier:
    """Build a binary probabilistic classifier from a caller-supplied config."""
    return xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric=config.get("eval_metric", "logloss"),
        tree_method=config.get("tree_method", "hist"),
        n_estimators=int(config.get("n_estimators", 1000)),
        max_depth=int(config.get("max_depth", 4)),
        learning_rate=float(config.get("learning_rate", 0.05)),
        subsample=float(config.get("subsample", 0.8)),
        colsample_bytree=float(config.get("colsample_bytree", 0.8)),
        reg_alpha=float(config.get("reg_alpha", 0.0)),
        reg_lambda=float(config.get("reg_lambda", 1.0)),
        scale_pos_weight=float(positive_weight),
        random_state=int(config.get("seed", 0)),
        n_jobs=int(config.get("n_jobs", -1)),
    )


def positive_class_probability(model: xgb.XGBClassifier, features: np.ndarray) -> np.ndarray:
    """Return the probability expected by the conformal layer."""
    return model.predict_proba(features)[:, 1]

