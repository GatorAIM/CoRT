"""Reusable summaries of CoRT state and pathway performance."""

from __future__ import annotations

from collections import Counter

import numpy as np

from cort.metrics import directional_summary
from cort.routing import route_prediction_sets


def summarize_triage(labels, tabular_sets, longitudinal_sets, scores=None) -> dict:
    decisions = [
        route_prediction_sets(tabular, longitudinal)
        for tabular, longitudinal in zip(tabular_sets, longitudinal_sets)
    ]
    states = Counter(decision.state.value for decision in decisions)
    directions = np.array([decision.direction for decision in decisions], dtype=object)
    return {
        "state_counts": dict(sorted(states.items())),
        **directional_summary(np.asarray(labels), directions, scores),
    }

