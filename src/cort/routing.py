"""Deterministic routing rules for CoRT triage states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from .state_assignment import TriageState, assign_state, normalize_set


@dataclass(frozen=True)
class RoutingDecision:
    state: TriageState
    direction: Optional[int]
    review_flag: bool
    monitoring_alert: bool


def route_prediction_sets(
    tabular_set: Iterable[int],
    longitudinal_set: Iterable[int],
) -> RoutingDecision:
    """Apply the prespecified A-D directional and E-F review policy."""
    tabular = normalize_set(tabular_set)
    longitudinal = normalize_set(longitudinal_set)
    state = assign_state(tabular, longitudinal)

    if state == TriageState.A:
        return RoutingDecision(state, 0, False, False)
    if state == TriageState.B:
        return RoutingDecision(state, 1, False, False)
    if state in {TriageState.C, TriageState.D}:
        singleton = tabular if len(tabular) == 1 else longitudinal
        return RoutingDecision(state, next(iter(singleton)), False, False)
    if state in {TriageState.E, TriageState.F}:
        return RoutingDecision(state, None, True, False)
    return RoutingDecision(state, None, True, not tabular or not longitudinal)

