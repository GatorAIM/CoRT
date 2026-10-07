"""Joint prediction-set taxonomy used by CoRT."""

from __future__ import annotations

from enum import Enum
from typing import Iterable


class TriageState(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    OTHER = "OTHER"


def normalize_set(values: Iterable[int]) -> frozenset[int]:
    result = frozenset(int(value) for value in values)
    if not result.issubset({0, 1}):
        raise ValueError("binary prediction sets may contain only 0 and 1")
    return result


def assign_state(tabular_set: Iterable[int], longitudinal_set: Iterable[int]) -> TriageState:
    """Assign one of the six nonempty joint prediction-set states."""
    tabular = normalize_set(tabular_set)
    longitudinal = normalize_set(longitudinal_set)
    pair = {tabular, longitudinal}

    if tabular == longitudinal == frozenset({0}):
        return TriageState.A
    if tabular == longitudinal == frozenset({1}):
        return TriageState.B
    if pair == {frozenset({0}), frozenset({0, 1})}:
        return TriageState.C
    if pair == {frozenset({1}), frozenset({0, 1})}:
        return TriageState.D
    if pair == {frozenset({0}), frozenset({1})}:
        return TriageState.E
    if tabular == longitudinal == frozenset({0, 1}):
        return TriageState.F
    return TriageState.OTHER

