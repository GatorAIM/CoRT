from cort.state_assignment import TriageState, assign_state


def test_six_states() -> None:
    assert assign_state({0}, {0}) == TriageState.A
    assert assign_state({1}, {1}) == TriageState.B
    assert assign_state({0}, {0, 1}) == TriageState.C
    assert assign_state({0, 1}, {1}) == TriageState.D
    assert assign_state({0}, {1}) == TriageState.E
    assert assign_state({0, 1}, {0, 1}) == TriageState.F


def test_state_assignment_is_order_invariant() -> None:
    assert assign_state({0, 1}, {0}) == TriageState.C
    assert assign_state({1}, {0, 1}) == TriageState.D
    assert assign_state({1}, {0}) == TriageState.E

