from cort.state_assignment import TriageState
from cort.routing import route_prediction_sets


def test_directional_routing() -> None:
    assert route_prediction_sets({0}, {0}).direction == 0
    assert route_prediction_sets({1}, {1}).direction == 1
    assert route_prediction_sets({0, 1}, {0}).direction == 0
    assert route_prediction_sets({1}, {0, 1}).direction == 1


def test_review_routing() -> None:
    conflict = route_prediction_sets({0}, {1})
    uncertainty = route_prediction_sets({0, 1}, {0, 1})
    assert conflict.state == TriageState.E and conflict.review_flag
    assert uncertainty.state == TriageState.F and uncertainty.review_flag
    assert conflict.direction is None and uncertainty.direction is None


def test_empty_set_monitoring() -> None:
    decision = route_prediction_sets(set(), {0, 1})
    assert decision.review_flag
    assert decision.monitoring_alert

