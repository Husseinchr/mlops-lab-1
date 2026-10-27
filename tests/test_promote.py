import pytest

from src.food11.promote import promotion_decision


def test_gate_boundary():
    accepted, margin = promotion_decision(0.9, 0.8, 1000)
    assert accepted
    assert margin == pytest.approx(0.02529822128)
    assert not promotion_decision(0.81, 0.8, 1000)[0]
    assert promotion_decision(0.8 + margin, 0.8, 1000)[0]


def test_invalid_gate_inputs():
    with pytest.raises(ValueError):
        promotion_decision(float("nan"), 0.8, 1000)
    with pytest.raises(ValueError):
        promotion_decision(0.9, 0.8, 0)
