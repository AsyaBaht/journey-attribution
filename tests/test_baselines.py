"""Tests for the heuristic attribution baselines (first/last touch, linear,
time-decay, position-based): every method's credits must be normalized and
non-negative regardless of the underlying weighting scheme.

Author: Anastasiia Bakhtoiarova
"""
from __future__ import annotations
import pytest

from journey_attribution.attribution.baselines import all_baselines


@pytest.mark.parametrize("result_index", range(5))
def test_credits_normalized(simulated_journeys, result_index):
    result = all_baselines(simulated_journeys)[result_index]
    assert result.credits, f"{result.method} produced no credits"
    assert sum(result.credits.values()) == pytest.approx(1.0, abs=1e-6)
    assert all(v >= 0 for v in result.credits.values())
