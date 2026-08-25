"""Tests for Markov removal-effect attribution: credits must be normalized
and non-negative, and the chain's total conversion probability must be a
valid probability that matches the empirical conversion rate.

Author: Anastasiia Bakhtoiarova
"""
from __future__ import annotations
import pytest

from journey_attribution.attribution.markov import markov_removal_effect, markov_conversion_probability


def test_removal_effect_credits_normalized(simulated_journeys):
    result = markov_removal_effect(simulated_journeys)
    assert result.credits
    assert sum(result.credits.values()) == pytest.approx(1.0, abs=1e-6)
    assert all(v >= 0 for v in result.credits.values())


def test_conversion_probability_is_a_probability(simulated_journeys):
    p = markov_conversion_probability(simulated_journeys)
    assert 0.0 <= p <= 1.0


def test_conversion_probability_matches_empirical_rate(simulated_journeys):
    actual_rate = sum(j.converted for j in simulated_journeys) / len(simulated_journeys)
    p = markov_conversion_probability(simulated_journeys)
    assert p == pytest.approx(actual_rate, abs=0.01)
