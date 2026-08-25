"""Tests for the data-driven (LightGBM + SHAP) attribution method: credits
must be normalized and non-negative, and the model must actually learn a
signal from journey features rather than just producing well-formed noise.

Author: Anastasiia Bakhtoiarova
"""
from __future__ import annotations
import pytest

from journey_attribution.attribution.datadriven import datadriven_attribution


def test_credits_normalized_and_diagnostics_present(simulated_journeys):
    result, diagnostics = datadriven_attribution(simulated_journeys)
    assert result.credits
    assert sum(result.credits.values()) == pytest.approx(1.0, abs=1e-6)
    assert all(v >= 0 for v in result.credits.values())
    assert "train_auc" in diagnostics
    assert 0.0 <= diagnostics["train_auc"] <= 1.0


def test_model_learns_better_than_chance(simulated_journeys):
    _, diagnostics = datadriven_attribution(simulated_journeys)
    assert diagnostics["train_auc"] > 0.5
