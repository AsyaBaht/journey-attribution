"""Shared fixtures: simulated journeys are the only place ground truth
exists, so every test that needs journeys should draw from here rather than
generating its own ad hoc data — keeps results comparable across tests and
matches what the offline smoke test / README results table were run on."""
from __future__ import annotations
import pytest

from journey_attribution.simulation.simulator import generate_journeys, ground_truth


@pytest.fixture(scope="session")
def simulated_journeys():
    return generate_journeys(n_users=8000, seed=1)


@pytest.fixture(scope="session")
def truth():
    return ground_truth()
