"""Tests for the simulator's data-generating process.

These guard the two properties the rest of the evaluation depends on and
that nothing else would catch:

1. Touchpoint timestamps are monotonic in generation order. `Journey.
   channels_in_order` sorts by timestamp, so if they aren't, every method
   downstream silently receives a shuffled sequence while the labels stay
   put — a bug that produces plausible-looking numbers rather than an error.
2. The DGP is actually order-dependent. Without this, the true transition
   matrix is rank-1 and a Markov model is being scored on structure that
   does not exist in the data.

Author: Anastasiia Bakhtoiarova
"""
from __future__ import annotations
import random

import pytest

from journey_attribution.simulation.simulator import (
    generate_journeys, ground_truth, true_removal_share,
    _conversion_probability, TRUE_CHANNEL_EFFECTS,
)


def test_timestamps_monotonic_in_generation_order(simulated_journeys):
    for j in simulated_journeys:
        stamps = [t.timestamp for t in j.touchpoints]
        assert stamps == sorted(stamps), (
            f"{j.user_id} has out-of-order timestamps; channels_in_order sorts "
            f"by timestamp, so downstream methods would see a shuffled journey"
        )


def test_dgp_is_order_dependent():
    """Permuting a journey's channels must change its conversion probability,
    otherwise there is no sequence signal for a Markov model to recover."""
    rng = random.Random(0)
    sequences = [rng.choices(list(TRUE_CHANNEL_EFFECTS), k=5) for _ in range(200)]
    changed = sum(
        1 for s in sequences
        if abs(_conversion_probability(s) - _conversion_probability(rng.sample(s, len(s)))) > 1e-9
    )
    assert changed > 0.25 * len(sequences), (
        f"only {changed}/{len(sequences)} permutations changed p(convert) — "
        f"the DGP is effectively order-free"
    )


def test_transition_matrix_is_not_rank_one(simulated_journeys):
    """Next-channel distribution must actually depend on the previous
    channel. If it doesn't, P(b|a) = prevalence(b) and Markov attribution
    has nothing to estimate."""
    from collections import Counter, defaultdict

    following: dict[str, Counter] = defaultdict(Counter)
    for j in simulated_journeys:
        order = j.channels_in_order
        for a, b in zip(order, order[1:]):
            following[a][b] += 1

    def dist(counter: Counter) -> dict[str, float]:
        n = sum(counter.values())
        return {c: counter[c] / n for c in TRUE_CHANNEL_EFFECTS}

    marginal = dist(sum(following.values(), Counter()))
    max_deviation = max(
        abs(dist(following[a])[c] - marginal[c])
        for a in following for c in TRUE_CHANNEL_EFFECTS
    )
    assert max_deviation > 0.05, (
        f"conditional and marginal next-channel distributions differ by at most "
        f"{max_deviation:.4f} — transitions carry no information"
    )


def test_removal_share_is_normalized(simulated_journeys):
    shares = true_removal_share(simulated_journeys)
    assert set(shares) == set(TRUE_CHANNEL_EFFECTS)
    assert sum(shares.values()) == pytest.approx(1.0, abs=1e-9)
    assert all(v >= 0 for v in shares.values())


def test_ground_truth_needs_journeys_for_removal_share(simulated_journeys):
    assert all(g.true_removal_share is None for g in ground_truth())
    assert all(g.true_removal_share is not None for g in ground_truth(simulated_journeys))


def test_generation_is_reproducible():
    a = generate_journeys(n_users=200, seed=3)
    b = generate_journeys(n_users=200, seed=3)
    assert [j.channels_in_order for j in a] == [j.channels_in_order for j in b]
    assert [j.converted for j in a] == [j.converted for j in b]
