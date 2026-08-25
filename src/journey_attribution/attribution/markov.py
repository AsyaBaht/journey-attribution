"""
Markov chain attribution.

Model: states are Start, one state per channel, and two absorbing states
(Conversion, Null). Transition probabilities are estimated empirically from
observed journeys. Total conversion probability from Start is computed via
absorbing-Markov-chain algebra: split the transition matrix into transient
(Q) and absorbing (R) blocks, compute the fundamental matrix N = (I - Q)^-1,
then B = N @ R gives absorption probabilities.

Attribution credit per channel = removal effect: rebuild the chain with
that channel's outgoing transitions redirected entirely to Null (simulating
"this channel no longer converts anyone who reaches it"), recompute total
conversion probability, and take the drop as that channel's credit. This is
the standard approach (Anderl et al.), not something invented for this
project — worth citing as such in the eventual write-up.

Author: Anastasiia Bakhtoiarova
"""
from __future__ import annotations
import numpy as np
from collections import defaultdict
from journey_attribution.schemas import Journey, AttributionResult

START, CONVERSION, NULL = "Start", "Conversion", "Null"


def _sequence(journey: Journey, collapse_repeats: bool = True) -> list[str]:
    order = journey.channels_in_order
    if collapse_repeats:
        deduped = []
        for c in order:
            if not deduped or deduped[-1] != c:
                deduped.append(c)
        order = deduped
    end = CONVERSION if journey.converted else NULL
    return [START, *order, end]


def _transition_counts(journeys: list[Journey]) -> dict[tuple[str, str], float]:
    counts: dict[tuple[str, str], float] = defaultdict(float)
    for j in journeys:
        seq = _sequence(j)
        for a, b in zip(seq, seq[1:]):
            counts[(a, b)] += 1.0
    return counts


def _states(journeys: list[Journey]) -> list[str]:
    channels = sorted({c for j in journeys for c in j.channels_in_order})
    return [START, *channels, CONVERSION, NULL]


def _matrix(counts: dict[tuple[str, str], float], states: list[str]) -> np.ndarray:
    idx = {s: i for i, s in enumerate(states)}
    n = len(states)
    m = np.zeros((n, n))
    for (a, b), c in counts.items():
        m[idx[a], idx[b]] = c
    # row-normalize; absorbing states get an explicit self-loop of 1.0
    for i, s in enumerate(states):
        row_sum = m[i].sum()
        if s in (CONVERSION, NULL):
            m[i, i] = 1.0
        elif row_sum > 0:
            m[i] = m[i] / row_sum
    return m


def _total_conversion_probability(m: np.ndarray, states: list[str]) -> float:
    idx = {s: i for i, s in enumerate(states)}
    absorbing = [idx[CONVERSION], idx[NULL]]
    transient = [i for i in range(len(states)) if i not in absorbing]

    Q = m[np.ix_(transient, transient)]
    R = m[np.ix_(transient, absorbing)]
    n_t = len(transient)
    N = np.linalg.inv(np.eye(n_t) - Q)  # fundamental matrix
    B = N @ R  # absorption probabilities: transient -> absorbing

    start_pos = transient.index(idx[START])
    conv_pos = absorbing.index(idx[CONVERSION])
    return float(B[start_pos, conv_pos])


def markov_removal_effect(journeys: list[Journey]) -> AttributionResult:
    counts = _transition_counts(journeys)
    states = _states(journeys)
    baseline_matrix = _matrix(counts, states)
    baseline_p = _total_conversion_probability(baseline_matrix, states)

    channels = [s for s in states if s not in (START, CONVERSION, NULL)]
    effects: dict[str, float] = {}

    for removed in channels:
        modified_counts = dict(counts)
        # redirect every outgoing transition from `removed` entirely to Null
        outgoing_keys = [k for k in modified_counts if k[0] == removed]
        outgoing_total = sum(modified_counts[k] for k in outgoing_keys)
        for k in outgoing_keys:
            del modified_counts[k]
        if outgoing_total > 0:
            modified_counts[(removed, NULL)] = outgoing_total

        modified_matrix = _matrix(modified_counts, states)
        removed_p = _total_conversion_probability(modified_matrix, states)
        effects[removed] = max(0.0, baseline_p - removed_p)

    total = sum(effects.values())
    credits = {k: (v / total if total > 0 else 0.0) for k, v in effects.items()}
    return AttributionResult(method="markov_removal_effect", credits=credits)


def transition_summary(journeys: list[Journey]) -> tuple[list[str], dict[tuple[str, str], float]]:
    """States and raw transition counts, exposed for callers that want to
    visualize journey flow (e.g. a Sankey diagram) rather than compute
    removal effects."""
    return _states(journeys), _transition_counts(journeys)


def markov_conversion_probability(journeys: list[Journey]) -> float:
    """Exposed separately so the eval suite can check this number against
    the journeys' actual empirical conversion rate as a sanity check."""
    counts = _transition_counts(journeys)
    states = _states(journeys)
    return _total_conversion_probability(_matrix(counts, states), states)
