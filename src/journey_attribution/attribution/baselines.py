"""
Heuristic attribution baselines. Not the point of the project, but every
attribution write-up needs these as a reference frame — the interesting
question is whether Markov/data-driven attribution actually beats these,
not just whether they produce numbers.

Author: Anastasiia Bakhtoiarova
"""
from __future__ import annotations
import math
from collections import defaultdict
from journey_attribution.schemas import Journey, AttributionResult


def _normalize(credits: dict[str, float]) -> dict[str, float]:
    total = sum(credits.values())
    if total == 0:
        return {k: 0.0 for k in credits}
    return {k: v / total for k, v in credits.items()}


def first_touch(journeys: list[Journey]) -> AttributionResult:
    credits: dict[str, float] = defaultdict(float)
    for j in journeys:
        if not j.converted:
            continue
        order = j.channels_in_order
        if order:
            credits[order[0]] += 1.0
    return AttributionResult(method="first_touch", credits=_normalize(dict(credits)))


def last_touch(journeys: list[Journey]) -> AttributionResult:
    credits: dict[str, float] = defaultdict(float)
    for j in journeys:
        if not j.converted:
            continue
        order = j.channels_in_order
        if order:
            credits[order[-1]] += 1.0
    return AttributionResult(method="last_touch", credits=_normalize(dict(credits)))


def linear(journeys: list[Journey]) -> AttributionResult:
    credits: dict[str, float] = defaultdict(float)
    for j in journeys:
        if not j.converted:
            continue
        order = j.channels_in_order
        if not order:
            continue
        share = 1.0 / len(order)
        for c in order:
            credits[c] += share
    return AttributionResult(method="linear", credits=_normalize(dict(credits)))


def time_decay(journeys: list[Journey], half_life_hours: float = 72.0) -> AttributionResult:
    credits: dict[str, float] = defaultdict(float)
    for j in journeys:
        if not j.converted:
            continue
        touches = sorted(j.touchpoints, key=lambda t: t.timestamp)
        if not touches:
            continue
        last_time = touches[-1].timestamp
        weights = []
        for t in touches:
            hours_before = (last_time - t.timestamp).total_seconds() / 3600.0
            weights.append(2 ** (-hours_before / half_life_hours))
        total_w = sum(weights)
        for t, w in zip(touches, weights):
            credits[t.channel] += w / total_w
    return AttributionResult(method="time_decay", credits=_normalize(dict(credits)))


def position_based(journeys: list[Journey], first_last_share: float = 0.4) -> AttributionResult:
    """U-shaped: first + last touch split `first_last_share` each,
    remaining credit split evenly across the middle touches."""
    credits: dict[str, float] = defaultdict(float)
    middle_share_total = 1.0 - 2 * first_last_share
    for j in journeys:
        if not j.converted:
            continue
        order = j.channels_in_order
        n = len(order)
        if n == 0:
            continue
        if n == 1:
            credits[order[0]] += 1.0
        elif n == 2:
            credits[order[0]] += 0.5
            credits[order[1]] += 0.5
        else:
            credits[order[0]] += first_last_share
            credits[order[-1]] += first_last_share
            middle = order[1:-1]
            share = middle_share_total / len(middle)
            for c in middle:
                credits[c] += share
    return AttributionResult(method="position_based", credits=_normalize(dict(credits)))


def all_baselines(journeys: list[Journey]) -> list[AttributionResult]:
    return [
        first_touch(journeys),
        last_touch(journeys),
        linear(journeys),
        time_decay(journeys),
        position_based(journeys),
    ]
