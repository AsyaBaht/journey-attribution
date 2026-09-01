"""
Structured types for the whole pipeline. Same discipline as parks-planner:
every stage passes typed objects, not raw dicts — makes the eval suite and
the simulator (which needs to compare against a known ground truth) both
much easier to reason about.

Author: Anastasiia Bakhtoiarova
"""
from __future__ import annotations
from datetime import datetime
from pydantic import BaseModel, Field


class Touchpoint(BaseModel):
    channel: str
    timestamp: datetime


class Journey(BaseModel):
    user_id: str
    touchpoints: list[Touchpoint]
    converted: bool
    conversion_value: float | None = None

    @property
    def channels_in_order(self) -> list[str]:
        return [tp.channel for tp in sorted(self.touchpoints, key=lambda t: t.timestamp)]

    @property
    def unique_channels(self) -> set[str]:
        return {tp.channel for tp in self.touchpoints}


class AttributionResult(BaseModel):
    """Output of any attribution method — heuristic, Markov, or data-driven.
    Credits should sum to roughly 1.0 (normalized) so different methods are
    directly comparable."""
    method: str
    credits: dict[str, float] = Field(description="channel -> normalized credit, sums to ~1.0")

    def top_channels(self, n: int = 3) -> list[tuple[str, float]]:
        return sorted(self.credits.items(), key=lambda kv: kv[1], reverse=True)[:n]


class GroundTruthEffect(BaseModel):
    """Only produced by the simulator, where the true generative effect of
    each channel is known by construction. This is what makes evaluation
    possible at all — real data never has this.

    Two different notions of "truth", because attribution methods estimate
    two different things and scoring one against the other marks correct
    methods wrong:

    - `true_log_odds_effect`: the per-exposure DGP parameter. Volume-free
      and context-free.
    - `true_removal_share`: the channel's share of total counterfactual
      conversion loss on a *specific* set of journeys, so it carries volume
      and adjacency effects. This is what a normalized credit share from
      any of the attribution methods here is actually estimating. Only
      populated when ground truth is requested against journeys.
    """
    channel: str
    true_log_odds_effect: float
    true_removal_share: float | None = None
