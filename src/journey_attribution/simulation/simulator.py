"""
Synthetic journey generator with a known, fixed data-generating process.

Why this exists: real attribution data has no ground truth for "what
actually caused the conversion." This simulator does — the per-channel
effect is set by construction. Both attribution methods (Markov and
data-driven) get validated against this before either one is trusted on
the real GA4 data. This is the single most important file in the project;
everything else is only as credible as this validation step.

Generative model: each channel has a true log-odds effect. A user's
conversion probability is a logistic function of the unique channels they
were exposed to (diminishing returns on repeat exposure to the same
channel — repeats count at a fraction of the first-exposure effect).
"""
from __future__ import annotations
import random
import math
from datetime import datetime, timedelta
from journey_attribution.schemas import Touchpoint, Journey, GroundTruthEffect

# True effects, fixed by construction. Positive = increases conversion odds.
# Deliberately includes one "high frequency, near-zero effect" channel
# (display) and one "low frequency, high effect" channel (email) — this is
# exactly the pattern that trips up naive last-touch/first-touch heuristics
# and is a good stress test for both Markov and data-driven methods.
TRUE_CHANNEL_EFFECTS: dict[str, float] = {
    "paid_search": 0.55,
    "organic_search": 0.35,
    "email": 0.70,
    "social": 0.10,
    "display": 0.05,
    "direct": 0.40,
    "referral": 0.20,
}
BASE_LOG_ODDS = -2.8  # low baseline conversion rate, realistic for e-commerce
REPEAT_EXPOSURE_DECAY = 0.35  # repeat touches count at 35% of first-exposure effect


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def ground_truth() -> list[GroundTruthEffect]:
    return [GroundTruthEffect(channel=c, true_log_odds_effect=e)
            for c, e in TRUE_CHANNEL_EFFECTS.items()]


def _conversion_probability(channels_in_order: list[str]) -> float:
    seen: dict[str, int] = {}
    log_odds = BASE_LOG_ODDS
    for c in channels_in_order:
        seen[c] = seen.get(c, 0) + 1
        effect = TRUE_CHANNEL_EFFECTS.get(c, 0.0)
        log_odds += effect if seen[c] == 1 else effect * REPEAT_EXPOSURE_DECAY
    return _sigmoid(log_odds)


def generate_journeys(
    n_users: int = 5000,
    min_touches: int = 1,
    max_touches: int = 8,
    seed: int = 42,
) -> list[Journey]:
    rng = random.Random(seed)
    channels = list(TRUE_CHANNEL_EFFECTS.keys())
    # rough prevalence weighting so the channel mix looks like real traffic
    # (paid/organic dominate volume; email is rarer but high-effect)
    weights = {"paid_search": 30, "organic_search": 28, "email": 8,
               "social": 18, "display": 22, "direct": 15, "referral": 10}
    w = [weights[c] for c in channels]

    journeys: list[Journey] = []
    start = datetime(2027, 1, 1)

    for i in range(n_users):
        n_touches = rng.randint(min_touches, max_touches)
        chosen = rng.choices(channels, weights=w, k=n_touches)
        base_time = start + timedelta(minutes=rng.randint(0, 60 * 24 * 60))
        touchpoints = [
            Touchpoint(channel=c, timestamp=base_time + timedelta(hours=idx * rng.randint(1, 36)))
            for idx, c in enumerate(chosen)
        ]
        p_convert = _conversion_probability(chosen)
        converted = rng.random() < p_convert
        journeys.append(Journey(
            user_id=f"sim_user_{i:06d}",
            touchpoints=touchpoints,
            converted=converted,
            conversion_value=round(rng.uniform(20, 250), 2) if converted else None,
        ))
    return journeys
