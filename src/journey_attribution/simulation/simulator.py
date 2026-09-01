"""
Synthetic journey generator with a known, fixed data-generating process.

Why this exists: real attribution data has no ground truth for "what
actually caused the conversion." This simulator does — the per-channel
effect is set by construction. Both attribution methods (Markov and
data-driven) get validated against this before either one is trusted on
the real GA4 data. This is the single most important file in the project;
everything else is only as credible as this validation step.

Generative model, in two parts:

1. *Sequence.* Channels are drawn from a first-order Markov process, not
   i.i.d. — the next channel depends on the previous one (see
   TRANSITION_AFFINITY). This matters: with i.i.d. draws the true
   transition matrix is rank-1 (P(b|a) = prevalence(b) for every a), so a
   Markov attribution model is being asked to recover order structure from
   a process that has none. Validating a sequence model against a
   sequence-free DGP tells you nothing about the model.

2. *Conversion.* Log-odds of conversion = a base rate, plus a per-channel
   main effect (TRUE_CHANNEL_EFFECTS, with diminishing returns on repeat
   exposure), plus an ordered-pair bonus for specific consecutive
   transitions (TRUE_SEQUENCE_EFFECTS). The pair term is what makes
   conversion genuinely order-dependent: permuting a journey's channels
   now changes its conversion probability.

Because conversion depends on order, the honest statement of "what a
channel is worth" is no longer a single log-odds number — see
`true_removal_effect()` below, which computes the counterfactual directly
from the DGP and is the estimand removal-effect attribution actually
targets.

Author: Anastasiia Bakhtoiarova
"""
from __future__ import annotations
import random
import math
from datetime import datetime, timedelta
from journey_attribution.schemas import Touchpoint, Journey, GroundTruthEffect

# True per-channel main effects, fixed by construction. Positive =
# increases conversion odds. Deliberately includes one "high frequency,
# near-zero effect" channel (display) and one "low frequency, high effect"
# channel (email) — exactly the pattern that trips up naive
# last-touch/first-touch heuristics.
TRUE_CHANNEL_EFFECTS: dict[str, float] = {
    "paid_search": 0.55,
    "organic_search": 0.35,
    "email": 0.70,
    "social": 0.10,
    "display": 0.05,
    "direct": 0.40,
    "referral": 0.20,
}

# True effects of specific *ordered consecutive* pairs, added on top of the
# main effects. This is the only part of the DGP that a sequence-aware
# method (Markov) can see and an order-blind one (channel-count features)
# cannot. One pair is deliberately negative, so methods that discard sign
# are penalised rather than rewarded.
TRUE_SEQUENCE_EFFECTS: dict[tuple[str, str], float] = {
    ("paid_search", "email"): 0.60,     # retargeted paid visitor opens email
    ("social", "paid_search"): 0.35,    # social discovery, then intent search
    ("display", "organic_search"): 0.25,  # display seeds branded search
    ("email", "display"): -0.30,        # over-served after opting in; fatigue
}

BASE_LOG_ODDS = -2.8  # low baseline conversion rate, realistic for e-commerce
REPEAT_EXPOSURE_DECAY = 0.35  # repeat touches count at 35% of first-exposure effect

# Marginal prevalence of each channel as a journey's *entry* point. Chosen
# so paid/organic dominate volume and email is rare but high-effect — i.e.
# volume is close to uncorrelated with true effect, so a method that just
# ranks channels by frequency scores badly on purpose.
ENTRY_WEIGHTS: dict[str, float] = {
    "paid_search": 30, "organic_search": 28, "email": 8,
    "social": 18, "display": 22, "direct": 15, "referral": 10,
}

# Multiplicative bumps applied to ENTRY_WEIGHTS when picking the *next*
# channel, conditional on the previous one. These are what make the true
# transition matrix genuinely prev-dependent. Note they only partly overlap
# with TRUE_SEQUENCE_EFFECTS: a transition can be common without being
# valuable (that is the whole confound Markov attribution has to survive).
TRANSITION_AFFINITY: dict[tuple[str, str], float] = {
    ("paid_search", "email"): 4.0,
    ("email", "direct"): 3.0,
    ("social", "paid_search"): 2.5,
    ("display", "organic_search"): 2.0,
    ("organic_search", "direct"): 2.0,
    ("referral", "organic_search"): 1.8,
    ("email", "display"): 1.5,          # common *and* harmful, by design
    ("display", "social"): 1.6,         # common and near-worthless, by design
}
SELF_TRANSITION_MULTIPLIER = 0.6  # mild anti-stickiness: repeats happen, but less


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def ground_truth(journeys: list[Journey] | None = None) -> list[GroundTruthEffect]:
    """Known truth for each channel.

    `true_log_odds_effect` is the per-exposure main effect — a parameter of
    the DGP, free of any volume or sequence context.

    `true_removal_share` is only populated when `journeys` is supplied: it
    is the channel's share of total counterfactual conversion loss on that
    specific set of journeys (see `true_removal_effect`). The two are
    different quantities and methods should be scored against whichever one
    they actually estimate — credit *shares* against removal share,
    per-exposure effects against log-odds.
    """
    shares = true_removal_share(journeys) if journeys else {}
    return [
        GroundTruthEffect(
            channel=c,
            true_log_odds_effect=e,
            true_removal_share=shares.get(c),
        )
        for c, e in TRUE_CHANNEL_EFFECTS.items()
    ]


def _conversion_probability(channels_in_order: list[str]) -> float:
    seen: dict[str, int] = {}
    log_odds = BASE_LOG_ODDS
    for c in channels_in_order:
        seen[c] = seen.get(c, 0) + 1
        effect = TRUE_CHANNEL_EFFECTS.get(c, 0.0)
        log_odds += effect if seen[c] == 1 else effect * REPEAT_EXPOSURE_DECAY
    for a, b in zip(channels_in_order, channels_in_order[1:]):
        log_odds += TRUE_SEQUENCE_EFFECTS.get((a, b), 0.0)
    return _sigmoid(log_odds)


def true_removal_effect(journeys: list[Journey]) -> dict[str, float]:
    """Ground-truth counterfactual: for each channel, the mean drop in
    conversion probability when every touch of that channel is deleted from
    every journey.

    This is the estimand that removal-effect attribution (Markov) and, less
    directly, SHAP-based credit are trying to hit. It is *not* the same as
    `TRUE_CHANNEL_EFFECTS`: a channel's counterfactual value depends on how
    often it appears and on which adjacencies it participates in, neither of
    which is in its log-odds parameter. Scoring a normalized credit share
    against a per-exposure log-odds effect compares two different things and
    will mark a correct method wrong.

    Deleting a channel closes the gap it leaves — (a, X, b) becomes (a, b) —
    so an adjacency the channel was blocking can form. That is the intended
    "this channel never existed" semantics, and it is why this has to be
    computed from the DGP rather than read off a parameter.
    """
    totals: dict[str, float] = {c: 0.0 for c in TRUE_CHANNEL_EFFECTS}
    for j in journeys:
        order = j.channels_in_order
        p_full = _conversion_probability(order)
        for c in totals:
            if c not in order:
                continue  # absent channel has no counterfactual effect here
            counterfactual = [x for x in order if x != c]
            totals[c] += p_full - _conversion_probability(counterfactual)
    n = len(journeys)
    return {c: v / n for c, v in totals.items()} if n else totals


def true_removal_share(journeys: list[Journey]) -> dict[str, float]:
    """`true_removal_effect` normalized to sum to 1.0, so it is directly
    comparable to the normalized credit shares every attribution method in
    this project emits."""
    effects = true_removal_effect(journeys)
    total = sum(effects.values())
    if total <= 0:
        return {c: 0.0 for c in effects}
    return {c: v / total for c, v in effects.items()}


def _sample_sequence(rng: random.Random, n_touches: int, channels: list[str]) -> list[str]:
    """First-order Markov draw over channels: the entry touch comes from
    ENTRY_WEIGHTS, each subsequent touch from those weights reshaped by
    TRANSITION_AFFINITY given the previous channel."""
    entry_w = [ENTRY_WEIGHTS[c] for c in channels]
    seq = [rng.choices(channels, weights=entry_w, k=1)[0]]
    for _ in range(n_touches - 1):
        prev = seq[-1]
        w = [
            ENTRY_WEIGHTS[c] * (
                SELF_TRANSITION_MULTIPLIER if c == prev
                else TRANSITION_AFFINITY.get((prev, c), 1.0)
            )
            for c in channels
        ]
        seq.append(rng.choices(channels, weights=w, k=1)[0])
    return seq


def generate_journeys(
    n_users: int = 5000,
    min_touches: int = 1,
    max_touches: int = 8,
    seed: int = 42,
) -> list[Journey]:
    rng = random.Random(seed)
    channels = list(TRUE_CHANNEL_EFFECTS.keys())

    journeys: list[Journey] = []
    start = datetime(2027, 1, 1)

    for i in range(n_users):
        n_touches = rng.randint(min_touches, max_touches)
        chosen = _sample_sequence(rng, n_touches, channels)
        base_time = start + timedelta(minutes=rng.randint(0, 60 * 24 * 60))

        # Gaps must accumulate: an offset of `idx * random()` is not
        # monotonic in idx (touch 2 at 36h lands after touch 3 at 2h), and
        # since Journey.channels_in_order sorts by timestamp, that silently
        # hands every downstream method a partially shuffled sequence.
        offsets_hours: list[int] = []
        elapsed = 0
        for _ in chosen:
            offsets_hours.append(elapsed)
            elapsed += rng.randint(1, 36)

        touchpoints = [
            Touchpoint(channel=c, timestamp=base_time + timedelta(hours=h))
            for c, h in zip(chosen, offsets_hours)
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
