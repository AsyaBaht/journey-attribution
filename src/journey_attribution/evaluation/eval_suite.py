"""
Evaluation suite for attribution methods.

Real journey data has no ground truth for "what caused the conversion," so
this suite checks four different, honest things instead of pretending to
score "correctness":

1. Simulation recovery — against the simulator's known true effects, does
   this method's ranking of channels actually resemble the truth?
   (Only possible because simulator.py exists. This is the whole point.)
2. Calibration — does the method's predicted overall conversion rate match
   the actual empirical rate? (Sanity check, not really about attribution
   quality, but a method that can't get this right shouldn't be trusted
   for anything else.)
3. Bootstrap stability — do a method's credits hold up across resampled
   data, or do they swing wildly? High-variance channels shouldn't be
   reported as confident point estimates.
4. Cross-model agreement — do two independent methods agree on which
   channels matter? Disagreement isn't automatically a bug in one of
   them — it's often the most interesting finding in the whole project.

Author: Anastasiia Bakhtoiarova
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field
from scipy.stats import spearmanr
import numpy as np
from journey_attribution.schemas import Journey, AttributionResult, GroundTruthEffect


@dataclass
class EvalResult:
    dimension: str
    detail: str
    value: float | None = None


@dataclass
class EvalReport:
    results: list[EvalResult] = field(default_factory=list)

    def summary(self) -> str:
        return "\n".join(f"[{r.dimension}] {r.detail}" for r in self.results)


def simulation_recovery(result: AttributionResult, truth: list[GroundTruthEffect]) -> EvalResult:
    truth_map = {g.channel: g.true_log_odds_effect for g in truth}
    common = [c for c in result.credits if c in truth_map]
    est = [result.credits[c] for c in common]
    tru = [truth_map[c] for c in common]
    rho, _ = spearmanr(est, tru)
    return EvalResult(
        dimension="simulation_recovery",
        value=float(rho),
        detail=f"{result.method}: Spearman rho={rho:.3f} between estimated credit "
                f"and true generative effect (1.0 = perfect rank recovery)",
    )


def calibration(predicted_rate: float, actual_rate: float, method: str) -> EvalResult:
    diff = abs(predicted_rate - actual_rate)
    return EvalResult(
        dimension="calibration",
        value=diff,
        detail=f"{method}: predicted conversion rate {predicted_rate:.4f} vs "
                f"actual {actual_rate:.4f} (abs diff {diff:.4f})",
    )


def bootstrap_stability(
    journeys: list[Journey],
    attribution_fn,
    method_name: str,
    n_bootstraps: int = 20,
    seed: int = 7,
) -> EvalResult:
    rng = random.Random(seed)
    n = len(journeys)
    channel_samples: dict[str, list[float]] = {}

    for _ in range(n_bootstraps):
        resample = [journeys[rng.randrange(n)] for _ in range(n)]
        result = attribution_fn(resample)
        if isinstance(result, tuple):  # datadriven_attribution returns (result, diagnostics)
            result = result[0]
        for c, credit in result.credits.items():
            channel_samples.setdefault(c, []).append(credit)

    cv_per_channel = {}
    for c, samples in channel_samples.items():
        arr = np.array(samples)
        mean = arr.mean()
        cv_per_channel[c] = float(arr.std() / mean) if mean > 0 else 0.0

    worst = max(cv_per_channel, key=cv_per_channel.get)
    return EvalResult(
        dimension="bootstrap_stability",
        value=cv_per_channel[worst],
        detail=f"{method_name}: {n_bootstraps} bootstraps. Least stable channel is "
                f"'{worst}' (coefficient of variation {cv_per_channel[worst]:.3f}). "
                f"Per-channel CV: {', '.join(f'{c}={v:.2f}' for c, v in sorted(cv_per_channel.items()))}",
    )


def cross_model_agreement(a: AttributionResult, b: AttributionResult) -> EvalResult:
    common = [c for c in a.credits if c in b.credits]
    va = [a.credits[c] for c in common]
    vb = [b.credits[c] for c in common]
    rho, _ = spearmanr(va, vb)
    return EvalResult(
        dimension="cross_model_agreement",
        value=float(rho),
        detail=f"{a.method} vs {b.method}: Spearman rho={rho:.3f} "
                f"(1.0 = perfect agreement, 0 = no relationship, negative = disagree)",
    )
