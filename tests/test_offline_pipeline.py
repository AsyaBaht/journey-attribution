"""
Offline smoke test: full pipeline against the simulator, which is the only
place ground truth exists. No BigQuery, no external data needed — this is
what proves the methodology sound before it ever touches the real GA4 data.

These are regression guards, not methodology judgments: a method can
legitimately score badly on simulation_recovery (see README — Markov's
known removal-effect confound) without that being a bug. What must hold is
that the harness itself behaves correctly (calibration close to exact,
agreement score in-range) and that results are actually produced.

Author: Anastasiia Bakhtoiarova
"""
from __future__ import annotations

from journey_attribution.attribution.baselines import all_baselines
from journey_attribution.attribution.markov import markov_removal_effect, markov_conversion_probability
from journey_attribution.attribution.datadriven import datadriven_attribution
from journey_attribution.evaluation.eval_suite import (
    simulation_recovery, calibration, bootstrap_stability, cross_model_agreement,
)


def test_full_pipeline_smoke(simulated_journeys, truth):
    actual_rate = sum(j.converted for j in simulated_journeys) / len(simulated_journeys)

    results = {r.method: r for r in all_baselines(simulated_journeys)}
    markov_result = markov_removal_effect(simulated_journeys)
    results[markov_result.method] = markov_result
    dd_result, dd_diag = datadriven_attribution(simulated_journeys)
    results[dd_result.method] = dd_result

    assert dd_diag["train_auc"] > 0.5, "data-driven model should learn something better than chance"

    for target in ("removal_share", "log_odds"):
        recovery_scores = {
            method: simulation_recovery(result, truth, target).value
            for method, result in results.items()
        }
        assert set(recovery_scores) == set(results)
        assert all(-1.0 <= v <= 1.0 for v in recovery_scores.values())

    markov_p = markov_conversion_probability(simulated_journeys)
    cal = calibration(markov_p, actual_rate, "markov")
    assert abs(cal.value) < 0.01, "Markov's predicted conversion rate should match empirical almost exactly"

    stability = bootstrap_stability(simulated_journeys, markov_removal_effect, "markov_removal_effect", n_bootstraps=15)
    assert stability.value >= 0.0

    agreement = cross_model_agreement(markov_result, dd_result)
    assert -1.0 <= agreement.value <= 1.0
