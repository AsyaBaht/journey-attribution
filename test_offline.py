"""
Offline smoke test: full pipeline against the simulator, which is the only
place ground truth exists. No BigQuery, no external data needed — this is
what proves the methodology sound before it ever touches the real GA4 data.
"""
import sys
from simulator import generate_journeys, ground_truth
from baselines import all_baselines
from markov_attribution import markov_removal_effect, markov_conversion_probability
from datadriven_attribution import datadriven_attribution
from evals.eval_suite import (
    simulation_recovery, calibration, bootstrap_stability, cross_model_agreement,
)

print("=== Generating synthetic journeys (known ground truth) ===")
journeys = generate_journeys(n_users=8000, seed=1)
truth = ground_truth()
actual_rate = sum(j.converted for j in journeys) / len(journeys)
print(f"{len(journeys)} journeys, empirical conversion rate {actual_rate:.4f}")

print("\n=== Running all attribution methods ===")
results = {}
for r in all_baselines(journeys):
    results[r.method] = r
markov_result = markov_removal_effect(journeys)
results[markov_result.method] = markov_result
dd_result, dd_diag = datadriven_attribution(journeys)
results[dd_result.method] = dd_result
print(f"Methods run: {list(results.keys())}")
print(f"Data-driven model AUC: {dd_diag['train_auc']:.3f}")

print("\n=== Eval 1: Simulation recovery (rank correlation vs true effects) ===")
recovery_scores = {}
for method, result in results.items():
    ev = simulation_recovery(result, truth)
    recovery_scores[method] = ev.value
    print(f"  {ev.detail}")

print("\n=== Eval 2: Calibration (Markov predicted vs actual conversion rate) ===")
markov_p = markov_conversion_probability(journeys)
cal = calibration(markov_p, actual_rate, "markov")
print(f"  {cal.detail}")

print("\n=== Eval 3: Bootstrap stability (Markov, 15 resamples) ===")
stability = bootstrap_stability(journeys, markov_removal_effect, "markov_removal_effect", n_bootstraps=15)
print(f"  {stability.detail}")

print("\n=== Eval 4: Cross-model agreement (Markov vs data-driven) ===")
agreement = cross_model_agreement(markov_result, dd_result)
print(f"  {agreement.detail}")

print("\n=== Interpretation ===")
best_method = max(recovery_scores, key=recovery_scores.get)
worst_method = min(recovery_scores, key=recovery_scores.get)
print(f"Best simulation-recovery method: {best_method} (rho={recovery_scores[best_method]:.3f})")
print(f"Worst simulation-recovery method: {worst_method} (rho={recovery_scores[worst_method]:.3f})")

# Sanity assertions — proves the harness itself works, not just that it runs.
assert abs(cal.value) < 0.01, "Markov calibration should match empirical rate almost exactly"
assert recovery_scores["last_touch"] < recovery_scores[dd_result.method] + 0.5, \
    "sanity: shouldn't wildly favor the weakest heuristic over the modeled approach"
assert -1.0 <= agreement.value <= 1.0

print("\nAll offline smoke tests passed.")
