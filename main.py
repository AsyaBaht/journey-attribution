"""
CLI entrypoint.

    python main.py --mode simulate                 # no setup needed
    python main.py --mode real --data data/raw_touchpoints.csv   # after extract_bigquery.py
"""
from __future__ import annotations
import argparse
from simulator import generate_journeys, ground_truth
from journey_builder import build_journeys_from_csv
from baselines import all_baselines
from markov_attribution import markov_removal_effect, markov_conversion_probability
from datadriven_attribution import datadriven_attribution
from evals.eval_suite import simulation_recovery, calibration, bootstrap_stability, cross_model_agreement


def run(journeys, truth=None) -> None:
    actual_rate = sum(j.converted for j in journeys) / len(journeys)
    print(f"{len(journeys)} journeys, empirical conversion rate {actual_rate:.4f}\n")

    results = {r.method: r for r in all_baselines(journeys)}
    markov_result = markov_removal_effect(journeys)
    results[markov_result.method] = markov_result
    dd_result, dd_diag = datadriven_attribution(journeys)
    results[dd_result.method] = dd_result

    for method, result in results.items():
        print(f"=== {method} ===")
        for c, credit in result.top_channels(7):
            print(f"  {c:15s} {credit:.4f}")
        print()

    if truth:
        print("=== Simulation recovery (only available with simulated ground truth) ===")
        for method, result in results.items():
            ev = simulation_recovery(result, truth)
            print(f"  {ev.detail}")
        print()

    markov_p = markov_conversion_probability(journeys)
    cal = calibration(markov_p, actual_rate, "markov")
    print(f"=== Calibration ===\n  {cal.detail}\n")

    stability = bootstrap_stability(journeys, markov_removal_effect, "markov_removal_effect", n_bootstraps=15)
    print(f"=== Bootstrap stability ===\n  {stability.detail}\n")

    agreement = cross_model_agreement(markov_result, dd_result)
    print(f"=== Cross-model agreement ===\n  {agreement.detail}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["simulate", "real"], default="simulate")
    parser.add_argument("--data", default="data/raw_touchpoints.csv")
    parser.add_argument("--n-users", type=int, default=8000)
    args = parser.parse_args()

    if args.mode == "simulate":
        journeys = generate_journeys(n_users=args.n_users, seed=1)
        run(journeys, truth=ground_truth())
    else:
        journeys = build_journeys_from_csv(args.data)
        run(journeys, truth=None)
