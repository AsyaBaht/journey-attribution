"""
CLI entrypoint.

    journey-attribution --mode simulate                 # no setup needed
    journey-attribution --mode real --data data/raw/raw_touchpoints.csv   # after `journey-attribution-extract`

Defaults (n_users, seed, data path) come from config/settings.yaml so a run
is reproducible without relying on argparse defaults buried in code; CLI
flags override the config file when passed explicitly.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import yaml

from journey_attribution.simulation.simulator import generate_journeys, ground_truth
from journey_attribution.transform.journey_builder import build_journeys_from_csv
from journey_attribution.attribution.baselines import all_baselines
from journey_attribution.attribution.markov import markov_removal_effect, markov_conversion_probability
from journey_attribution.attribution.datadriven import datadriven_attribution
from journey_attribution.evaluation.eval_suite import (
    simulation_recovery, calibration, bootstrap_stability, cross_model_agreement,
)
from journey_attribution.reporting.report import build_report

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def run(journeys, truth=None, report_path: str | None = None) -> None:
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

    if report_path:
        build_report(journeys, results, report_path, diagnostics={"datadriven train AUC": dd_diag["train_auc"]})
        print(f"Wrote HTML report to {report_path}")


def main() -> None:
    config = load_config()
    sim_cfg = config.get("simulate", {})
    real_cfg = config.get("real", {})

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["simulate", "real"], default="simulate")
    parser.add_argument("--data", default=real_cfg.get("data_path", "data/raw/raw_touchpoints.csv"))
    parser.add_argument("--n-users", type=int, default=sim_cfg.get("n_users", 8000))
    parser.add_argument("--seed", type=int, default=sim_cfg.get("seed", 1))
    parser.add_argument("--report", default=None, help="Path to write an HTML report with attribution "
                         "comparison charts and journey visualizations (e.g. reports/report.html)")
    args = parser.parse_args()

    if args.mode == "simulate":
        journeys = generate_journeys(n_users=args.n_users, seed=args.seed)
        run(journeys, truth=ground_truth(), report_path=args.report)
    else:
        journeys = build_journeys_from_csv(args.data)
        run(journeys, truth=None, report_path=args.report)


if __name__ == "__main__":
    main()
