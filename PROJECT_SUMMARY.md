# Customer Journey Attribution — Project Summary

## Quick summary

Multi-touch marketing attribution on real digital-marketing journey data:
Markov-chain removal-effect attribution and a data-driven (LightGBM + SHAP)
model, both benchmarked against classic heuristics (first/last-touch, linear,
time-decay, position-based). Data used: 267,084 real user journeys
(~360,662 touchpoints) pulled from the public
`bigquery-public-data.ga4_obfuscated_sample_ecommerce` dataset (Google
Merchandise Store), plus 8,000 synthetic journeys from a simulator with a
known data-generating process. Analysis compares the credit each method
assigns per channel and validates each method against the simulator's known
ground truth before trusting it on the real data.

## PROBLEM

Deciding how to allocate marketing budget across channels requires knowing
which touchpoints along a customer's path actually drove the conversion;
without that, spend defaults to last-click or intuition and high-value
channels are underfunded — but conversions carry no ground truth for what
caused them, so the attribution method itself must be validated before its
numbers can guide those decisions.

## APPROACH

Ran seven attribution methods over 8,000 simulated journeys (channels with
fixed, known log-odds effects) and 267,084 real GA4 journeys (empirical
conversion rate 1.63%). Validation on the simulator scored each method four
ways: rank recovery of the true channel effects (Spearman ρ), calibration of
predicted vs. actual conversion rate, bootstrap stability across resamples,
and cross-model agreement between Markov and the data-driven model.

## OUTCOME

On the simulator, only the data-driven LightGBM + SHAP model recovered the
true channel ranking (Spearman ρ = 0.79); all heuristics scored ≈0 (chance)
and Markov removal effect scored −0.18 (anti-correlated, because removal
effect is confounded by channel frequency — it ranked the rare, high-effect
`email` channel last). Markov calibration was exact (conversion-rate error
0.0000) and its wrong ranking was stable across bootstraps, showing
stability ≠ correctness. On the real GA4 data the two trusted methods
agreed only moderately (Spearman ρ = 0.49): heuristics and Markov all rank
channels roughly in proportion to raw touchpoint volume (organic_search
first), while the data-driven model (AUC only 0.712) assigns dominant credit
to the catch-all `other` bucket. Given the dataset's documented obfuscation
and the simplified channel grouping, that `other` result reads as a
data-quality artifact rather than a causal signal, and the real-data
attribution numbers should not be trusted at face value until the `other`
bucket is broken down further. 83% of real journeys are single-touch (mean
length 1.31), so multi-touch attribution meaningfully applies to only ~17%
of traffic.
