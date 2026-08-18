# Customer Journey Attribution

Multi-touch attribution on real digital-marketing journey data (GA4 public
sample dataset — the Google Merchandise Store) — Markov chain attribution as
an interpretable baseline, compared against a data-driven (LightGBM + SHAP)
approach. Built because attribution is a genuinely hard evaluation problem:
there is almost never ground truth for what actually caused a conversion.

## Why this exists

Most attribution write-ups compute a method and stop. This project exists
to actually *validate* attribution methodology before trusting it on real
data — which real data structurally cannot do, since there's no ground
truth to check against. `simulator.py` solves that by generating synthetic
journeys with a **known, fixed data-generating process**, so every method
gets scored against the truth before it's ever pointed at the real GA4 data.

## A real finding from the simulator (not a hypothetical)

Running the full pipeline against 8,000 simulated journeys with known
channel effects:

| Method | Spearman ρ vs. true effect |
|---|---|
| First/last-touch, linear, time-decay, position-based | ~0.00 (no better than chance) |
| **Markov removal effect** | **−0.18** (worse than chance) |
| **Data-driven (LightGBM + SHAP)** | **0.79** (strong recovery) |

The Markov model's predicted overall conversion rate matched the empirical
rate *exactly* (calibration error 0.0000) — the chain math is correct. But
its per-channel removal-effect ranking is actively **anti-correlated**
with the true effects, because removal effect is confounded by channel
**frequency**, not just causal strength. In the simulation, `email` is rare
but has the strongest true effect; `display` is common but has almost no
effect. Markov ranks them in nearly the opposite order of the truth. This
is a documented, real limitation of Markov attribution (Anderl et al.) —
not a bug in this implementation — and it's exactly the kind of finding
that never shows up if you only look at real data with no ground truth to
check against.

Bootstrap resampling shows the Markov ranking is *stable* (low variance
across resamples) despite being wrong — a useful reminder that stability
and correctness are different things.

## Architecture

Package layout mirrors the pipeline stages (ingest → transform → attribute
/ simulate → evaluate):

- `src/journey_attribution/schemas.py` — typed `Touchpoint`, `Journey`,
  `AttributionResult`, `GroundTruthEffect`. Every stage passes these, never
  raw dicts.
- `src/journey_attribution/simulation/simulator.py` — synthetic journey
  generator with known per-channel effects. The most important file in the
  repo.
- `src/journey_attribution/attribution/baselines.py` — first-touch,
  last-touch, linear, time-decay, position-based (U-shaped) heuristics.
- `src/journey_attribution/attribution/markov.py` — transition-probability
  chain + removal-effect attribution, via absorbing Markov chain algebra
  (fundamental matrix).
- `src/journey_attribution/attribution/datadriven.py` — LightGBM
  classifier + SHAP, channel-level credit from summed `|SHAP|` across each
  channel's features.
- `src/journey_attribution/evaluation/eval_suite.py` — simulation
  recovery, calibration, bootstrap stability, cross-model agreement.
- `src/journey_attribution/ingestion/bigquery.py` — pulls real touchpoint +
  purchase data from the public
  `bigquery-public-data.ga4_obfuscated_sample_ecommerce` dataset into
  `data/raw/`. Requires your own (free-tier) GCP project — see file header
  for setup.
- `src/journey_attribution/transform/journey_builder.py` — turns the raw
  BigQuery export into typed `Journey` objects (drops touchpoints after a
  user's first purchase in the window, to avoid contaminating a converted
  journey with post-purchase activity).
- `src/journey_attribution/cli.py` — CLI (`journey-attribution`), runs
  either mode end to end. Defaults come from `config/settings.yaml`.
- `tests/` — pytest suite: offline pipeline smoke test plus per-method
  invariant checks (credits normalize to ~1.0, probabilities stay in
  range). Runs in CI against the simulator only — no BigQuery credentials
  needed.

## Running it

```bash
pip install -e ".[dev]"

# No setup needed — runs entirely against the simulator:
pytest
journey-attribution --mode simulate

# Real data — requires a free Google Cloud project with BigQuery enabled:
pip install -e ".[bigquery]"
gcloud auth application-default login
gcloud config set project <your-gcp-project-id>
python -m journey_attribution.ingestion.bigquery --project <your-gcp-project-id>
journey-attribution --mode real
```

Or via `make setup`, `make test`, `make simulate`, `make real`,
`make extract PROJECT=<your-gcp-project-id>`.

## Known open items

- The GA4 sample dataset is obfuscated — Google's own docs note some
  fields carry placeholder values and "internal consistency might be
  somewhat limited." The channel-grouping logic in `extract_bigquery.py`
  is a simplified version of Google's default channel grouping and will
  need real data-quality triage once run against the live export — that's
  expected, not a bug to pre-solve.
- `conversion_value` isn't populated from real data yet (GA4 obfuscated
  sample doesn't reliably expose purchase revenue) — attribution here is
  by conversion count, not revenue-weighted.
- Next: run `extract_bigquery.py` against the real dataset, sanity-check
  the channel grouping against actual data, and compare real-data
  attribution results against what the simulator predicted would happen.
