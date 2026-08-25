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
channel effects (`journey-attribution --mode simulate`):

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
but has the strongest true effect (log-odds 0.70); `display` is common but
has almost no effect (log-odds 0.05). Markov ranks `email` *last* by credit
and `display` third — nearly the opposite of the truth. This is a
documented, real limitation of Markov attribution (Anderl et al.) — not a
bug in this implementation — and it's exactly the kind of finding that
never shows up if you only look at real data with no ground truth to check
against.

Bootstrap resampling shows the Markov ranking is *stable* (low variance
across resamples) despite being wrong — a useful reminder that stability
and correctness are different things.

## Results on the real GA4 data

Having validated methodology against the simulator, the same pipeline run
against 267,084 real journeys pulled from
`bigquery-public-data.ga4_obfuscated_sample_ecommerce`
(`journey-attribution --mode real`, empirical conversion rate 1.63%):

| Method | Top channel | 2nd | 3rd |
|---|---|---|---|
| Markov removal effect | organic_search (0.29) | direct (0.23) | referral (0.21) |
| Data-driven (LightGBM + SHAP) | **other (0.43)** | referral (0.26) | organic_search (0.13) |

Calibration is again exact (predicted vs. actual conversion rate,
abs diff 0.0000). Cross-model agreement between Markov and the data-driven
model is moderate (Spearman ρ = 0.49) — weaker than either method's
self-consistency, and driven mostly by the `other` channel: the
data-driven model assigns it dominant credit, which given the sample
dataset's documented obfuscation (see "Known open items" below) reads more
like a **data-quality artifact of the channel-grouping simplification**
than a genuine causal signal, and is exactly the sort of thing the
simulator validation step is meant to make you suspicious of rather than
trust at face value. Full comparison across all seven methods, plus a
journey-flow diagram, is in the HTML report (see below).

## Interactive report

```bash
journey-attribution --mode real --report reports/report.html
```

Writes a single self-contained interactive HTML file (Plotly, opens in any
browser, no server) with:

- attribution credit by channel, grouped by method, plus the underlying
  comparison table
- a Sankey diagram of actual journey flow between channels (built from the
  same Markov transition counts used for removal-effect attribution)
- channel touchpoint frequency, the 10 most common journey paths, and the
  journey-length distribution

`reports/` is gitignored (regenerable output, not source).

### Adding written commentary to the report

`journey-attribution --report` only draws charts — it doesn't know what any
of them mean. To layer your own analysis on top as styled callout boxes,
edit the `INSIGHTS` list in
`src/journey_attribution/reporting/annotate.py` (each entry is a
`(section_heading, html_text)` pair — the callout is inserted right before
that `<h2>`, or at the very end of the report if `heading=None`), then run
it against an already-generated report:

```bash
python -m journey_attribution.reporting.annotate reports/report.html
```

It edits the file in place by inserting `<div class="insight">` boxes at
those fixed points — it never touches the Plotly library or the chart
`<script>` blocks, and it's safe to re-run (each callout is inserted at
most once, so re-running against an already-annotated file is a no-op).
Because `--report` regenerates `reports/report.html` from scratch each
time, the normal workflow is: regenerate the report, then re-run
`annotate.py` to put your commentary back on top.

## Architecture

Package layout mirrors the pipeline stages (ingest → transform → attribute
/ simulate → evaluate → report):

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
  for setup. Run via `python -m journey_attribution.ingestion.bigquery`.
- `src/journey_attribution/transform/journey_builder.py` — turns the raw
  BigQuery export into typed `Journey` objects (drops touchpoints after a
  user's first purchase in the window, to avoid contaminating a converted
  journey with post-purchase activity).
- `src/journey_attribution/reporting/report.py` — builds the interactive
  HTML report described above from a run's attribution results and
  journeys.
- `src/journey_attribution/reporting/annotate.py` — post-processes a
  generated `report.html` to insert hand-written insight callouts at fixed
  points in the page (see "Adding written commentary to the report"
  above). Standalone; doesn't touch chart rendering.
- `src/journey_attribution/cli.py` — CLI (`journey-attribution`), runs
  either mode end to end and optionally writes the HTML report via
  `--report <path>`. Defaults come from `config/settings.yaml`.
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

# Optional: interactive HTML report (attribution comparison + journey visualizations)
pip install -e ".[report]"
journey-attribution --mode real --report reports/report.html   # or --mode simulate
```

Or via `make setup`, `make test`, `make simulate`, `make real`,
`make extract PROJECT=<your-gcp-project-id>`, `make report`.

## Known open items

- The GA4 sample dataset is obfuscated — Google's own docs note some
  fields carry placeholder values and "internal consistency might be
  somewhat limited." The channel-grouping logic in
  `src/journey_attribution/ingestion/bigquery.py` is a simplified version
  of Google's default channel grouping; on the real pull, a large and
  disputed share of traffic lands in the catch-all `other` bucket (see
  "Results on the real GA4 data" above) — that bucket needs to be broken
  down further, or the grouping rules tightened, before the real-data
  attribution numbers should be trusted at face value.
- `conversion_value` isn't populated from real data yet (GA4 obfuscated
  sample doesn't reliably expose purchase revenue) — attribution here is
  by conversion count, not revenue-weighted.
- Next: tighten the `other`-bucket channel grouping against the real
  `traffic_source.source`/`medium` values actually observed in the pull,
  re-run, and check whether Markov/data-driven agreement improves once
  that source of noise is reduced.
