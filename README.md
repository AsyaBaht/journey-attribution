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

The DGP is deliberately *order-dependent*: channels are drawn from a
first-order Markov process rather than i.i.d., and specific ordered pairs
of consecutive channels carry their own effects. This matters more than it
sounds. With i.i.d. draws the true transition matrix is rank-1 —
`P(b | a) = prevalence(b)` for every `a` — so a Markov attribution model
would be scored on recovering sequence structure from a process that has
none, and would fail for reasons that say nothing about the method.
Validating a sequence model requires a DGP with sequences in it.

## Two definitions of "truth", and why the choice decides the answer

Before any result: scoring an attribution method requires deciding what
you're scoring it *against*, and the obvious choice is wrong.

Every method here emits a **normalized credit share** — a number that
necessarily carries channel volume, because a channel touched three times
as often contributes more total conversions at the same per-touch strength.
The simulator's `TRUE_CHANNEL_EFFECTS`, meanwhile, are **per-exposure
log-odds parameters** — volume-free by construction. Rank-correlating one
against the other compares two different quantities and marks a correct
method wrong for correctly reflecting volume.

So the simulator computes both (`simulator.true_removal_effect`):

- **true removal share** — the mean drop in conversion probability when
  every touch of a channel is deleted from every journey, computed
  directly from the DGP. This is the estimand removal-effect attribution
  actually targets, and it's the default scoring target.
- **true log-odds effect** — the per-exposure parameter, reported
  alongside it.

The two only correlate at ρ = 0.89 with each other, and the gap between a
method's two scores *is* the frequency confound, made legible.

## A real finding from the simulator (not a hypothetical)

Running the full pipeline against 8,000 simulated journeys with known
channel effects *and* known sequence structure (`journey-attribution
--mode simulate`, empirical conversion rate 23.7%):

| Method | ρ vs. removal share | ρ vs. log-odds |
|---|---|---|
| first_touch | +0.36 | −0.04 |
| **last_touch** | **+0.79** | +0.61 |
| linear / time_decay / position_based | +0.50 | +0.21 |
| **Markov removal effect** | **+0.39** (worst of the seven) | +0.07 |
| **Data-driven (LightGBM + SHAP)** | **+0.82** | +0.86 |

**The finding: Markov's removal effect is channel frequency wearing a
different hat.** Its credit ranking is a *perfect* rank-copy of raw
touchpoint volume — ρ = **1.000**, not approximately — and its score
against the correct estimand equals the volume correlation exactly:

```
rho(markov credit, channel volume)  = +1.000
rho(channel volume, removal share)  = +0.393
rho(markov credit, removal share)   = +0.393   <- identical
```

Markov contributes nothing beyond ranking channels by how often they
appear. That is not a coincidence of the channel mix: the simulator draws
channels from a first-order Markov process with genuinely
previous-channel-dependent transitions (`P(paid_search | social) = 0.438`
against a marginal of `0.222`) and pays ordered-pair bonuses on specific
consecutive transitions, so permuting a journey's channels moves its
conversion probability by 0.04 on average and up to 0.42. There is real
sequence structure sitting in the data for a sequence model to find, and
the removal effect finds none of it. This is a documented limitation of
Markov attribution (Anderl et al.) — not a bug in this implementation —
and it's exactly the kind of finding that never shows up if you only look
at real data with no ground truth to check against.

The per-channel table shows where it goes wrong:

| channel | true log-odds | true removal share | touch volume | Markov credit |
|---|---|---|---|---|
| paid_search | 0.55 | 0.358 | 8,122 | 0.203 |
| **email** | **0.70** | **0.223** | **2,978** | **0.109** |
| organic_search | 0.35 | 0.167 | 7,654 | 0.189 |
| direct | 0.40 | 0.119 | 4,746 | 0.139 |
| social | 0.10 | 0.074 | 4,653 | 0.132 |
| referral | 0.20 | 0.030 | 2,436 | 0.081 |
| **display** | **0.05** | **0.029** | **5,451** | **0.147** |

`email` is rare but the strongest channel and is second by true removal
share; Markov ranks it sixth. `display` is common and nearly worthless and
is last by removal share; Markov ranks it third. Both errors are volume.

Two things worth flagging that fall out of the same run:

- **`last_touch` scores second-best (+0.79)** — better than every other
  heuristic and far better than Markov. That's not an accident either: the
  DGP's pair effects put value on being downstream, so the last touch
  genuinely carries signal here. A crude heuristic beating the
  sophisticated method on the sophisticated method's home turf is the sort
  of result worth keeping rather than tuning away.
- **Bootstrap resampling shows the Markov ranking is extremely stable**
  (worst per-channel coefficient of variation 0.019 across 15 resamples)
  despite being the least accurate method in the table — a useful reminder
  that stability and correctness are different things.

### What the calibration check does and doesn't prove

The Markov chain's predicted overall conversion rate matches the empirical
rate exactly (0.2370 vs 0.2370, and 0.0163 vs 0.0163 on real data). This
is a **mathematical identity, not a model-fit result**: for a chain
estimated by MLE from transition counts, flow is conserved and each
journey enters `Start` exactly once, so absorption probability from `Start`
*must* equal the empirical rate — on any dataset, including degenerate
ones (a converters-only subset returns 1.000000 against an empirical
1.000000). It's a sound regression test on the linear algebra and nothing
more; it carries no information about attribution quality, and shouldn't
be read as evidence the model is right.

## Results on the real GA4 data

Having validated methodology against the simulator, the same pipeline run
against 267,084 real journeys pulled from
`bigquery-public-data.ga4_obfuscated_sample_ecommerce`
(`journey-attribution --mode real`, empirical conversion rate 1.63%):

| Method | Top channel | 2nd | 3rd |
|---|---|---|---|
| Markov removal effect | organic_search (0.29) | direct (0.23) | referral (0.21) |
| Data-driven (LightGBM + SHAP) | **other (0.43)** | referral (0.26) | organic_search (0.13) |

The simulator finding above is directly load-bearing for how to read this
table. Markov's real-data ranking (organic_search, direct, referral) tracks
raw touchpoint volume almost exactly — organic_search is 34.4% of all
touches — which is the *same* signature the simulator shows is the method
reproducing frequency rather than contribution. On real data there's no way
to tell those two apart; the simulator is the only reason we know which one
this most likely is. **The Markov column here should be read as a volume
ranking until the channel grouping is fixed** (see "Known open items").

Calibration is again exact (0.0163 vs 0.0163) — which, per the section
above, is an identity rather than evidence of fit. Cross-model agreement
between Markov and the data-driven model is moderate (Spearman ρ = 0.49) —
weaker than either method's self-consistency, and driven mostly by the
`other` channel: the
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
  generator with known per-channel effects, known ordered-pair sequence
  effects, and a first-order Markov transition process over channels. Also
  computes `true_removal_effect()` — the DGP's own counterfactual, which is
  what the attribution methods are scored against. The most important file
  in the repo.
- `src/journey_attribution/attribution/baselines.py` — first-touch,
  last-touch, linear, time-decay, position-based (U-shaped) heuristics.
- `src/journey_attribution/attribution/markov.py` — transition-probability
  chain + removal-effect attribution, via absorbing Markov chain algebra
  (fundamental matrix).
- `src/journey_attribution/attribution/datadriven.py` — LightGBM
  classifier + SHAP, channel-level credit from summed `|SHAP|` across each
  channel's features.
- `src/journey_attribution/evaluation/eval_suite.py` — simulation
  recovery (against either truth target — see "Two definitions of truth"),
  calibration, bootstrap stability, cross-model agreement.
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
- `tests/` — pytest suite: offline pipeline smoke test, per-method
  invariant checks (credits normalize to ~1.0, probabilities stay in
  range), plus DGP guards in `test_simulator.py` that assert the two
  properties everything else depends on — touchpoint timestamps are
  monotonic in generation order, and the DGP is genuinely order-dependent.
  Both failure modes produce plausible-looking numbers rather than errors,
  so nothing else would catch them. Runs in CI against the simulator only —
  no BigQuery credentials needed.

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
- The simulator's baseline conversion rate is **too high to be realistic**:
  `BASE_LOG_ODDS = -2.8` plus the positive pair effects yields 23.7%,
  against 1.63% on the real GA4 pull. Rank-recovery scores are unlikely to
  be sensitive to this, but the low-conversion regime is exactly where
  attribution methods are hardest and least stable, so the validation is
  currently running on the easy end. Should come down to roughly −4.5 and
  the recovery table should be re-run.
- The data-driven method computes SHAP on its own training data with no
  holdout, and sums `|SHAP|` — an importance measure that discards sign —
  to build channel credit. Held-out AUC is 0.64 against 0.73 in-sample, and
  the reported `train_auc` diagnostic is in-sample. Both should be fixed
  before the +0.82 recovery score is quoted as a clean win.
- Next: tighten the `other`-bucket channel grouping against the real
  `traffic_source.source`/`medium` values actually observed in the pull,
  re-run, and check whether Markov/data-driven agreement improves once
  that source of noise is reduced.
