"""
Extracts raw touchpoint + conversion data from the public GA4 sample
dataset (bigquery-public-data.ga4_obfuscated_sample_ecommerce) and writes
it to data/raw_touchpoints.csv for journey_builder.py to consume.

Requires: a Google Cloud project with the BigQuery API enabled (free
sandbox tier is sufficient at this data volume — you are not querying the
full 4.3M-event dataset repeatedly, and BigQuery's public-dataset billing
only charges for bytes scanned by *your* project).

    pip install google-cloud-bigquery
    gcloud auth application-default login
    gcloud config set project <your-gcp-project-id>
    python extract_bigquery.py --project <your-gcp-project-id>

Known wrinkle, stated plainly rather than papered over: this is an
*obfuscated* sample dataset — Google's own documentation notes some fields
contain placeholder values and "internal consistency might be somewhat
limited." The channel-grouping CASE statement below mirrors Google's
default channel grouping logic but is a simplification. Expect to spend
real time on data-quality triage once you actually pull this — that's
part of the job, not a bug in this script.
"""
from __future__ import annotations
import argparse
import pandas as pd

QUERY = """
-- One row per (user, event) where the event is either a touchpoint-worthy
-- session start (carrying a traffic source) or a purchase (the conversion
-- event). Channel grouping is a simplified version of Google's default
-- channel grouping rules, applied to traffic_source.source/medium.
WITH events AS (
  SELECT
    user_pseudo_id,
    event_name,
    TIMESTAMP_MICROS(event_timestamp) AS event_ts,
    traffic_source.source AS source,
    traffic_source.medium AS medium
  FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
  WHERE _TABLE_SUFFIX BETWEEN '20201101' AND '20210131'
    AND event_name IN ('session_start', 'purchase')
),
channel_grouped AS (
  SELECT
    user_pseudo_id,
    event_name,
    event_ts,
    CASE
      WHEN medium = 'cpc' OR medium = 'ppc' OR medium = 'paidsearch'      THEN 'paid_search'
      WHEN medium = 'organic'                                            THEN 'organic_search'
      WHEN medium = 'email'                                              THEN 'email'
      WHEN medium = 'referral'                                           THEN 'referral'
      WHEN LOWER(source) IN ('facebook','instagram','twitter','pinterest',
                              'linkedin','tiktok')                        THEN 'social'
      WHEN medium = '(none)' OR source = '(direct)'                      THEN 'direct'
      WHEN medium IS NULL OR medium = '' OR medium = '<Other>'           THEN 'display'
      ELSE 'other'
    END AS channel
  FROM events
)
SELECT user_pseudo_id, event_name, event_ts, channel
FROM channel_grouped
ORDER BY user_pseudo_id, event_ts
"""


def main(project_id: str, output_path: str) -> None:
    from google.cloud import bigquery  # deferred import — optional dependency

    client = bigquery.Client(project=project_id)
    print(f"Running query against project {project_id} ...")
    df = client.query(QUERY).to_dataframe()
    print(f"Pulled {len(df)} rows, {df['user_pseudo_id'].nunique()} unique users.")
    df.to_csv(output_path, index=False)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, help="Your GCP project ID")
    parser.add_argument("--output", default="data/raw_touchpoints.csv")
    args = parser.parse_args()
    main(args.project, args.output)
