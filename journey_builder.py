"""
Turns the raw CSV from extract_bigquery.py into typed Journey objects.

Design decision worth flagging explicitly: touchpoints AFTER a user's first
purchase in the window are dropped. Without this, a returning customer's
second-purchase journey gets contaminated with touchpoints that happened
after they'd already converted once — which would bias every attribution
method here, not just Markov. This is exactly the kind of assumption that
needs to be stated, not buried in code.
"""
from __future__ import annotations
import pandas as pd
from schemas import Touchpoint, Journey


def build_journeys_from_csv(path: str = "data/raw_touchpoints.csv") -> list[Journey]:
    df = pd.read_csv(path, parse_dates=["event_ts"])
    journeys: list[Journey] = []

    for user_id, group in df.groupby("user_pseudo_id"):
        group = group.sort_values("event_ts")
        purchase_rows = group[group["event_name"] == "purchase"]
        converted = len(purchase_rows) > 0
        cutoff = purchase_rows["event_ts"].iloc[0] if converted else None

        touchpoint_rows = group[group["event_name"] == "session_start"]
        if converted:
            touchpoint_rows = touchpoint_rows[touchpoint_rows["event_ts"] <= cutoff]
        if touchpoint_rows.empty:
            continue  # no session_start touchpoints for this user in-window — skip

        touchpoints = [
            Touchpoint(channel=row["channel"], timestamp=row["event_ts"])
            for _, row in touchpoint_rows.iterrows()
        ]
        journeys.append(Journey(
            user_id=str(user_id),
            touchpoints=touchpoints,
            converted=converted,
            conversion_value=None,  # GA4 obfuscated sample doesn't reliably expose revenue
        ))

    return journeys
