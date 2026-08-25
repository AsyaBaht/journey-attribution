"""
Data-driven attribution: a LightGBM classifier predicts conversion from
journey features, SHAP values attribute credit per channel. Same
explainability tool as the segmentation project — deliberate reuse, not
coincidence.

Feature design choice: one feature per channel (count of exposures), plus
first/last-touch indicators. This lets SHAP attribute credit at the
channel level directly, which is what we need to compare against Markov's
per-channel removal effect on equal footing.

Author: Anastasiia Bakhtoiarova
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from journey_attribution.schemas import Journey, AttributionResult


def _feature_frame(journeys: list[Journey]) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    channels = sorted({c for j in journeys for c in j.channels_in_order})
    rows = []
    labels = []
    for j in journeys:
        order = j.channels_in_order
        counts = {c: order.count(c) for c in channels}
        row = dict(counts)
        row["journey_length"] = len(order)
        row["n_unique_channels"] = len(set(order))
        for c in channels:
            row[f"is_first_{c}"] = int(bool(order) and order[0] == c)
            row[f"is_last_{c}"] = int(bool(order) and order[-1] == c)
        rows.append(row)
        labels.append(int(j.converted))
    df = pd.DataFrame(rows).fillna(0)
    return df, channels, np.array(labels)


def datadriven_attribution(journeys: list[Journey], seed: int = 42) -> tuple[AttributionResult, dict]:
    """Returns the attribution result plus a small diagnostics dict
    (train AUC) so callers can sanity-check the model actually learned
    something before trusting its SHAP-based attribution."""
    df, channels, y = _feature_frame(journeys)

    model = lgb.LGBMClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        random_state=seed, verbose=-1,
    )
    model.fit(df, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df)
    # LightGBM binary classifier: shap_values is a single array (impact on
    # positive class) in recent shap versions, or a list [neg, pos] in
    # older ones — handle both.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Sum |SHAP| across all features belonging to each channel (the raw
    # count feature plus its is_first/is_last indicators) to get one
    # credit number per channel, comparable to Markov's per-channel credit.
    channel_importance: dict[str, float] = {c: 0.0 for c in channels}
    for i, col in enumerate(df.columns):
        for c in channels:
            if col == c or col == f"is_first_{c}" or col == f"is_last_{c}":
                channel_importance[c] += float(np.abs(shap_values[:, i]).sum())

    total = sum(channel_importance.values())
    credits = {k: (v / total if total > 0 else 0.0) for k, v in channel_importance.items()}

    train_auc = float(model.predict_proba(df)[:, 1].round().mean())  # rough diagnostic only
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(y, model.predict_proba(df)[:, 1]))

    return AttributionResult(method="datadriven_shap", credits=credits), {"train_auc": auc}
