"""
monitoring.py
Reads the accumulated prediction log and checks whether the live
transactions look statistically different from the training data —
a simple drift check, not a real-time system. Run this periodically
(manually, or as a scheduled job) to log a drift score into MLflow.
"""

import os
import json
import mlflow
import pandas as pd

from src import config
from src.data_utils import load_features, split_data

mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

LOG_PATH = os.path.join("logs", "prediction_log.jsonl")

# A handful of representative features to check drift on, rather than all 426 —
# a mix of business-meaningful ones and a top feature from feature_importances_.
DRIFT_FEATURES = ["TransactionAmt", "TransactionAmt_log", "C13", "V258"]


def load_logged_requests():
    """Load every logged prediction request's feature values into a DataFrame."""
    rows = []
    with open(LOG_PATH, "r") as f:
        for line in f:
            entry = json.loads(line)
            rows.append(entry["features"])
    return pd.DataFrame(rows)


def compute_drift_scores(live_df, train_df):
    """For each tracked feature, compute a z-score: how many training-set
    standard deviations away is the live data's mean? Large |z| = drift."""
    scores = {}
    for feature in DRIFT_FEATURES:
        train_mean = train_df[feature].mean()
        train_std = train_df[feature].std()
        live_mean = live_df[feature].mean()

        z_score = (live_mean - train_mean) / train_std if train_std > 0 else 0.0
        scores[feature] = z_score
    return scores


def run_monitoring_check():
    live_df = load_logged_requests()
    print(f"Loaded {len(live_df)} logged prediction requests")

    X, y = load_features()
    X_train, _, _, _ = split_data(X, y)

    drift_scores = compute_drift_scores(live_df, X_train)

    with mlflow.start_run(run_name="drift_check"):
        mlflow.set_tag("model_role", "monitoring")
        mlflow.log_param("n_requests_checked", len(live_df))

        for feature, z in drift_scores.items():
            print(f"  {feature}: z-score = {z:.3f}")
            mlflow.log_metric(f"drift_zscore_{feature}", z)

        overall = sum(abs(z) for z in drift_scores.values()) / len(drift_scores)
        print(f"Overall average |z-score|: {overall:.3f}")
        mlflow.log_metric("drift_overall_avg_abs_zscore", overall)

    return drift_scores


if __name__ == "__main__":
    run_monitoring_check()