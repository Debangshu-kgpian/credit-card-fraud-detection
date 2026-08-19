"""
anomaly_detection.py
Unsupervised anomaly detection baseline using Isolation Forest.
Trained ONLY on non-fraud transactions (no fraud labels used during training) —
learns what "normal" transaction behavior looks like, then scores every
transaction (including fraud) by how anomalous it is.
"""

import gc
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score

from src.data_utils import load_features, split_data


def train_isolation_forest(X_train, y_train):
    """Train Isolation Forest using ONLY non-fraud transactions from the
    training set. This is what makes it 'unsupervised' in spirit even though
    labels are technically available — we deliberately withhold them from
    training and use them only afterward, for evaluation."""
    X_train_normal = X_train[y_train == 0]
    print(f"Training Isolation Forest on {X_train_normal.shape[0]} normal transactions only")

    model = IsolationForest(
        n_estimators=200,
        contamination=0.035,   # roughly matches the known fraud rate
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_normal)
    return model


def score_anomalies(model, X_test):
    """sklearn's score_samples(): HIGHER = more normal, LOWER = more anomalous.
    We flip the sign so higher = more anomalous/more fraud-like."""
    raw_scores = model.score_samples(X_test)
    return -raw_scores


def evaluate_anomaly_scores(y_test, anomaly_scores):
    auc = roc_auc_score(y_test, anomaly_scores)
    ap = average_precision_score(y_test, anomaly_scores)
    print("Isolation Forest anomaly score vs. actual fraud labels:")
    print(f"  ROC-AUC: {auc:.4f}")
    print(f"  Average Precision (AUPRC): {ap:.4f}")
    return auc, ap


def run_anomaly_detection():
    X, y = load_features()
    X_train, X_test, y_train, y_test = split_data(X, y)
    del X, y
    gc.collect()

    model = train_isolation_forest(X_train, y_train)
    anomaly_scores = score_anomalies(model, X_test)
    evaluate_anomaly_scores(y_test, anomaly_scores)

    return model, X_test, y_test, anomaly_scores


if __name__ == "__main__":
    run_anomaly_detection()