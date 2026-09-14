"""
anomaly_detection.py
Unsupervised anomaly detection baseline using Isolation Forest.
Trained ONLY on non-fraud transactions (no fraud labels used during training) —
learns what "normal" transaction behavior looks like, then scores every
transaction (including fraud) by how anomalous it is.
"""

import gc
import mlflow
import mlflow.sklearn
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score

from src import config
from src.data_utils import load_features, split_data

mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)


def train_isolation_forest(X_train, y_train, params):
    """Train Isolation Forest using ONLY non-fraud transactions from the
    training set. This is what makes it 'unsupervised' in spirit even though
    labels are technically available — we deliberately withhold them from
    training and use them only afterward, for evaluation."""
    X_train_normal = X_train[y_train == 0]
    print(f"Training Isolation Forest on {X_train_normal.shape[0]} normal transactions only")

    model = IsolationForest(**params)
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

    params = {
        "n_estimators": 200,
        "contamination": 0.035,
        "random_state": 42,
        "n_jobs": -1,
    }

    with mlflow.start_run(run_name="isolation_forest_challenger"):
        mlflow.set_tag("model_role", "challenger")
        mlflow.log_params(params)

        model = train_isolation_forest(X_train, y_train, params)
        anomaly_scores = score_anomalies(model, X_test)
        auc, ap = evaluate_anomaly_scores(y_test, anomaly_scores)

        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("average_precision", ap)
        mlflow.sklearn.log_model(model, "model")

        print(f"Logged run to MLflow experiment: {config.MLFLOW_EXPERIMENT_NAME}")

    return model, X_test, y_test, anomaly_scores


if __name__ == "__main__":
    run_anomaly_detection()