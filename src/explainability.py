"""
explainability.py
Global feature importance for the trained XGBoost fraud classifier.
(SHAP-based explainability was attempted but hit a version-compatibility
bug between XGBoost 2.x and the installed SHAP release — this uses
XGBoost's built-in importance as a reliable fallback.)
"""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from src import config
from src.data_utils import load_features, split_data


def load_model_and_test_set():
    X, y = load_features()
    _, X_test, _, y_test = split_data(X, y)
    del X, y

    model_path = os.path.join(config.MODELS_DIR, "xgboost_fraud_model.joblib")
    model = joblib.load(model_path)

    return model, X_test, y_test


def plot_feature_importance(model, X_test, top_n=20):
    """XGBoost's built-in importance: 'gain' measures the average
    improvement in accuracy each feature brings when it's used in a split —
    generally the most informative of the built-in importance types."""
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance": importances
    }).sort_values("importance", ascending=False).head(top_n)

    plt.figure(figsize=(8, 8))
    plt.barh(importance_df["feature"][::-1], importance_df["importance"][::-1])
    plt.xlabel("Importance (gain)")
    plt.title(f"Top {top_n} Features Driving Fraud Predictions")

    plot_path = os.path.join(config.MODELS_DIR, "feature_importance.png")
    plt.savefig(plot_path, bbox_inches="tight")
    print(f"Saved feature importance plot to {plot_path}")
    plt.close()

    print("\nTop 10 features:")
    print(importance_df.head(10).to_string(index=False))

    return importance_df


def run_explainability():
    """Feature importance pipeline (SHAP fallback)."""
    model, X_test, y_test = load_model_and_test_set()
    importance_df = plot_feature_importance(model, X_test)
    return importance_df


if __name__ == "__main__":
    run_explainability()