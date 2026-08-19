"""
evaluation.py
Cost-based threshold selection for the XGBoost fraud classifier.
Instead of the default 0.5 probability threshold, we pick the threshold
that MINIMIZES total expected business cost:
    cost = (missed fraud $ amount) + (false alarms x investigation cost)
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

from src import config
from src.data_utils import load_features, split_data


def load_model_and_test_set():
    """Reload the SAME test set (deterministic via fixed random_state +
    stratify) and the trained model — split_data() reproduces the identical
    split every time, so no need to persist X_test/y_test separately."""
    X, y = load_features()
    _, X_test, _, y_test = split_data(X, y)
    del X, y

    model_path = os.path.join(config.MODELS_DIR, "xgboost_fraud_model.joblib")
    model = joblib.load(model_path)

    test_scores = model.predict_proba(X_test)[:, 1]
    return X_test, y_test, test_scores


def compute_cost_at_threshold(y_test, test_scores, amounts, threshold, investigation_cost):
    """Total expected cost at a given threshold:
    - False negatives (missed fraud): cost = the actual dollar amount lost
    - False positives (false alarms): cost = a fixed investigation cost each
    """
    predicted_positive = test_scores >= threshold

    fn_mask = (y_test == 1) & (~predicted_positive)
    fp_mask = (y_test == 0) & predicted_positive

    fn_cost = amounts[fn_mask].sum()
    fp_cost = fp_mask.sum() * investigation_cost

    return fn_cost + fp_cost, fn_mask.sum(), fp_mask.sum()


def find_optimal_threshold(y_test, test_scores, amounts, investigation_cost=config.AVG_INVESTIGATION_COST):
    """Sweep thresholds and find the one minimizing total expected cost."""
    thresholds = np.linspace(0.01, 0.99, 99)
    results = []

    for t in thresholds:
        cost, n_fn, n_fp = compute_cost_at_threshold(y_test, test_scores, amounts, t, investigation_cost)
        results.append({"threshold": t, "cost": cost, "n_fn": n_fn, "n_fp": n_fp})

    results_df = pd.DataFrame(results)
    best_row = results_df.loc[results_df["cost"].idxmin()]

    print(f"Optimal threshold: {best_row['threshold']:.2f}")
    print(f"  Expected cost at optimal threshold: ${best_row['cost']:,.2f}")
    print(f"  Missed fraud (FN): {int(best_row['n_fn'])}, False alarms (FP): {int(best_row['n_fp'])}")

    return best_row["threshold"], results_df


def compare_to_default_threshold(y_test, test_scores, amounts, optimal_threshold,
                                  investigation_cost=config.AVG_INVESTIGATION_COST):
    """Compare cost and classification metrics at 0.5 (naive default)
    vs. the cost-optimal threshold — quantifies the business value of
    doing threshold selection properly instead of using sklearn's default."""
    default_cost, default_fn, default_fp = compute_cost_at_threshold(
        y_test, test_scores, amounts, 0.5, investigation_cost
    )
    optimal_cost, optimal_fn, optimal_fp = compute_cost_at_threshold(
        y_test, test_scores, amounts, optimal_threshold, investigation_cost
    )

    savings = default_cost - optimal_cost
    savings_pct = (savings / default_cost) * 100 if default_cost > 0 else 0

    print("\n--- Threshold comparison ---")
    print(f"Default (0.5)   -> cost: ${default_cost:,.2f}  (FN={default_fn}, FP={default_fp})")
    print(f"Cost-optimal    -> cost: ${optimal_cost:,.2f}  (FN={optimal_fn}, FP={optimal_fp})")
    print(f"Estimated savings from cost-based threshold: ${savings:,.2f} ({savings_pct:.1f}%)")

    for label, t in [("Default (0.5)", 0.5), ("Cost-optimal", optimal_threshold)]:
        preds = (test_scores >= t).astype(int)
        print(f"\n{label} threshold={t:.3f}")
        print(f"  Precision: {precision_score(y_test, preds):.4f}")
        print(f"  Recall:    {recall_score(y_test, preds):.4f}")
        print(f"  F1:        {f1_score(y_test, preds):.4f}")

    return savings, savings_pct


def plot_cost_curve(results_df, optimal_threshold):
    """Save a plot of total cost vs. threshold, marking the optimal point."""
    plt.figure(figsize=(8, 5))
    plt.plot(results_df["threshold"], results_df["cost"])
    plt.axvline(optimal_threshold, color="red", linestyle="--",
                label=f"Optimal threshold = {optimal_threshold:.2f}")
    plt.xlabel("Decision threshold")
    plt.ylabel("Total expected cost ($)")
    plt.title("Expected Cost vs. Decision Threshold")
    plt.legend()

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    plot_path = os.path.join(config.MODELS_DIR, "cost_vs_threshold.png")
    plt.savefig(plot_path, bbox_inches="tight")
    print(f"\nSaved cost-vs-threshold plot to {plot_path}")
    plt.close()


def run_evaluation():
    """Full cost-based evaluation pipeline."""
    X_test, y_test, test_scores = load_model_and_test_set()
    amounts = X_test["TransactionAmt"].values

    optimal_threshold, results_df = find_optimal_threshold(y_test.values, test_scores, amounts)
    compare_to_default_threshold(y_test.values, test_scores, amounts, optimal_threshold)
    plot_cost_curve(results_df, optimal_threshold)

    return optimal_threshold, results_df


if __name__ == "__main__":
    run_evaluation()