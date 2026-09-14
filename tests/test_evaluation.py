import numpy as np
from src.evaluation import compute_cost_at_threshold


def test_compute_cost_at_threshold():
    y_test = np.array([1, 0, 1, 0])          # fraud, legit, fraud, legit
    test_scores = np.array([0.9, 0.2, 0.05, 0.6])
    amounts = np.array([100, 50, 200, 30])
    threshold = 0.5
    investigation_cost = 5

    cost, n_fn, n_fp = compute_cost_at_threshold(
        y_test, test_scores, amounts, threshold, investigation_cost
    )

    # Index 2 (fraud, score 0.05 < 0.5) is missed -> FN, costs its $200 amount
    # Index 3 (legit, score 0.6 >= 0.5) is a false alarm -> FP, costs $5
    assert n_fn == 1
    assert n_fp == 1
    assert cost == 205