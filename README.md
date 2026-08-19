# Credit Card Fraud Detection — Hybrid Anomaly Detection & Cost-Sensitive Classification

A fraud detection pipeline built on the IEEE-CIS Fraud Detection dataset (590K real-world credit card transactions, 3.5% fraud rate), combining two distinct ML paradigms — unsupervised anomaly detection and supervised, cost-sensitive classification — with business-aware threshold optimization and model explainability.

## Why two models

Most fraud detection portfolios stop at "trained a classifier." This project deliberately pairs two approaches that solve different problems:

- **Isolation Forest (challenger)** — unsupervised, trained only on non-fraud transactions, with no access to fraud labels during training. Acts as a safety net for novel fraud patterns a supervised model has never seen.
- **XGBoost (champion)** — supervised, Optuna-tuned, trained directly on historical fraud labels. Far more precise on known fraud patterns, and the primary production model.

## Results

| Model | ROC-AUC | AUPRC |
|---|---|---|
| Isolation Forest (unsupervised) | 0.7675 | 0.1376 |
| XGBoost (Optuna-tuned, supervised) | 0.9700 | 0.8504 |

Isolation Forest's AUPRC of 0.1376 is ~4x the base fraud rate (0.035) — real signal, but far weaker than the supervised model, exactly as expected: it never sees fraud labels during training.

**Cost-based threshold optimization**: rather than using the default 0.5 probability threshold, the decision threshold was chosen to minimize total expected business cost (missed fraud = full transaction amount lost; false alarm = fixed investigation cost). This shifted the optimal threshold to 0.07 and reduced expected cost by **35.1%** versus the naive default.

**Feature importance**: the top predictive features (V258, V70, V218, V91, V201) are Vesta's proprietary engineered features — anonymized, so their exact business meaning is undisclosed, but they align with what's been reported across public solutions to this dataset, validating the model is learning genuine signal.

## Pipeline


Every stage is modular (`src/`), config-driven (`config.py`), and reproducible via a fixed random seed.

## Experiment tracking

Experiments are tracked in MLflow (local SQLite backend), logging both models under one experiment with `challenger`/`champion` tags, and registering the XGBoost model in the MLflow Model Registry.


## Tech stack

Python, pandas, scikit-learn, XGBoost, Optuna, MLflow, matplotlib, joblib

## Dataset

[IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) (Kaggle) — real anonymized credit card transaction data with identity/device features.

## Running it

