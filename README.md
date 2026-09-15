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

Python, pandas, scikit-learn, XGBoost, Optuna, MLflow, matplotlib, joblib, FastAPI, Uvicorn, Docker, pytest, GitHub Actions, Render

## API & Deployment

The trained XGBoost model is served as a live REST API, built with **FastAPI**, containerized with **Docker**, and deployed on **Render**.

**Live API**: `https://credit-card-fraud-detection-sujt.onrender.com`

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Basic liveness check |
| `/predict` | POST | Accepts a transaction's engineered features, returns a fraud probability and flag using the cost-optimal 0.07 threshold |

Example request:
```bash
curl -X POST https://credit-card-fraud-detection-sujt.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"TransactionAmt": 107.95, "...": "..."}}'
```

Example response:
```json
{"fraud_probability": 0.0006, "is_fraud": false, "threshold_used": 0.07}
```

## Testing & CI/CD

- **Unit tests** (`pytest`, in `tests/`) validate core pipeline logic (e.g., the transaction/identity join never drops rows, the cost-based threshold math matches hand-calculated expected values) against small, hand-built fixture data — not the real dataset.
- **CI**: a GitHub Actions workflow (`.github/workflows/ci-cd.yml`) runs the full test suite on every push to `main`.
- **CD**: Render is connected directly to this GitHub repo and automatically rebuilds and redeploys the Docker container on every push to `main` that passes CI.

## Monitoring

Every `/predict` request is logged (`src/api.py`, `logs/prediction_log.jsonl`). A separate script (`src/monitoring.py`) periodically compares the logged requests' feature distributions against the training data (z-scores on a set of representative features) and logs a drift score into MLflow as a `monitoring`-tagged run — a lightweight foundation for detecting when live traffic starts to look statistically different from training data.

## Dataset

[IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) (Kaggle) — real anonymized credit card transaction data with identity/device features.

## Running it

