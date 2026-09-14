"""
model_training.py
Supervised fraud classifier: cost-sensitive XGBoost, hyperparameter-tuned with Optuna.
Uses the SAME train/test split as anomaly_detection.py (via data_utils),
so both models are compared on identical, held-out data.
"""

import os
import gc
import joblib
import mlflow
import mlflow.xgboost
import optuna
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score

from src import config
from src.data_utils import load_features, split_data

mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)


def make_validation_split(X_train, y_train):
    """Carve a validation set out of the training data, used ONLY for
    Optuna hyperparameter tuning. The original held-out test set stays
    completely untouched until final evaluation, so hyperparameter
    choices can't leak into the final reported numbers."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2,
        random_state=config.RANDOM_SEED,
        stratify=y_train
    )
    print(f"Sub-split for tuning -> train: {X_tr.shape}, val: {X_val.shape}")
    return X_tr, X_val, y_tr, y_val


def compute_scale_pos_weight(y):
    """XGBoost's built-in way to handle class imbalance: weight the minority
    (fraud) class higher during training, proportional to how outnumbered
    it is. Chosen over SMOTE-style oversampling, which would need to
    generate synthetic rows for the minority class — expensive in memory
    on a 400+ column dataset, and unnecessary given XGBoost's native option."""
    neg = (y == 0).sum()
    pos = (y == 1).sum()
    weight = neg / pos
    print(f"scale_pos_weight = {weight:.2f} (neg={neg}, pos={pos})")
    return weight


def objective(trial, X_tr, y_tr, X_val, y_val, spw):
    """Optuna objective: train an XGBoost model with trial-suggested
    hyperparameters, score it on the validation set using Average Precision
    (AUPRC) — the metric that matters most given the class imbalance."""
    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "scale_pos_weight": spw,
        "random_state": config.RANDOM_SEED,
        "n_estimators": trial.suggest_int("n_estimators", 100, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X_tr, y_tr, verbose=False)

    val_scores = model.predict_proba(X_val)[:, 1]
    return average_precision_score(y_val, val_scores)


def run_optuna_tuning(X_tr, y_tr, X_val, y_val, spw, n_trials=20):
    """Run Optuna to search for the best hyperparameters, maximizing validation AUPRC."""
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, X_tr, y_tr, X_val, y_val, spw),
        n_trials=n_trials
    )
    print(f"Best validation AUPRC: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    return study.best_params


def train_final_model(X_train, y_train, best_params, spw):
    """Train the final XGBoost model on the FULL training set (not just the
    tuning sub-split), using the best hyperparameters Optuna found."""
    params = dict(best_params)
    params.update({
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "scale_pos_weight": spw,
        "random_state": config.RANDOM_SEED,
    })
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Final evaluation on the untouched test set — the same test set
    anomaly_detection.py was evaluated on, for a fair comparison."""
    test_scores = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, test_scores)
    ap = average_precision_score(y_test, test_scores)

    print("XGBoost classifier vs. actual fraud labels (held-out test set):")
    print(f"  ROC-AUC: {auc:.4f}")
    print(f"  Average Precision (AUPRC): {ap:.4f}")
    return auc, ap, test_scores


def run_model_training(n_trials=20):
    """Full supervised training pipeline."""
    X, y = load_features()
    X_train, X_test, y_train, y_test = split_data(X, y)
    del X, y
    gc.collect()

    spw = compute_scale_pos_weight(y_train)
    X_tr, X_val, y_tr, y_val = make_validation_split(X_train, y_train)

    with mlflow.start_run(run_name="xgboost_champion"):
        mlflow.set_tag("model_role", "champion")

        best_params = run_optuna_tuning(X_tr, y_tr, X_val, y_val, spw, n_trials=n_trials)
        del X_tr, X_val, y_tr, y_val
        gc.collect()

        mlflow.log_params(best_params)
        mlflow.log_param("scale_pos_weight", spw)
        mlflow.log_param("n_trials", n_trials)

        model = train_final_model(X_train, y_train, best_params, spw)
        auc, ap, test_scores = evaluate_model(model, X_test, y_test)

        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("average_precision", ap)

        mlflow.xgboost.log_model(
            model,
            "model",
            registered_model_name="fraud_xgboost_champion"
        )

        os.makedirs(config.MODELS_DIR, exist_ok=True)
        model_path = os.path.join(config.MODELS_DIR, "xgboost_fraud_model.joblib")
        joblib.dump(model, model_path)
        print(f"Saved trained model to {model_path}")

        print(f"Logged and registered run to MLflow experiment: {config.MLFLOW_EXPERIMENT_NAME}")

    return model, X_test, y_test, test_scores


if __name__ == "__main__":
    run_model_training(n_trials=20)