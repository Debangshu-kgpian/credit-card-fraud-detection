"""
data_utils.py
Shared data-loading and train/test-splitting logic used by BOTH the
anomaly detection model and the supervised classifier — kept in one place
so both models are guaranteed to train/evaluate on the exact same split.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


def load_features():
    """Load the feature-engineered dataset and split into X (features) and y (target)."""
    df = pd.read_parquet(config.FEATURES_DATA_FILE)

    drop_cols = ["TransactionID", config.TARGET_COL]
    X = df.drop(columns=drop_cols)
    y = df[config.TARGET_COL]

    print(f"Loaded features: X={X.shape}, y={y.shape}")
    return X, y


def split_data(X, y):
    """Stratified train/test split — keeps the ~3.5% fraud ratio consistent
    in both sets. Same seed + stratify everywhere this is called ensures
    every model in this project is evaluated on the identical test set."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=y
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Train fraud rate: {y_train.mean()*100:.3f}%, Test fraud rate: {y_test.mean()*100:.3f}%")
    return X_train, X_test, y_train, y_test