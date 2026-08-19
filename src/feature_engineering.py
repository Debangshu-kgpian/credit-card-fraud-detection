"""
feature_engineering.py
Derives additional features from the cleaned dataset and encodes
categorical columns into numeric form for modeling.
"""

import numpy as np
import pandas as pd
from src import config


def add_time_features(df):
    """Derive hour-of-day and day-of-week from TransactionDT.
    TransactionDT is seconds elapsed from an arbitrary reference point,
    not a real calendar timestamp, but it still captures cyclical patterns
    (e.g., fraud rates often differ by time of day)."""
    df["transaction_hour"] = (df["TransactionDT"] // 3600) % 24
    df["transaction_day"] = (df["TransactionDT"] // (3600 * 24)) % 7
    print("Added time features: transaction_hour, transaction_day")
    return df


def add_amount_features(df):
    """Transform TransactionAmt: log-scale (amounts are heavily right-skewed,
    which hurts many models) and isolate the cents portion — unusual cents
    patterns (e.g., suspiciously round amounts) are a known fraud signal."""
    df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])
    df["TransactionAmt_cents"] = df["TransactionAmt"] - df["TransactionAmt"].astype(int)
    print("Added amount features: TransactionAmt_log, TransactionAmt_cents")
    return df


def encode_categoricals(df):
    """Label-encode categorical (object) columns into integers.
    Tree models and sklearn's Isolation Forest both need numeric input.
    We use label encoding rather than one-hot here because several categorical
    columns (DeviceInfo, P_emaildomain) have very high cardinality — one-hot
    would explode the column count."""
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    for col in categorical_cols:
        df[col] = df[col].astype("category").cat.codes

    print(f"Label-encoded {len(categorical_cols)} categorical columns")
    return df

def optimize_dtypes(df):
    """Downcast numeric dtypes to cut memory footprint roughly in half.
    Done column-by-column (NOT via select_dtypes) — select_dtypes internally
    builds one large consolidated copy of every matching column at once,
    which needs as much memory as the columns being optimized. Looping
    column-by-column keeps peak memory usage to just one column at a time."""
    float_downcast_count = 0
    int_downcast_count = 0

    for col in df.columns:
        col_dtype = df[col].dtype
        if col_dtype == "float64":
            df[col] = df[col].astype("float32")
            float_downcast_count += 1
        elif col_dtype == "int64":
            df[col] = pd.to_numeric(df[col], downcast="integer")
            int_downcast_count += 1

    print(f"Downcast {float_downcast_count} float64 columns to float32, "
          f"{int_downcast_count} int64 columns to smaller int types")
    return df


def run_feature_engineering(df):
    """Full feature engineering pipeline."""
    df = add_time_features(df)
    df = add_amount_features(df)
    df = encode_categoricals(df)
    df = optimize_dtypes(df)
    return df


if __name__ == "__main__":
    df = pd.read_parquet(config.CLEANED_DATA_FILE)
    df = run_feature_engineering(df)
    print(f"Final shape after feature engineering: {df.shape}")

    df.to_parquet(config.FEATURES_DATA_FILE, index=False)
    print(f"Saved feature-engineered data to {config.FEATURES_DATA_FILE}")