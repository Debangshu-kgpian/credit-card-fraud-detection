"""
preprocessing.py
Cleans the merged transaction+identity dataset:
- creates missingness-pattern indicator features (before dropping sparse columns)
- drops columns that are too sparse to be useful as raw features
- imputes remaining missing values
"""

import numpy as np
import pandas as pd
from src import config

# Columns above this missing-fraction are considered too sparse to keep as raw features
MISSING_THRESHOLD = 0.90


def add_missingness_indicators(df):
    """Create summary features that capture the *pattern* of missingness
    before we drop the underlying sparse identity columns."""
    id_cols = [c for c in df.columns if c.startswith("id_")]

    # Whether this transaction has ANY identity data at all
    df["has_identity_data"] = df[id_cols].notnull().any(axis=1).astype(int)

    # How many identity fields are populated (captures "how complete" the profile is)
    df["id_fields_populated"] = df[id_cols].notnull().sum(axis=1)

    print("Added missingness indicators: has_identity_data, id_fields_populated")
    return df


def drop_sparse_columns(df, threshold=MISSING_THRESHOLD):
    """Drop columns whose missing-value fraction exceeds the threshold."""
    missing_frac = df.isnull().mean()
    sparse_cols = missing_frac[missing_frac > threshold].index.tolist()
    sparse_cols = [c for c in sparse_cols if c != config.TARGET_COL]  # never drop the target

    print(f"Dropping {len(sparse_cols)} columns with >{threshold*100:.0f}% missing values")
    df = df.drop(columns=sparse_cols)
    return df, sparse_cols


def impute_missing_values(df):
    """Impute remaining missing values: median for numeric, 'unknown' for categorical."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    if config.TARGET_COL in numeric_cols:
        numeric_cols.remove(config.TARGET_COL)

    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna("unknown")

    print(f"Imputed {len(numeric_cols)} numeric columns (median) and "
          f"{len(categorical_cols)} categorical columns ('unknown')")
    return df


def run_preprocessing(df):
    """Full preprocessing pipeline."""
    df = add_missingness_indicators(df)
    df, dropped_cols = drop_sparse_columns(df)
    df = impute_missing_values(df)
    return df, dropped_cols


if __name__ == "__main__":
    df = pd.read_parquet(config.MERGED_DATA_FILE)
    df, dropped_cols = run_preprocessing(df)
    print(f"Final shape after preprocessing: {df.shape}")

    df.to_parquet(config.CLEANED_DATA_FILE, index=False)
    print(f"Saved cleaned data to {config.CLEANED_DATA_FILE}")