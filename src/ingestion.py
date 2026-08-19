"""
ingestion.py
Loads raw transaction and identity CSVs, merges them on TransactionID,
and saves the merged dataset to the processed data folder.
"""

import os
import pandas as pd
from src import config


def load_transaction_data():
    """Load the raw transaction CSV."""
    print(f"Loading transaction data from {config.TRANSACTION_FILE} ...")
    df = pd.read_csv(config.TRANSACTION_FILE)
    print(f"  -> {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def load_identity_data():
    """Load the raw identity CSV."""
    print(f"Loading identity data from {config.IDENTITY_FILE} ...")
    df = pd.read_csv(config.IDENTITY_FILE)
    print(f"  -> {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def merge_transaction_identity(transaction_df, identity_df):
    """Left-join identity data onto transaction data using TransactionID.
    Left join keeps ALL transactions (since that's where isFraud lives),
    even the ~76% that have no matching identity record."""
    print("Merging transaction and identity data on TransactionID ...")
    merged_df = transaction_df.merge(identity_df, on="TransactionID", how="left")
    print(f"  -> merged shape: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns")
    return merged_df


def save_merged_data(df):
    """Save the merged dataset as a parquet file in the processed folder."""
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    df.to_parquet(config.MERGED_DATA_FILE, index=False)
    print(f"Saved merged data to {config.MERGED_DATA_FILE}")


def run_ingestion():
    """Full ingestion pipeline: load, merge, and save."""
    transaction_df = load_transaction_data()
    identity_df = load_identity_data()
    merged_df = merge_transaction_identity(transaction_df, identity_df)
    save_merged_data(merged_df)
    return merged_df


if __name__ == "__main__":
    run_ingestion()