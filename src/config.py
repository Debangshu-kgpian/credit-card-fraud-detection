"""
config.py
Central configuration for the fraud detection project:
file paths, constants, and settings used across all modules.
"""

import os

# --- Project paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# --- Raw data files ---
TRANSACTION_FILE = os.path.join(RAW_DATA_DIR, "train_transaction.csv")
IDENTITY_FILE = os.path.join(RAW_DATA_DIR, "train_identity.csv")

# --- Processed data output ---
MERGED_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, "merged_transactions.parquet")
CLEANED_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, "cleaned_transactions.parquet")
FEATURES_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, "features_transactions.parquet")

# --- Target column ---
TARGET_COL = "isFraud"

# --- Reproducibility ---
RANDOM_SEED = 42

# --- Train/test split ---
TEST_SIZE = 0.2

# --- Cost assumptions (used later for cost-sensitive threshold selection) ---
# Placeholder values — we'll refine these once we see actual fraud amount stats in EDA.
AVG_INVESTIGATION_COST = 5   # cost of manually reviewing one flagged transaction

# --- MLflow tracking ---
MLFLOW_TRACKING_URI = "sqlite:///" + os.path.join(BASE_DIR, "mlflow.db")
MLFLOW_EXPERIMENT_NAME = "fraud_detection_champion_challenger"