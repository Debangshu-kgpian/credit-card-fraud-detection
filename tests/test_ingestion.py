import pandas as pd
from src.ingestion import merge_transaction_identity


def test_merge_keeps_all_transactions():
    """Left join must keep every transaction, even ones with no identity match."""
    transactions = pd.DataFrame({
        "TransactionID": [1, 2, 3],
        "isFraud": [0, 1, 0],
    })
    identity = pd.DataFrame({
        "TransactionID": [1, 3],
        "DeviceType": ["mobile", "desktop"],
    })

    merged = merge_transaction_identity(transactions, identity)

    assert len(merged) == 3
    assert merged.loc[merged["TransactionID"] == 2, "DeviceType"].isna().all()
    assert merged.loc[merged["TransactionID"] == 1, "DeviceType"].iloc[0] == "mobile"