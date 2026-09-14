"""
test_api_request.py
Quick script to test the /predict endpoint using one real transaction
from the held-out test set.
"""

import requests
from src.data_utils import load_features, split_data

X, y = load_features()
_, X_test, _, y_test = split_data(X, y)

# grab the first test transaction, converted to plain Python floats (JSON-safe)
sample_row = X_test.iloc[0].astype(float).to_dict()
actual_label = int(y_test.iloc[0])

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"features": sample_row}
)

print("Actual label (isFraud):", actual_label)
print("API response:", response.json())