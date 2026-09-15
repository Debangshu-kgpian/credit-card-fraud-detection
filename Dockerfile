FROM python:3.10-slim

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ ./src/
COPY models/xgboost_fraud_model.joblib ./models/xgboost_fraud_model.joblib

EXPOSE 7860

#CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "7860"]
CMD uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-7860}