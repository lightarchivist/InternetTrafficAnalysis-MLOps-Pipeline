"""

Flask REST API for network anomaly detection.
Loads model from MLflow Model Registry at startup.

Usage:
    python app.py

Test:
    curl http://localhost:8089/health
    curl -X POST http://localhost:8089/predict \
         -H "Content-Type: application/json" \
         -d '{"flag_syn": 1, "flag_ack": 0, "payload_len": 0}'
"""

import logging
import os
import sys

import mlflow
import mlflow.sklearn
import numpy as np
from flask import Flask, jsonify, request

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app          = Flask(__name__)
model        = None
scaler       = None
svd          = None
feature_cols = None

#  Config 
MLFLOW_URI   = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5555")
MODEL_NAME   = os.environ.get("MODEL_NAME",           "network-anomaly-svm")
MODEL_STAGE  = os.environ.get("MODEL_STAGE",           "Production")


def load_model():
    """Load scaler, SVD and SVM from MLflow Model Registry."""
    global model, scaler, svd, feature_cols

    mlflow.set_tracking_uri(MLFLOW_URI)
    log.info(f"Loading {MODEL_STAGE} model '{MODEL_NAME}' from {MLFLOW_URI}")

    client   = mlflow.tracking.MlflowClient()
    model    = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")
    versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
    run_id   = versions[0].run_id

    scaler = mlflow.sklearn.load_model(f"runs:/{run_id}/scaler")
    svd    = mlflow.sklearn.load_model(f"runs:/{run_id}/svd")

    local_path = client.download_artifacts(run_id, "feature_cols.txt")
    with open(local_path) as f:
        feature_cols = [line.strip() for line in f if line.strip()]

    log.info(f"Model loaded. Expecting {len(feature_cols)} features.")


#  Routes 

@app.route("/health")
def health():
    if model is None:
        return jsonify({"status": "loading"}), 503
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "Empty request body"}), 400

    try:
        X = np.array(
            [float(data.get(col, 0)) for col in feature_cols]
        ).reshape(1, -1)
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Feature error: {e}"}), 422

    X_scaled  = scaler.transform(X)
    X_reduced = svd.transform(X_scaled)
    label     = int(model.predict(X_reduced)[0])
    prob      = float(model.predict_proba(X_reduced)[0][1])

    return jsonify({
        "prediction":  "anomaly" if label == 1 else "normal",
        "label":        label,
        "probability":  round(prob, 4),
    })


@app.route("/model-info")
def model_info():
    return jsonify({
        "model_name":    MODEL_NAME,
        "model_stage":   MODEL_STAGE,
        "feature_count": len(feature_cols) if feature_cols else None,
        "model_type":    type(model).__name__ if model else None,
    })


#  Startup 

if __name__ == "__main__":
    load_model()
    port = int(os.environ.get("PORT", 8089))
    log.info(f"Starting Flask on port {port}")
    app.run(host="0.0.0.0", port=port)