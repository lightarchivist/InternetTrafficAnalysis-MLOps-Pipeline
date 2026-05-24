"""
app.py

Flask REST API for network anomaly detection.
Imports your parsing functions directly from main.py.
Loads the Production model from MLflow at startup.

"""

import logging
import os
import sys

import mlflow
import mlflow.sklearn
import numpy as np
from flask import Flask, jsonify, request

# Import your parsing functions from main.py 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from main import expand_info_column, flag_combo_label

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app          = Flask(__name__)
model        = None
scaler       = None
svd          = None
feature_cols = None


def load_model():
    """Load Production model, scaler and SVD from MLflow at startup."""
    global model, scaler, svd, feature_cols

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5555")
    model_name   = os.environ.get("MODEL_NAME",           "network-anomaly-svm")
    model_stage  = os.environ.get("MODEL_STAGE",           "Production")

    mlflow.set_tracking_uri(tracking_uri)
    log.info(f"Loading {model_stage} model '{model_name}' from {tracking_uri}")

    # Load SVM from registry
    model    = mlflow.sklearn.load_model(f"models:/{model_name}/{model_stage}")

    # Load scaler and SVD from the same run
    client   = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions(model_name, stages=[model_stage])
    run_id   = versions[0].run_id

    scaler   = mlflow.sklearn.load_model(f"runs:/{run_id}/scaler")
    svd      = mlflow.sklearn.load_model(f"runs:/{run_id}/svd")

    # Load feature column order saved during training
    local_path   = client.download_artifacts(run_id, "feature_cols.txt")
    with open(local_path) as f:
        feature_cols = [line.strip() for line in f if line.strip()]

    log.info(f"Model ready. Expecting {len(feature_cols)} features.")


#  Routes 

@app.route("/health")
def health():
    """Kubernetes liveness probe — returns 503 until model is loaded."""
    if model is None:
        return jsonify({"status": "loading"}), 503
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts a JSON object of flow features and returns anomaly prediction.

    Example request body:
    {
        "flag_syn": 1,
        "flag_ack": 0,
        "payload_len": 0,
        "src_port": 54321,
        "dst_port": 80
    }

    Returns:
    {
        "prediction": "anomaly" | "normal",
        "label": 1 | 0,
        "probability": 0.97
    }
    """
    if model is None:
        return jsonify({"error": "Model not loaded yet"}), 503

    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "Empty request body"}), 400

    try:
        # Align incoming features to the exact order used during training
        X = np.array(
            [float(data.get(col, 0)) for col in feature_cols]
        ).reshape(1, -1)
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Feature error: {e}"}), 422

    # Apply same pipeline as training: scale > SVD > predict
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
    """Returns info about the currently loaded model."""
    return jsonify({
        "model_name":    os.environ.get("MODEL_NAME",  "network-anomaly-svm"),
        "model_stage":   os.environ.get("MODEL_STAGE", "Production"),
        "feature_count": len(feature_cols) if feature_cols else None,
        "model_type":    type(model).__name__ if model else None,
    })


#  Startup 

if __name__ == "__main__":
    load_model()
    port = int(os.environ.get("PORT", 8089))
    log.info(f"Starting Flask on port {port}")
    app.run(host="0.0.0.0", port=port)
