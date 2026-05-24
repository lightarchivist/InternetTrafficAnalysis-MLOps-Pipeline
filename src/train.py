"""
train.py
--------
Wraps main.py with MLflow tracking.
Imports your existing functions directly from main.py.

Usage:
    python train.py
    
MLflow must be running first:
    cd ../mlflow && docker compose up -d
"""

import os
import sys
import time
import importlib.util

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

# import existing code from main.py 
# load main.py as a module so I can reuse your functions directly
spec   = importlib.util.spec_from_file_location("main", 
         os.path.join(os.path.dirname(__file__), "..", "main.py"))
main   = importlib.util.module_from_spec(spec)


from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (f1_score, precision_score,
                              recall_score, roc_auc_score)

# parsing functions directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from main import (
    parse_tcp, parse_udp, parse_icmp,
    parse_info, assign_protocol_group,
    expand_info_column, flag_combo_label,
    TCP_LIKE, UDP_LIKE, DROP_PROTOCOLS,
)

# Config 
DATA_DIR         = os.environ.get("DATA_DIR",         "../data")
MLFLOW_URI       = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5555")
EXPERIMENT_NAME  = os.environ.get("EXPERIMENT_NAME",  "network-anomaly-detection")
MODEL_NAME       = os.environ.get("MODEL_NAME",        "network-anomaly-svm")


#  load and clean CSVs
def load_data(data_dir: str) -> pd.DataFrame:
    import glob
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    print(f"Loading {len(files)} file(s)...")
    frames = []
    for f in files:
        print(f"  Reading {os.path.basename(f)}...")
        df = pd.read_csv(f, encoding="latin-1")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def prepare_features(df: pd.DataFrame):
    """
    Applies your notebook pipeline:
    expand_info_column → label anomalies → build feature matrix
    """
    print("Parsing Info column...")
    df = expand_info_column(df)

    # Drop encrypted protocols
    df = df[~df["protocol_group"].isin(["DROP", "OTHER"])].reset_index(drop=True)

    # Fill flag NaNs for non-TCP rows
    flag_cols = ["flag_syn","flag_ack","flag_psh","flag_fin",
                 "flag_rst","flag_urg","flag_ecn","flag_cwr","flag_ae"]
    for col in flag_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Apply your anomaly labelling function
    print("Labelling anomalies...")
    df["is_anomaly"] = df.apply(flag_combo_label, axis=1)

    # Build feature matrix — same columns your notebook uses
    exclude = ["No.", "Time", "Source", "Destination", "Protocol",
               "Length", "Info", "protocol_group", "tcp_anomaly",
               "is_anomaly"]
    feature_cols = [c for c in df.columns
                    if c not in exclude
                    and pd.api.types.is_numeric_dtype(df[c])]

    X = df[feature_cols].fillna(0).values
    y = df["is_anomaly"].values

    print(f"Features: {len(feature_cols)}")
    print(f"Samples:  {len(X):,}")
    print(f"Anomaly rate: {y.mean()*100:.2f}%")

    return X, y, feature_cols


#  training  
def run():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"Logging to MLflow at {MLFLOW_URI}")

    with mlflow.start_run() as run:

        # load data 
        df = load_data(DATA_DIR)
        mlflow.log_metric("raw_rows", len(df))

        #  Feature engineering 
        X, y, feature_cols = prepare_features(df)
        mlflow.log_metric("sample_count", len(X))
        mlflow.log_metric("feature_count", len(feature_cols))
        mlflow.log_metric("anomaly_rate", round(float(y.mean()), 4))

        #  Scale 
        print("Scaling...")
        scaler_all   = StandardScaler()
        X_all_scaled = scaler_all.fit_transform(X)

        #  SVD (same as your notebook — 95% variance) 
        print("Reducing dimensions with SVD...")
        svd_full = TruncatedSVD(n_components=min(21, len(feature_cols)-1), random_state=42)
        svd_full.fit(X_all_scaled)
        cumvar = np.cumsum(svd_full.explained_variance_ratio_)
        n_components_all = int(np.argmax(cumvar >= 0.95)) + 1
        print(f"SVD components for 95% variance: {n_components_all}")

        svd_all   = TruncatedSVD(n_components=n_components_all, random_state=42)
        X_all_svd = svd_all.fit_transform(X_all_scaled)

        mlflow.log_params({
            "svd_components":       n_components_all,
            "svd_variance_covered": round(float(cumvar[n_components_all-1]), 4),
            "svm_kernel":           "rbf",
            "svm_class_weight":     "balanced",
            "test_size":            0.2,
        })

        #  Train/test split 
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_all_svd, y, test_size=0.2, random_state=42, stratify=y)

        #  SVM (your primary model) 
        print("Training SVM...")
        t0      = time.time()
        svm_all = SVC(kernel="rbf", class_weight="balanced",
                      probability=True, random_state=42)
        svm_all.fit(X_tr, y_tr)
        svm_time = time.time() - t0

        y_pred_svm = svm_all.predict(X_te)
        y_prob_svm = svm_all.predict_proba(X_te)[:, 1]
        svm_f1     = round(f1_score(y_te, y_pred_svm), 4)

        mlflow.log_metrics({
            "svm_precision":  round(precision_score(y_te, y_pred_svm), 4),
            "svm_recall":     round(recall_score(y_te,    y_pred_svm), 4),
            "svm_f1":         svm_f1,
            "svm_roc_auc":    round(roc_auc_score(y_te,   y_prob_svm), 4),
            "svm_train_sec":  round(svm_time, 2),
        })
        print(f"SVM F1={svm_f1}")

        #  Naive Bayes (baseline) 
        print("Training Naive Bayes...")
        nb_all = GaussianNB()
        nb_all.fit(X_tr, y_tr)
        y_pred_nb = nb_all.predict(X_te)
        y_prob_nb = nb_all.predict_proba(X_te)[:, 1]

        mlflow.log_metrics({
            "nb_f1":      round(f1_score(y_te,    y_pred_nb), 4),
            "nb_roc_auc": round(roc_auc_score(y_te, y_prob_nb), 4),
        })

        #  Log model artifacts to MLflow 
        print("Saving models to MLflow...")
        mlflow.sklearn.log_model(svm_all,    "svm_model",
                                 registered_model_name=MODEL_NAME)
        mlflow.sklearn.log_model(scaler_all, "scaler")
        mlflow.sklearn.log_model(svd_all,    "svd")

        # Save feature column order — app.py needs this to align inputs
        mlflow.log_text("\n".join(feature_cols), "feature_cols.txt")

        #  Promote to Production if F1 >= 0.90 
        if svm_f1 >= 0.90:
            client   = mlflow.tracking.MlflowClient()
            versions = client.get_latest_versions(MODEL_NAME, stages=["None"])
            if versions:
                client.transition_model_version_stage(
                    name=MODEL_NAME,
                    version=versions[0].version,
                    stage="Production",
                    archive_existing_versions=True,
                )
                print(f"Model promoted to Production (F1={svm_f1})")
        else:
            print(f"Model NOT promoted — F1={svm_f1} below 0.90 threshold")

        print(f"\nDone. Run ID: {run.info.run_id}")
        print(f"View results at {MLFLOW_URI}")


if __name__ == "__main__":
    run()