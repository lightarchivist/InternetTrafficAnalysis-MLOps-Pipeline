"""
train.py

"""

import os
import sys
import time

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

# import functions from main.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from main import (
    parse_info, assign_protocol_group,
    expand_info_column,
)

# config
DATA_DIR        = os.environ.get("DATA_DIR",            "../data/train")
MLFLOW_URI      = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5555")
EXPERIMENT_NAME = os.environ.get("EXPERIMENT_NAME",     "network-anomaly-detection")
MODEL_NAME      = os.environ.get("MODEL_NAME",          "network-anomaly-svm")


#  Load CSVs
def load_data(data_dir: str) -> pd.DataFrame:
    import glob
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    print(f"Loading {len(files)} file(s)...")
    frames = []
    for f in files:
        print(f"  Reading {os.path.basename(f)}...")
        frames.append(pd.read_csv(f, encoding="latin-1"))
    return pd.concat(frames, ignore_index=True)


#  Feature engineering
def prepare_features(df: pd.DataFrame):
    print("Parsing Info column...")
    df = expand_info_column(df)

    # Drop encrypted/L2 protocols
    df = df[~df["protocol_group"].isin(["DROP", "OTHER"])].reset_index(drop=True)

    # Fill flag NaNs for non-TCP rows
    flag_cols_tcp = ["flag_syn","flag_ack","flag_psh","flag_fin",
                     "flag_rst","flag_urg","flag_ecn","flag_cwr","flag_ae"]
    for col in flag_cols_tcp:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # anomaly labelling
    print("Labelling anomalies...")

    # Rule 1: TCP anomalies detected in Info column
    df["outlier_tcp"] = df["Info"].str.contains(
        "Retransmission|Dup ACK|size limited|Zero Window|Out-Of-Order",
        case=False, na=False).astype(int)

    # Rule 2: suspicious flag combinations
    flag_syn = df["flag_syn"] if "flag_syn" in df.columns else pd.Series(0, index=df.index)
    flag_fin = df["flag_fin"] if "flag_fin" in df.columns else pd.Series(0, index=df.index)
    flag_urg = df["flag_urg"] if "flag_urg" in df.columns else pd.Series(0, index=df.index)

    df["outlier_flags"] = (
        (flag_urg == 1) |
        ((flag_syn == 1) & (flag_fin == 1))
    ).astype(int)

    df["is_anomaly"] = ((df["outlier_tcp"] + df["outlier_flags"]) > 0).astype(int)

    #  Build feature matrix
    exclude = ["No.","Time","Source","Destination","Protocol",
               "Length","Info","protocol_group","tcp_anomaly",
               "is_anomaly","outlier_tcp","outlier_flags"]
    feature_cols = [c for c in df.columns
                    if c not in exclude
                    and pd.api.types.is_numeric_dtype(df[c])]

    X = df[feature_cols].fillna(0).values
    y = df["is_anomaly"].values.astype(float)

    print(f"Features:     {len(feature_cols)}")
    print(f"Samples:      {len(X):,}")
    print(f"Anomaly rate: {y.mean()*100:.2f}%")

    if y.sum() == 0:
        raise ValueError("No anomalies found in dataset — check anomaly rules")

    return X, y, feature_cols


#  Main training run
def run():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"Logging to MLflow at {MLFLOW_URI}")

    with mlflow.start_run() as active_run:

        # Load data
        df = load_data(DATA_DIR)
        mlflow.log_metric("raw_rows", len(df))

        # Feature engineering
        X, y, feature_cols = prepare_features(df)
        mlflow.log_metric("sample_count",  len(X))
        mlflow.log_metric("feature_count", len(feature_cols))
        mlflow.log_metric("anomaly_rate",  round(float(y.mean()), 4))

        # Scale
        print("Scaling...")
        scaler_all   = StandardScaler()
        X_all_scaled = scaler_all.fit_transform(X)

        # SVD - 95% variance threshold
        print("Reducing dimensions with SVD...")
        svd_full     = TruncatedSVD(n_components=min(21, len(feature_cols)-1), random_state=42)
        svd_full.fit(X_all_scaled)
        cumvar           = np.cumsum(svd_full.explained_variance_ratio_)
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

        # Train/test split
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_all_svd, y, test_size=0.2, random_state=42, stratify=y)

        # SVM
        print("Training SVM...")
        t0      = time.time()
        svm_all = SVC(kernel="rbf", class_weight="balanced",
                      probability=True, random_state=42)
        svm_all.fit(X_tr, y_tr)
        svm_time   = time.time() - t0
        y_pred_svm = svm_all.predict(X_te)
        y_prob_svm = svm_all.predict_proba(X_te)[:, 1]
        svm_f1     = round(f1_score(y_te, y_pred_svm), 4)

        mlflow.log_metrics({
            "svm_precision": round(precision_score(y_te, y_pred_svm), 4),
            "svm_recall":    round(recall_score(y_te,    y_pred_svm), 4),
            "svm_f1":        svm_f1,
            "svm_roc_auc":   round(roc_auc_score(y_te,   y_prob_svm), 4),
            "svm_train_sec": round(svm_time, 2),
        })
        print(f"SVM F1={svm_f1}")

        # Naive Bayes baseline
        print("Training Naive Bayes...")
        nb_all    = GaussianNB()
        nb_all.fit(X_tr, y_tr)
        y_pred_nb = nb_all.predict(X_te)
        y_prob_nb = nb_all.predict_proba(X_te)[:, 1]
        mlflow.log_metrics({
            "nb_f1":      round(f1_score(y_te,     y_pred_nb), 4),
            "nb_roc_auc": round(roc_auc_score(y_te, y_prob_nb), 4),
        })

        # Save artifacts to MLflow
        print("Saving models to MLflow...")
        mlflow.sklearn.log_model(svm_all,    "svm_model",
                                 registered_model_name=MODEL_NAME)
        mlflow.sklearn.log_model(scaler_all, "scaler")
        mlflow.sklearn.log_model(svd_all,    "svd")
        mlflow.log_text("\n".join(feature_cols), "feature_cols.txt")

        # Promote to Production if F1 >= 0.90
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
            print(f"Model NOT promoted - F1={svm_f1} below 0.90 threshold")

        print(f"\nDone. Run ID: {active_run.info.run_id}")
        print(f"View results at {MLFLOW_URI}")


if __name__ == "__main__":
    run()