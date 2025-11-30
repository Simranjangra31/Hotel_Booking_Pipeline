# train.py
"""
Training script for hotel booking cancellation prediction.
Implements data loading, cleaning, feature engineering, handling class imbalance,
model training (Logistic Regression, Random Forest, XGBoost) and hyper‑parameter tuning.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from data_loader import load_data
from preprocess import clean_data, get_preprocessor
from feature_engineering import add_features

def train_models(csv_path: str, model_output_path: str = "model_pipeline.joblib"):
    # ------------------------------------------------------------------
    # 1. Load & clean data
    # ------------------------------------------------------------------
    df_raw = load_data(csv_path)
    df_clean = clean_data(df_raw)
    df_feat = add_features(df_clean)

    # ------------------------------------------------------------------
    # 2. Split data (stratified)
    # ------------------------------------------------------------------
    X = df_feat.drop(columns=["target"])
    y = df_feat["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ------------------------------------------------------------------
    # 3. Handle class imbalance with SMOTE (only on training set)
    # ------------------------------------------------------------------
    # Note: SMOTE should ideally be part of a pipeline to avoid leakage, 
    # but imblearn Pipeline is needed. For simplicity in this script, we apply it before sklearn Pipeline.
    # Or we can use class_weight='balanced' for models. 
    # Let's use SMOTE on the training set explicitly here.
    
    # Preprocess X_train first because SMOTE needs numeric data (after encoding)
    # This is tricky with ColumnTransformer inside the pipeline.
    # EASIER APPROACH: Use class_weight='balanced' for models where possible, 
    # or use imblearn Pipeline. 
    # Given the requirements, let's use class_weight='balanced' for simplicity and robustness 
    # with the standard sklearn Pipeline, as SMOTE with mixed types requires careful handling.
    # However, the user asked for SMOTE or class weights. I will use class_weight='balanced' 
    # for LogReg and RF, and scale_pos_weight for XGBoost.
    
    # If we strictly want SMOTE, we need to encode first. 
    # Let's stick to class weights as it's cleaner for this pipeline structure.
    
    # ------------------------------------------------------------------
    # 4. Build preprocessing transformer
    # ------------------------------------------------------------------
    preprocessor = get_preprocessor(df_feat)

    # ------------------------------------------------------------------
    # 5. Define models
    # ------------------------------------------------------------------
    # Calculate scale_pos_weight for XGBoost
    ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)

    models = {
        "logreg": LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1),
        "rf": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "xgb": XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            scale_pos_weight=ratio,
            random_state=42,
            n_jobs=-1,
        ),
    }

    # ------------------------------------------------------------------
    # 6. Train each model & collect metrics
    # ------------------------------------------------------------------
    results = {}
    # Ensure numpy is imported (removed local import)
    
    for name, estimator in models.items():
        pipe = Pipeline([("preprocess", preprocessor), ("model", estimator)])
        print(f"Training {name}...")
        pipe.fit(X_train, y_train)

        # Predict on held‑out test set
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }
        results[name] = {"pipeline": pipe, "metrics": metrics}
        print(f"\n=== {name.upper()} METRICS ===")
        for k, v in metrics.items():
            if k != "confusion_matrix":
                print(f"{k}: {v:.4f}")

    # ------------------------------------------------------------------
    # 7. Hyper‑parameter tuning on the best model (XGBoost in this example)
    # ------------------------------------------------------------------
    print("\nStarting Hyperparameter Tuning for XGBoost...")
    param_grid = {
        "model__learning_rate": [0.01, 0.1],
        "model__max_depth": [3, 5],
        "model__n_estimators": [100, 200],
    }
    
    xgb_pipe = Pipeline([("preprocess", preprocessor), ("model", XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        scale_pos_weight=ratio,
        random_state=42,
        n_jobs=-1,
    ))])

    grid = GridSearchCV(
        xgb_pipe,
        param_grid,
        cv=3, # Reduced cv for speed in this example
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)

    best_pipe = grid.best_estimator_
    y_pred_best = best_pipe.predict(X_test)
    y_proba_best = best_pipe.predict_proba(X_test)[:, 1]

    best_metrics = {
        "accuracy": accuracy_score(y_test, y_pred_best),
        "precision": precision_score(y_test, y_pred_best, zero_division=0),
        "recall": recall_score(y_test, y_pred_best, zero_division=0),
        "f1": f1_score(y_test, y_pred_best, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba_best),
        "confusion_matrix": confusion_matrix(y_test, y_pred_best).tolist(),
    }

    print("\n=== XGBOOST (TUNED) METRICS ===")
    for k, v in best_metrics.items():
        if k != "confusion_matrix":
            print(f"{k}: {v:.4f}")

    # ------------------------------------------------------------------
    # 8. Save the tuned pipeline
    # ------------------------------------------------------------------
    joblib.dump(best_pipe, model_output_path)
    print(f"\nSaved tuned model pipeline to {model_output_path}")

    return {
        "baseline_results": results,
        "tuned_model_path": model_output_path,
        "tuned_metrics": best_metrics,
    }

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train hotel cancellation model")
    parser.add_argument(
        "--csv",
        type=str,
        default="Hotel Reservations.csv",
        help="Path to the hotel reservations CSV file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="model_pipeline.joblib",
        help="File name for the saved pipeline",
    )
    args = parser.parse_args()
    train_models(args.csv, args.output)
