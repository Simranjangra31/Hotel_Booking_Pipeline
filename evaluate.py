# evaluate.py
"""
Evaluation script that loads the saved pipeline, makes predictions on a test set,
computes performance metrics and visualises ROC / Precision‑Recall curves.
"""

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

from data_loader import load_data
from preprocess import clean_data, get_preprocessor
from feature_engineering import add_features

def evaluate_model(csv_path: str, model_path: str = "model_pipeline.joblib"):
    # Load data and apply same preprocessing/feature steps
    df_raw = load_data(csv_path)
    df_clean = clean_data(df_raw)
    df_feat = add_features(df_clean)

    X = df_feat.drop(columns=["target"])
    y_true = df_feat["target"]

    # Load the trained pipeline
    try:
        pipeline = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Model file not found at {model_path}. Please run training first.")
        return

    # Predictions
    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)[:, 1]

    # Metrics
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    print("\n=== EVALUATION METRICS ===")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"{k}: {v:.4f}")

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"ROC AUC = {metrics['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_curve.png")
    plt.close()

    # Plot Precision‑Recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label="Precision‑Recall")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision‑Recall Curve")
    plt.tight_layout()
    plt.savefig("pr_curve.png")
    plt.close()

    # Feature importance (if model has it)
    if hasattr(pipeline.named_steps["model"], "feature_importances_"):
        import numpy as np

        # Get feature names after one‑hot encoding
        preproc = pipeline.named_steps["preprocess"]
        # sklearn >=1.0 provides get_feature_names_out
        try:
            feature_names = preproc.get_feature_names_out()
        except AttributeError:
            # fallback for older versions
            feature_names = np.array(preproc.transformers_[0][2] + preproc.transformers_[1][2])

        importances = pipeline.named_steps["model"].feature_importances_
        
        # Ensure lengths match
        if len(feature_names) == len(importances):
            importance_df = pd.DataFrame(
                {"feature": feature_names, "importance": importances}
            ).sort_values(by="importance", ascending=False)

            top5 = importance_df.head(5)
            print("\nTop 5 important features:")
            print(top5.to_string(index=False))

            # Save to CSV for later reference
            top5.to_csv("top5_features.csv", index=False)
        else:
            print(f"Feature names count ({len(feature_names)}) does not match importance count ({len(importances)}). Skipping feature importance table.")

    print("\nPlots saved: roc_curve.png, pr_curve.png")
    print("Top-5 feature importances saved to top5_features.csv (if applicable).")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate saved hotel cancellation model")
    parser.add_argument(
        "--csv",
        type=str,
        default="Hotel Reservations.csv",
        help="Path to the CSV file used for evaluation",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="model_pipeline.joblib",
        help="Path to the saved model pipeline",
    )
    args = parser.parse_args()
    evaluate_model(args.csv, args.model)
