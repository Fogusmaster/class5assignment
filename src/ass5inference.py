import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


DATA_PATH = "data/ass5test.csv"
TARGET_COLUMN = "ProdTaken"
OUTPUT_DIR = "output_ass5"
MODEL_PATH = os.path.join(OUTPUT_DIR, "ass5_model.pkl")
THRESHOLD_PATH = os.path.join(OUTPUT_DIR, "ass5_threshold.json")
INFERENCE_OUTPUT_DIR = "output_ass5"
PREDICTIONS_OUTPUT_PATH = os.path.join(INFERENCE_OUTPUT_DIR, "ass5_predictions.csv")
METRICS_OUTPUT_PATH = os.path.join(INFERENCE_OUTPUT_DIR, "ass5_evaluation_metrics.txt")
CONFUSION_MATRIX_OUTPUT_PATH = os.path.join(INFERENCE_OUTPUT_DIR, "ass5_confusion_matrix.jpg")
CONFUSION_MATRIX_NORMALIZED_OUTPUT_PATH = os.path.join(
    INFERENCE_OUTPUT_DIR, "ass5_confusion_matrix_normalized.jpg"
)


def clean_dataframe(df):
    df = df.copy()
    df.columns = [col.strip() for col in df.columns]
    object_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in object_cols:
        df[col] = df[col].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)

    replacements = {
        "Gender": {"Fe Male": "Female"},
        "Occupation": {"Free Lancer": "Freelancer"},
        "MaritalStatus": {"Unmarried": "Single"},
    }
    for col, mapping in replacements.items():
        if col in df.columns:
            df[col] = df[col].replace(mapping)
    return df


def fill_missing_values(df, target_column):
    df = df.copy()
    if target_column in df.columns:
        df = df.dropna(subset=[target_column])
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    for col in numeric_cols:
        if col == target_column:
            continue
        df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if col == target_column:
            continue
        df[col] = df[col].fillna(df[col].mode().iloc[0])

    return df


def split_inference_features(df, target_column):
    if target_column in df.columns:
        y_true = df[target_column].astype(int)
        X = df.drop(columns=[target_column]).copy()
    else:
        y_true = None
        X = df.copy()

    if target_column in X.columns:
        raise ValueError(f"Leakage guard failed: target column '{target_column}' is present in inference features.")
    return X, y_true


def compute_metrics(y_true, y_pred, y_prob):
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }
    try:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["auc"] = None
    return metrics


def save_metrics_text(path, threshold, metrics):
    with open(path, "w") as metrics_file:
        metrics_file.write("ASS5 INFERENCE EVALUATION METRICS\n")
        metrics_file.write("=" * 40 + "\n\n")
        metrics_file.write("Threshold source: saved training artifact\n")
        metrics_file.write(f"Threshold used: {threshold:.6f}\n")
        metrics_file.write(f"Accuracy: {metrics['accuracy']:.6f}\n")
        if metrics["auc"] is None:
            metrics_file.write("AUC: Could not calculate\n")
        else:
            metrics_file.write(f"AUC: {metrics['auc']:.6f}\n")
        metrics_file.write(f"Precision: {metrics['precision']:.6f}\n")
        metrics_file.write(f"Recall: {metrics['recall']:.6f}\n")
        metrics_file.write(f"F1: {metrics['f1']:.6f}\n\n")
        metrics_file.write("Confusion Matrix:\n")
        for row in metrics["confusion_matrix"]:
            metrics_file.write(" ".join(str(value) for value in row) + "\n")


os.makedirs(INFERENCE_OUTPUT_DIR, exist_ok=True)

print("Loading trained model and threshold artifacts...")
with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)
with open(THRESHOLD_PATH, "r") as threshold_file:
    threshold = float(json.load(threshold_file).get("threshold", 0.5))
print("Model and threshold loaded successfully.")

print("\nLoading inference data...")
df = clean_dataframe(pd.read_csv(DATA_PATH))
df = fill_missing_values(df, TARGET_COLUMN)
print(f"Inference dataset shape: {df.shape}")

X, y_true = split_inference_features(df, TARGET_COLUMN)
print("\nMaking predictions with saved threshold (no tuning)...")
y_pred_proba = model.predict_proba(X)[:, 1]
y_pred = (y_pred_proba >= threshold).astype(int)

results_df = df.copy()
results_df["predicted_label"] = y_pred
results_df["prediction_probability"] = y_pred_proba

if y_true is not None:
    y_true_values = y_true.values
    results_df["true_label"] = y_true_values
    results_df["correct_prediction"] = results_df["true_label"] == results_df["predicted_label"]

print("\n" + "=" * 60)
print("INFERENCE RESULTS")
print("=" * 60)

if y_true is not None:
    metrics = compute_metrics(y_true_values, y_pred, y_pred_proba)
    print(f"\nThreshold used: {threshold:.6f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    if metrics["auc"] is None:
        print("AUC Score: Could not calculate")
    else:
        print(f"AUC Score: {metrics['auc']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print("\nConfusion Matrix:")
    print(metrics["confusion_matrix"])

    save_metrics_text(METRICS_OUTPUT_PATH, threshold, metrics)
    print(f"Evaluation metrics saved to: {METRICS_OUTPUT_PATH}")

    cm = metrics["confusion_matrix"]
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[0, 1],
        yticklabels=[0, 1],
        cbar_kws={"label": "Count"},
    )
    plt.title("Confusion Matrix - Ass5 Classification Model", fontsize=16, fontweight="bold")
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)

    tn, fp, fn, tp = cm.ravel()
    stats_text = f"True Negatives: {tn}\nFalse Positives: {fp}\nFalse Negatives: {fn}\nTrue Positives: {tp}"
    stats_text += f"\n\nAccuracy: {metrics['accuracy']:.4f}"
    plt.text(
        2.5,
        0.5,
        stats_text,
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        verticalalignment="center",
    )

    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"\nConfusion matrix plot saved to: {CONFUSION_MATRIX_OUTPUT_PATH}")

    plt.figure(figsize=(10, 8))
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2%",
        cmap="Greens",
        xticklabels=[0, 1],
        yticklabels=[0, 1],
        cbar_kws={"label": "Percentage"},
    )
    plt.title("Normalized Confusion Matrix - Ass5 Classification Model", fontsize=16, fontweight="bold")
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)

    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_NORMALIZED_OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Normalized confusion matrix plot saved to: {CONFUSION_MATRIX_NORMALIZED_OUTPUT_PATH}")
else:
    print("\nNo ground-truth labels found. Skipping metrics and plots.")

results_df.to_csv(PREDICTIONS_OUTPUT_PATH, index=False)
print(f"Predictions saved to: {PREDICTIONS_OUTPUT_PATH}")

print("\n" + "=" * 60)
print("Inference completed successfully!")
print("=" * 60)
