from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import json
import os
import pickle

os.environ.setdefault("MPLCONFIGDIR", os.path.join(
    "output_ass5", ".mplconfig"))
matplotlib.use("Agg")

try:
    import tensorflow as tf
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "tensorflow is required. Install dependencies with: pip install -r requirements.txt"
    ) from exc


DATA_PATH = "data/test.csv"
OUTPUT_DIR = "output_ass5"
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "ass5_best_model.keras")
ARTIFACTS_PATH = os.path.join(OUTPUT_DIR, "ass5_artifacts.pkl")
THRESHOLD_PATH = os.path.join(OUTPUT_DIR, "ass5_threshold.json")

PREDICTIONS_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "ass5_predictions.csv")
METRICS_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "ass5_evaluation_metrics.txt")
CONFUSION_MATRIX_OUTPUT_PATH = os.path.join(
    OUTPUT_DIR, "ass5_confusion_matrix.jpg")
CONFUSION_MATRIX_NORMALIZED_OUTPUT_PATH = os.path.join(
    OUTPUT_DIR, "ass5_confusion_matrix_normalized.jpg"
)

REPLACEMENTS = {
    "Gender": {"Fe Male": "Female"},
    "Occupation": {"Free Lancer": "Freelancer"},
    "MaritalStatus": {"Unmarried": "Single"},
}


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [col.strip() for col in df.columns]

    object_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in object_cols:
        df[col] = df[col].astype(str).str.strip(
        ).str.replace(r"\s+", " ", regex=True)

    for col, mapping in REPLACEMENTS.items():
        if col in df.columns:
            df[col] = df[col].replace(mapping)

    return df


def split_inference_features(df: pd.DataFrame, target_column: str):
    if target_column in df.columns:
        y_true = df[target_column].copy()
        X = df.drop(columns=[target_column]).copy()
    else:
        y_true = None
        X = df.copy()
    return X, y_true


def encode_target_for_metrics(y_series: pd.Series, positive_label: str):
    y_str = y_series.astype(str).str.strip()
    return (y_str == str(positive_label)).astype(np.int32).values


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float):
    y_pred = (y_prob >= threshold).astype(int)
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


def save_metrics_text(path: str, threshold: float, metrics: dict):
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


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading model and artifacts...")
    model = tf.keras.models.load_model(BEST_MODEL_PATH)
    with open(ARTIFACTS_PATH, "rb") as f:
        artifacts = pickle.load(f)
    with open(THRESHOLD_PATH, "r") as f:
        threshold = float(json.load(f).get("threshold", 0.5))

    target_column = artifacts["target_column"]
    positive_label = artifacts["positive_label"]
    preprocessor = artifacts["preprocessor"]

    print("Loading inference data...")
    df = clean_dataframe(pd.read_csv(DATA_PATH))
    X_raw, y_true_raw = split_inference_features(df, target_column)

    X_processed = preprocessor.transform(X_raw)

    print("Running inference...")
    y_prob = model.predict(X_processed, verbose=0).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    results_df = df.copy()
    results_df["predicted_label"] = y_pred
    results_df["prediction_probability"] = y_prob

    if y_true_raw is not None:
        y_true = encode_target_for_metrics(y_true_raw, positive_label)
        results_df["true_label"] = y_true
        results_df["correct_prediction"] = (
            results_df["true_label"] == results_df["predicted_label"])

        metrics = compute_metrics(y_true, y_prob, threshold)

        save_metrics_text(METRICS_OUTPUT_PATH, threshold, metrics)

        cm = metrics["confusion_matrix"]
        tn, fp, fn, tp = cm.ravel()
        accuracy = metrics["accuracy"]

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=[0, 1],
            yticklabels=[0, 1],
            cbar_kws={"label": "Count"},
            ax=ax,
        )
        plt.title("Confusion Matrix - Ass5 Classification Model",
                  fontsize=16, fontweight="bold")
        plt.ylabel("True Label", fontsize=12)
        plt.xlabel("Predicted Label", fontsize=12)

        legend_text = (
            f"True Negatives: {tn}\n"
            f"False Positives: {fp}\n"
            f"False Negatives: {fn}\n"
            f"True Positives: {tp}\n\n"
            f"Accuracy: {accuracy:.4f}"
        )
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        ax.text(
            1.35, 0.95, legend_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=props,
        )

        plt.tight_layout()
        plt.savefig(CONFUSION_MATRIX_OUTPUT_PATH, dpi=300, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(10, 8))
        row_sums = cm.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            cm_normalized = np.divide(
                cm.astype("float"), row_sums, where=row_sums != 0)
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".2%",
            cmap="Greens",
            xticklabels=[0, 1],
            yticklabels=[0, 1],
            cbar_kws={"label": "Percentage"},
        )
        plt.title("Normalized Confusion Matrix - Ass5 Classification Model",
                  fontsize=16, fontweight="bold")
        plt.ylabel("True Label", fontsize=12)
        plt.xlabel("Predicted Label", fontsize=12)
        plt.tight_layout()
        plt.savefig(CONFUSION_MATRIX_NORMALIZED_OUTPUT_PATH,
                    dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Threshold used: {threshold:.6f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1: {metrics['f1']:.4f}")
        if metrics["auc"] is None:
            print("AUC: Could not calculate")
        else:
            print(f"AUC: {metrics['auc']:.4f}")
    else:
        print(
            "No ground-truth labels in inference file. Skipping evaluation metrics/plots.")

    results_df.to_csv(PREDICTIONS_OUTPUT_PATH, index=False)
    print(f"Predictions saved to: {PREDICTIONS_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
