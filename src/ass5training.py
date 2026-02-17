import json
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


VALIDATION_SIZE = 0.2
TARGET_COLUMN = "ProdTaken"

TRAIN_DATA_PATH = "data/ass5train.csv"

OUTPUT_DIR = "output_ass5"
MODEL_PATH = os.path.join(OUTPUT_DIR, "ass5_model.pkl")
THRESHOLD_PATH = os.path.join(OUTPUT_DIR, "ass5_threshold.json")
METADATA_PATH = os.path.join(OUTPUT_DIR, "ass5_model_metadata.json")
VALIDATION_METRICS_PATH = os.path.join(
    OUTPUT_DIR, "ass5_validation_metrics.txt")


def clean_dataframe(df):
    df = df.copy()
    df.columns = [col.strip() for col in df.columns]
    object_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in object_cols:
        df[col] = df[col].astype(str).str.strip(
        ).str.replace(r"\s+", " ", regex=True)

    replacements = {
        "Gender": {"Fe Male": "Female"},
        "Occupation": {"Free Lancer": "Freelancer"},
        "MaritalStatus": {"Unmarried": "Single"},
    }
    for col, mapping in replacements.items():
        if col in df.columns:
            df[col] = df[col].replace(mapping)
    return df


def split_features_target(df, target_column):
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found.")

    X = df.drop(columns=[target_column]).copy()
    if target_column in X.columns:
        raise ValueError(
            "Leakage guard failed: target column is present in feature matrix.")
    y = df[target_column].astype(int).copy()
    return X, y


def build_preprocessor(X):
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])

    return ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ])


def find_best_threshold(y_true, y_prob):
    thresholds = np.linspace(0.05, 0.95, 181)
    best_threshold = 0.5
    best_accuracy = -1.0
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        acc = accuracy_score(y_true, y_pred)
        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = float(threshold)
    return best_threshold, best_accuracy


def compute_metrics(y_true, y_pred, y_prob):
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    try:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["auc"] = None
    return metrics


def format_metrics_text(title, threshold, metrics):
    auc_text = "Could not calculate" if metrics[
        "auc"] is None else f"{metrics['auc']:.6f}"
    lines = [
        title,
        "=" * len(title),
        "",
        f"Threshold: {threshold:.6f}",
        f"Accuracy: {metrics['accuracy']:.6f}",
        f"AUC: {auc_text}",
        f"Precision: {metrics['precision']:.6f}",
        f"Recall: {metrics['recall']:.6f}",
        f"F1: {metrics['f1']:.6f}",
        "",
        "Confusion Matrix:",
    ]
    for row in metrics["confusion_matrix"]:
        lines.append(" ".join(str(value) for value in row))
    lines.append("")
    return "\n".join(lines)


def save_metrics_text(path, title, threshold, metrics):
    with open(path, "w") as metrics_file:
        metrics_file.write(format_metrics_text(title, threshold, metrics))


def main():
    print("Loading training data...")
    train_df = clean_dataframe(pd.read_csv(TRAIN_DATA_PATH))
    if TARGET_COLUMN not in train_df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found in training data.")
    train_df = train_df.dropna(subset=[TARGET_COLUMN]).copy()
    train_df[TARGET_COLUMN] = train_df[TARGET_COLUMN].astype(int)
    print(f"Training dataset shape: {train_df.shape}")
    print(
        f"\nTraining target distribution:\n{train_df[TARGET_COLUMN].value_counts().sort_index()}")

    X_full_train, y_full_train = split_features_target(train_df, TARGET_COLUMN)
    X_train, X_val, y_train, y_val = train_test_split(
        X_full_train,
        y_full_train,
        test_size=VALIDATION_SIZE,
        stratify=y_full_train,
    )

    candidate_models = {
        "decision_tree": DecisionTreeClassifier(
            class_weight="balanced",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=800,
            n_jobs=1,
            class_weight="balanced_subsample",
            max_features="sqrt",
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=800,
            n_jobs=1,
            class_weight="balanced_subsample",
            max_features="sqrt",
        ),
    }

    print("\nRunning model selection on training split only...")
    model_results = {}

    for model_name, estimator in candidate_models.items():
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor(X_train)),
            ("model", clone(estimator)),
        ])
        pipeline.fit(X_train, y_train)

        val_prob = pipeline.predict_proba(X_val)[:, 1]
        threshold, _ = find_best_threshold(y_val, val_prob)
        val_pred = (val_prob >= threshold).astype(int)
        val_metrics = compute_metrics(y_val, val_pred, val_prob)

        model_results[model_name] = {
            "threshold": float(threshold),
            "validation_metrics": val_metrics,
        }
        auc_text = "N/A" if val_metrics["auc"] is None else f"{val_metrics['auc']:.4f}"
        print(
            f"{model_name}: "
            f"val_accuracy={val_metrics['accuracy']:.4f}, "
            f"val_auc={auc_text}, "
            f"threshold={threshold:.3f}"
        )

    best_model_name = max(
        model_results,
        key=lambda name: (
            model_results[name]["validation_metrics"]["accuracy"],
            -1.0 if model_results[name]["validation_metrics"]["auc"] is None else model_results[name]["validation_metrics"]["auc"],
        ),
    )
    best_threshold = float(model_results[best_model_name]["threshold"])
    validation_metrics = model_results[best_model_name]["validation_metrics"]
    print(f"\nSelected model: {best_model_name}")
    print(f"Validation threshold source: train/validation split only")

    final_pipeline = Pipeline([
        ("preprocessor", build_preprocessor(X_full_train)),
        ("model", clone(candidate_models[best_model_name])),
    ])
    final_pipeline.fit(X_full_train, y_full_train)
    print("Final model fitted on full training data only.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(MODEL_PATH, "wb") as model_file:
        pickle.dump(final_pipeline, model_file)
    with open(THRESHOLD_PATH, "w") as threshold_file:
        json.dump({"threshold": best_threshold,
                  "source": "validation_split_train_only"}, threshold_file, indent=2)

    save_metrics_text(
        VALIDATION_METRICS_PATH,
        "ASS5 VALIDATION METRICS (TUNING DATA ONLY)",
        best_threshold,
        validation_metrics,
    )

    metadata = {
        "random_seed": None,
        "target_column": TARGET_COLUMN,
        "split_strategy": "single_train_validation_split",
        "validation_size": VALIDATION_SIZE,
        "threshold_source": "validation_split_train_only",
        "selected_model": best_model_name,
        "candidate_models": list(candidate_models.keys()),
        "train_rows_total": int(train_df.shape[0]),
        "train_rows_for_split_train": int(X_train.shape[0]),
        "train_rows_for_split_validation": int(X_val.shape[0]),
        "train_rows_used_for_final_fit": int(X_full_train.shape[0]),
        "test_data_used_for_model_selection": False,
        "test_data_used_for_threshold_tuning": False,
        "test_data_used_for_final_fit": False,
        "best_threshold": best_threshold,
        "validation_metrics": validation_metrics,
        "paths": {
            "model": MODEL_PATH,
            "threshold": THRESHOLD_PATH,
            "metadata": METADATA_PATH,
            "validation_metrics": VALIDATION_METRICS_PATH,
        },
    }
    with open(METADATA_PATH, "w") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Threshold saved to {THRESHOLD_PATH}")
    print(f"Metadata saved to {METADATA_PATH}")
    print(f"Validation metrics saved to {VALIDATION_METRICS_PATH}")


if __name__ == "__main__":
    main()
