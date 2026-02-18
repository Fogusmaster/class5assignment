from tensorflow.keras import callbacks, layers, models, regularizers
from sklearn.utils import class_weight
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
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


RANDOM_SEED = 42
TARGET_COLUMN = "ProdTaken"
VALIDATION_SIZE = 0.2
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

TRAIN_DATA_PATH = "data/train.csv"
OUTPUT_DIR = "output_ass5"
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "ass5_best_model.keras")
ARTIFACTS_PATH = os.path.join(OUTPUT_DIR, "ass5_artifacts.pkl")
THRESHOLD_PATH = os.path.join(OUTPUT_DIR, "ass5_threshold.json")
METADATA_PATH = os.path.join(OUTPUT_DIR, "ass5_model_metadata.json")
VALIDATION_METRICS_PATH = os.path.join(
    OUTPUT_DIR, "ass5_validation_metrics.txt")
TRAINING_HISTORY_PLOT_PATH = os.path.join(
    OUTPUT_DIR, "ass5_training_history.png")


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


def encode_binary_target(y_series: pd.Series):
    y_str = y_series.astype(str).str.strip()
    unique_values = sorted(y_str.unique().tolist())
    if len(unique_values) != 2:
        raise ValueError(
            f"Target must be binary, found values: {unique_values}")

    lowered = [v.lower() for v in unique_values]
    if "yes" in lowered:
        positive_label = unique_values[lowered.index("yes")]
    elif "1" in lowered:
        positive_label = unique_values[lowered.index("1")]
    else:
        positive_label = unique_values[1]

    y_binary = (y_str == positive_label).astype(np.int32)
    return y_binary, unique_values, positive_label


def get_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = X.select_dtypes(
        include=["object", "category"]).columns.tolist()
    numerical_cols = X.select_dtypes(
        include=["number", "bool"]).columns.tolist()

    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", num_pipeline, numerical_cols),
            ("cat", cat_pipeline, categorical_cols),
        ]
    )
    return preprocessor


def build_model(input_dim: int) -> tf.keras.Model:
    model = models.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(512, activation="relu",
                         kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(256, activation="relu",
                         kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(128, activation="relu",
                         kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(64, activation="relu",
                         kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(32, activation="relu",
                         kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float):
    y_pred = (y_prob >= threshold).astype(int)
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


def save_metrics_text(path: str, title: str, threshold: float, metrics: dict):
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
        lines.append(" ".join(str(v) for v in row))
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def save_training_history_plot(history, path: str):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Acc")
    plt.plot(history.history["val_accuracy"], label="Val Acc")
    plt.legend()
    plt.title("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.legend()
    plt.title("Loss")

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading and cleaning training data...")
    df = clean_dataframe(pd.read_csv(TRAIN_DATA_PATH))

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found in training data")

    df = df.dropna(subset=[TARGET_COLUMN]).copy()

    X = df.drop(columns=[TARGET_COLUMN])
    y, target_labels, positive_label = encode_binary_target(df[TARGET_COLUMN])

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    print("Fitting preprocessor...")
    preprocessor = get_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)

    class_weights_vals = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train,
    )
    class_weights_dict = {
        int(cls): float(weight)
        for cls, weight in zip(np.unique(y_train), class_weights_vals)
    }
    print(f"Class weights: {class_weights_dict}")

    model = build_model(X_train_processed.shape[1])

    callbacks_list = [
        callbacks.ModelCheckpoint(
            BEST_MODEL_PATH,
            monitor="val_auc",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            verbose=1,
        ),
    ]

    print("Training model...")
    history = model.fit(
        X_train_processed,
        y_train,
        validation_data=(X_val_processed, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks_list,
        class_weight=class_weights_dict,
        verbose=1,
    )

    threshold = 0.5
    y_val_prob = model.predict(X_val_processed, verbose=0).ravel()
    val_metrics = compute_metrics(y_val, y_val_prob, threshold)

    artifacts = {
        "preprocessor": preprocessor,
        "target_column": TARGET_COLUMN,
        "target_labels": target_labels,
        "positive_label": positive_label,
        "positive_index": 1,
    }
    with open(ARTIFACTS_PATH, "wb") as f:
        pickle.dump(artifacts, f)

    with open(THRESHOLD_PATH, "w") as f:
        json.dump({"threshold": threshold,
                  "source": "fixed_default"}, f, indent=2)

    save_metrics_text(
        VALIDATION_METRICS_PATH,
        "ASS5 VALIDATION METRICS",
        threshold,
        val_metrics,
    )
    save_training_history_plot(history, TRAINING_HISTORY_PLOT_PATH)

    metadata = {
        "random_seed": RANDOM_SEED,
        "train_data_path": TRAIN_DATA_PATH,
        "target_column": TARGET_COLUMN,
        "target_labels": target_labels,
        "positive_label": positive_label,
        "validation_size": VALIDATION_SIZE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "architecture": [512, 256, 128, 64, 32],
        "loss": "binary_crossentropy",
        "optimizer": "Adam",
        "metrics": ["accuracy", "auc"],
        "threshold": threshold,
        "validation_metrics": val_metrics,
        "paths": {
            "best_model": BEST_MODEL_PATH,
            "artifacts": ARTIFACTS_PATH,
            "threshold": THRESHOLD_PATH,
            "validation_metrics": VALIDATION_METRICS_PATH,
            "training_history": TRAINING_HISTORY_PLOT_PATH,
        },
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print("Training complete.")
    print(f"Model saved to: {BEST_MODEL_PATH}")
    print(f"Artifacts saved to: {ARTIFACTS_PATH}")
    print(f"Threshold saved to: {THRESHOLD_PATH}")
    print(f"Validation metrics saved to: {VALIDATION_METRICS_PATH}")
    print(f"Training history plot saved to: {TRAINING_HISTORY_PLOT_PATH}")


if __name__ == "__main__":
    main()
