import json
import os
import pickle

import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


DATA_PATH = "data/ass5data.csv"
OUTPUT_DIR = "ass5examples"
MODEL_PATH = os.path.join(OUTPUT_DIR, "ass5_model.h5")
SCALER_PATH = os.path.join(OUTPUT_DIR, "ass5_scaler.pkl")
ENCODER_PATH = os.path.join(OUTPUT_DIR, "ass5_label_encoder.pkl")
COLUMNS_PATH = os.path.join(OUTPUT_DIR, "ass5_feature_columns.json")
INFERENCE_OUTPUT_DIR = "data/output"


def clean_dataframe(df):
	df = df.copy()
	df.columns = [col.strip() for col in df.columns]
	object_cols = df.select_dtypes(include=["object"]).columns.tolist()
	for col in object_cols:
		df[col] = df[col].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
	if "Gender" in df.columns:
		df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
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
		median_value = df[col].median()
		df[col] = df[col].fillna(median_value)

	for col in categorical_cols:
		if col == target_column:
			continue
		mode_value = df[col].mode().iloc[0]
		df[col] = df[col].fillna(mode_value)

	return df


os.makedirs(INFERENCE_OUTPUT_DIR, exist_ok=True)

print("Loading trained model and preprocessing artifacts...")
model = keras.models.load_model(MODEL_PATH)
with open(SCALER_PATH, "rb") as scaler_file:
	scaler = pickle.load(scaler_file)
with open(ENCODER_PATH, "rb") as encoder_file:
	label_encoder = pickle.load(encoder_file)
with open(COLUMNS_PATH, "r") as columns_file:
	feature_columns = json.load(columns_file)

print("Model and preprocessing artifacts loaded successfully!")
model.summary()

print("\nLoading inference data...")
df = pd.read_csv(DATA_PATH)
df = clean_dataframe(df)

target_column = "ProdTaken"
df = fill_missing_values(df, target_column)

print(f"Inference dataset shape: {df.shape}")

if target_column in df.columns:
	y_true = df[target_column].astype(int)
	X = df.drop(target_column, axis=1)
else:
	y_true = None
	X = df

categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()
print(f"\nCategorical columns: {categorical_columns}")

X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
X_encoded = X_encoded.reindex(columns=feature_columns, fill_value=0)

print(f"Features after encoding: {X_encoded.shape[1]}")

X_scaled = scaler.transform(X_encoded)

print("\nMaking predictions...")
y_pred_proba = model.predict(X_scaled, verbose=0)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

results_df = df.copy()
results_df["predicted_label"] = label_encoder.inverse_transform(y_pred)
results_df["prediction_probability"] = y_pred_proba.flatten()

if y_true is not None:
	y_true_encoded = label_encoder.transform(y_true)
	results_df["true_label"] = label_encoder.inverse_transform(y_true_encoded)
	results_df["correct_prediction"] = results_df["true_label"] == results_df["predicted_label"]

print("\n" + "=" * 60)
print("INFERENCE RESULTS")
print("=" * 60)

if y_true is not None:
	accuracy = accuracy_score(y_true_encoded, y_pred)
	print(f"\nAccuracy: {accuracy:.4f}")

	try:
		auc_score = roc_auc_score(y_true_encoded, y_pred_proba)
		print(f"AUC Score: {auc_score:.4f}")
	except ValueError:
		print("AUC Score: Could not calculate")

	print("\nClassification Report:")
	target_names = [str(cls) for cls in label_encoder.classes_]
	print(classification_report(y_true_encoded, y_pred, target_names=target_names))

	cm = confusion_matrix(y_true_encoded, y_pred)
	print("\nConfusion Matrix:")
	print(cm)

	plt.figure(figsize=(10, 8))
	sns.heatmap(
		cm,
		annot=True,
		fmt="d",
		cmap="Blues",
		xticklabels=label_encoder.classes_,
		yticklabels=label_encoder.classes_,
		cbar_kws={"label": "Count"},
	)
	plt.title("Confusion Matrix - Ass5 Classification Model", fontsize=16, fontweight="bold")
	plt.ylabel("True Label", fontsize=12)
	plt.xlabel("Predicted Label", fontsize=12)

	tn, fp, fn, tp = cm.ravel()
	stats_text = f"True Negatives: {tn}\nFalse Positives: {fp}\nFalse Negatives: {fn}\nTrue Positives: {tp}"
	stats_text += f"\n\nAccuracy: {accuracy:.4f}"
	plt.text(
		2.5,
		0.5,
		stats_text,
		fontsize=10,
		bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
		verticalalignment="center",
	)

	plt.tight_layout()
	output_path = os.path.join(INFERENCE_OUTPUT_DIR, "ass5_confusion_matrix.jpg")
	plt.savefig(output_path, dpi=300, bbox_inches="tight")
	print(f"\nConfusion matrix plot saved to: {output_path}")

	plt.figure(figsize=(10, 8))
	cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
	sns.heatmap(
		cm_normalized,
		annot=True,
		fmt=".2%",
		cmap="Greens",
		xticklabels=label_encoder.classes_,
		yticklabels=label_encoder.classes_,
		cbar_kws={"label": "Percentage"},
	)
	plt.title("Normalized Confusion Matrix - Ass5 Classification Model", fontsize=16, fontweight="bold")
	plt.ylabel("True Label", fontsize=12)
	plt.xlabel("Predicted Label", fontsize=12)

	plt.tight_layout()
	output_path_normalized = os.path.join(INFERENCE_OUTPUT_DIR, "ass5_confusion_matrix_normalized.jpg")
	plt.savefig(output_path_normalized, dpi=300, bbox_inches="tight")
	print(f"Normalized confusion matrix plot saved to: {output_path_normalized}")
else:
	print("\nNo ground-truth labels found. Skipping metrics and plots.")

output_csv_path = os.path.join(INFERENCE_OUTPUT_DIR, "ass5_predictions.csv")
results_df.to_csv(output_csv_path, index=False)
print(f"Predictions saved to: {output_csv_path}")

print("\n" + "=" * 60)
print("Inference completed successfully!")
print("=" * 60)
