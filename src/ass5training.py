import json
import os
import pickle

import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight


DATA_PATH = "data/ass5data.csv"
OUTPUT_DIR = "ass5examples"
MODEL_PATH = os.path.join(OUTPUT_DIR, "ass5_model.h5")
SCALER_PATH = os.path.join(OUTPUT_DIR, "ass5_scaler.pkl")
ENCODER_PATH = os.path.join(OUTPUT_DIR, "ass5_label_encoder.pkl")
COLUMNS_PATH = os.path.join(OUTPUT_DIR, "ass5_feature_columns.json")


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


print("Loading data...")
df = pd.read_csv(DATA_PATH)
df = clean_dataframe(df)

target_column = "ProdTaken"
df = fill_missing_values(df, target_column)

print(f"Dataset shape: {df.shape}")
print(f"\nColumn names: {df.columns.tolist()}")
print(f"\nTarget distribution:\n{df[target_column].value_counts()}")

X = df.drop(target_column, axis=1)
y = df[target_column].astype(int)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()
print(f"\nCategorical columns: {categorical_columns}")

X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
feature_columns = X_encoded.columns.tolist()

print(f"Features after encoding: {X_encoded.shape[1]}")

X_train, X_test, y_train, y_test = train_test_split(
	X_encoded,
	y_encoded,
	test_size=0.2,
	random_state=42,
	stratify=y_encoded,
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set size: {X_train_scaled.shape[0]}")
print(f"Testing set size: {X_test_scaled.shape[0]}")

print("\nBuilding neural network model...")
model = keras.Sequential([
	keras.layers.Dense(128, activation="relu", input_shape=(X_train_scaled.shape[1],)),
	keras.layers.Dropout(0.3),
	keras.layers.Dense(64, activation="relu"),
	keras.layers.Dropout(0.3),
	keras.layers.Dense(32, activation="relu"),
	keras.layers.Dropout(0.2),
	keras.layers.Dense(1, activation="sigmoid"),
])

model.compile(
	optimizer="adam",
	loss="binary_crossentropy",
	metrics=["accuracy", keras.metrics.AUC(name="auc")],
)

model.summary()

classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weight_dict = {int(cls): float(weight) for cls, weight in zip(classes, class_weights)}

print("\nTraining the model...")
history = model.fit(
	X_train_scaled,
	y_train,
	epochs=100,
	batch_size=32,
	validation_split=0.2,
	verbose=1,
	class_weight=class_weight_dict,
	callbacks=[
		keras.callbacks.EarlyStopping(
			monitor="val_loss",
			patience=8,
			restore_best_weights=True,
		),
		keras.callbacks.ReduceLROnPlateau(
			monitor="val_loss",
			factor=0.5,
			patience=4,
			min_lr=1e-5,
		),
	],
)

print("\nEvaluating the model...")
test_loss, test_accuracy, test_auc = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test AUC: {test_auc:.4f}")

y_pred_proba = model.predict(X_test_scaled, verbose=0)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

print("\nClassification Report:")
target_names = [str(cls) for cls in label_encoder.classes_]
print(classification_report(y_test, y_pred, target_names=target_names))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save(MODEL_PATH)
with open(SCALER_PATH, "wb") as scaler_file:
	pickle.dump(scaler, scaler_file)
with open(ENCODER_PATH, "wb") as encoder_file:
	pickle.dump(label_encoder, encoder_file)
with open(COLUMNS_PATH, "w") as columns_file:
	json.dump(feature_columns, columns_file)

print(f"\nModel saved to {MODEL_PATH}")
print(f"Scaler saved to {SCALER_PATH}")
print(f"Label encoder saved to {ENCODER_PATH}")
print(f"Feature columns saved to {COLUMNS_PATH}")
