import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers, callbacks
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.utils import class_weight
import joblib

# ============================================================
# Configuration
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_DATA_FILE = os.path.join(BASE_DIR, 'data', 'train.csv')
TEST_DATA_FILE = os.path.join(BASE_DIR, 'data', 'test.csv')

MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'examples')
PREPROCESSOR_FILE = os.path.join(MODEL_SAVE_DIR, 'preprocessor.pkl')

BATCH_SIZE = 32
EPOCHS = 120
LEARNING_RATE = 0.0005
TARGET_COL = 'ProdTaken'

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# ============================================================
# Data Preparation
# ============================================================


def clean_data(df):
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].replace('Fe Male', 'Female')
    return df


def load_data_files():
    if not os.path.exists(TRAIN_DATA_FILE):
        raise FileNotFoundError(f"Train file not found at: {TRAIN_DATA_FILE}")
    if not os.path.exists(TEST_DATA_FILE):
        raise FileNotFoundError(f"Test file not found at: {TEST_DATA_FILE}")

    print(f"Loading Train: {TRAIN_DATA_FILE}")
    print(f"Loading Test:  {TEST_DATA_FILE}")

    df_train = pd.read_csv(TRAIN_DATA_FILE)
    df_test = pd.read_csv(TEST_DATA_FILE)

    df_train = clean_data(df_train)
    df_test = clean_data(df_test)

    if TARGET_COL not in df_train.columns:
        raise ValueError(f"Target '{TARGET_COL}' missing from Training data.")

    return df_train, df_test


def create_preprocessor(X):
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    print(
        f"Features: {len(numeric_features)} Numeric, {len(categorical_features)} Categorical")

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor

# ============================================================
# Model Creation: Wide & Deep Architecture
# ============================================================


def create_wide_and_deep_model(input_dim):
    """
    Implements a Wide & Deep architecture.
    It combines a linear path (Wide) with a deep path (Deep)
    to capture both simple rules and complex patterns.
    """
    # Input
    inputs = layers.Input(shape=(input_dim,), name="input_features")

    # --- Deep Path (The Heavy Lifter) ---
    x = layers.Dense(512, activation='relu',
                     kernel_regularizer=regularizers.l2(0.0001))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(256, activation='relu',
                     kernel_regularizer=regularizers.l2(0.0001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    # --- Wide Path (The Shortcut) ---
    # We concatenate the original inputs directly with the deep features
    # This allows the model to see the raw data right at the end
    combined = layers.concatenate([x, inputs])

    # --- Output Head ---
    # A final dense layer to mix the Wide and Deep signals
    z = layers.Dense(64, activation='relu')(combined)
    outputs = layers.Dense(1, activation='sigmoid')(z)

    model = keras.Model(inputs=inputs, outputs=outputs,
                        name="Wide_and_Deep_Tabular")
    return model

# ============================================================
# Visualization
# ============================================================


def plot_training_history(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax1.set_title('Accuracy')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_title('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"History plot saved to {save_path}")

# ============================================================
# Main Execution
# ============================================================


def main():
    print("=" * 60)
    print("Class 5.1 Training (Wide & Deep Architecture)")
    print("=" * 60)

    np.random.seed(42)
    tf.random.set_seed(42)

    # 1. Load Data
    try:
        df_train_full, df_test = load_data_files()
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    X_train_full = df_train_full.drop(TARGET_COL, axis=1)
    y_train_full = df_train_full[TARGET_COL]

    # Check for test labels
    if TARGET_COL in df_test.columns:
        X_test = df_test.drop(TARGET_COL, axis=1)
        y_test = df_test[TARGET_COL]
        has_test_labels = True
    else:
        X_test = df_test
        y_test = None
        has_test_labels = False

    # 3. Validation Split
    print("\nSplitting train data...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.2,
        random_state=42,
        stratify=y_train_full
    )

    # Compute Class Weights (Keep this! It helps Recall)
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = dict(enumerate(weights))
    print(f"Class Weights: {class_weights}")

    # 4. Preprocessing
    print("Fitting preprocessor...")
    preprocessor = create_preprocessor(X_train)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    joblib.dump(preprocessor, PREPROCESSOR_FILE)
    print(f"✅ Preprocessor saved.")

    # 5. Build Model
    input_dim = X_train_processed.shape[1]
    model = create_wide_and_deep_model(input_dim)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Recall(
            name='recall'), tf.keras.metrics.Precision(name='precision')]
    )

    # 6. Training
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODEL_SAVE_DIR, f'class5.1_model_{timestamp}.h5')
    history_plot_path = os.path.join(
        MODEL_SAVE_DIR, f'class5.1_history_{timestamp}.png')

    callbacks_list = [
        callbacks.ModelCheckpoint(
            model_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
        callbacks.EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=6, min_lr=1e-6, verbose=1)
    ]

    print(f"\nStarting training...")
    history = model.fit(
        X_train_processed, y_train,
        validation_data=(X_val_processed, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks_list,
        class_weight=class_weights,
        verbose=1
    )

    # 7. Final Evaluation
    print("\n" + "="*30)
    print("FINAL RESULTS")
    print("="*30)

    print("\n--- Internal Validation Set ---")
    val_metrics = model.evaluate(X_val_processed, y_val, verbose=0)
    print(f"Loss: {val_metrics[0]:.4f} | Accuracy: {val_metrics[1]:.4f}")

    if has_test_labels:
        print("\n--- External Test Set (The Final Score) ---")
        test_metrics = model.evaluate(X_test_processed, y_test, verbose=0)
        print(f"Loss: {test_metrics[0]:.4f} | Accuracy: {test_metrics[1]:.4f}")
        print(
            f"Recall: {test_metrics[2]:.4f} | Precision: {test_metrics[3]:.4f}")

    final_model_path = os.path.join(MODEL_SAVE_DIR, 'class5.1_model_final.h5')
    model.save(final_model_path)
    print(f"\n✅ Final model saved to: {final_model_path}")

    plot_training_history(history, history_plot_path)


if __name__ == "__main__":
    main()
