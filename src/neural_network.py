"""
ChurnIQ - Module 3: Multilayer Perceptron Neural Network
=========================================================
Keras 3 with PyTorch backend. No TensorFlow imports.
"""

import os

# Must be set BEFORE any keras import
os.environ.setdefault("KERAS_BACKEND", "torch")

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Config import (also sets KERAS_BACKEND via os.environ.setdefault)
# ---------------------------------------------------------------------------
from src.config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    RANDOM_SEED,
    MLPHyperparameters,
    set_all_seeds,
)

DEFAULT_MLP_CONFIG = MLPHyperparameters()

# ---------------------------------------------------------------------------
# Keras import (after backend env var is set)
# ---------------------------------------------------------------------------
import keras
from keras import layers, regularizers

# ---------------------------------------------------------------------------
# Baseline metrics for comparison (from Module 2 run)
# ---------------------------------------------------------------------------
BASELINE_METRICS = {
    "model": "Logistic Regression",
    "roc_auc": 0.8449,
    "pr_auc": 0.6314,
    "precision": 0.5067,
    "recall": 0.8143,
    "f1": 0.6247,
}

# Expected 46 features in exact order from preprocessor
EXPECTED_FEATURES = [
    "tenure", "MonthlyCharges", "TotalCharges",
    "gender_Female", "gender_Male",
    "SeniorCitizen_0", "SeniorCitizen_1",
    "Partner_No", "Partner_Yes",
    "Dependents_No", "Dependents_Yes",
    "PhoneService_No", "PhoneService_Yes",
    "MultipleLines_No", "MultipleLines_No phone service", "MultipleLines_Yes",
    "InternetService_DSL", "InternetService_Fiber optic", "InternetService_No",
    "OnlineSecurity_No", "OnlineSecurity_No internet service", "OnlineSecurity_Yes",
    "OnlineBackup_No", "OnlineBackup_No internet service", "OnlineBackup_Yes",
    "DeviceProtection_No", "DeviceProtection_No internet service", "DeviceProtection_Yes",
    "TechSupport_No", "TechSupport_No internet service", "TechSupport_Yes",
    "StreamingTV_No", "StreamingTV_No internet service", "StreamingTV_Yes",
    "StreamingMovies_No", "StreamingMovies_No internet service", "StreamingMovies_Yes",
    "Contract_Month-to-month", "Contract_One year", "Contract_Two year",
    "PaperlessBilling_No", "PaperlessBilling_Yes",
    "PaymentMethod_Bank transfer (automatic)",
    "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check",
    "PaymentMethod_Mailed check",
]


# ===========================================================================
# 1. Backend verification
# ===========================================================================
def verify_backend() -> None:
    """Assert Keras is using the PyTorch backend."""
    active = keras.backend.backend()
    if active != "torch":
        raise RuntimeError(
            f"Expected Keras backend 'torch', got '{active}'. "
            "Set KERAS_BACKEND=torch before importing Keras."
        )
    print(f"[OK] Keras backend: {active}")


# ===========================================================================
# 2. Data loading
# ===========================================================================
def load_train_val_data():
    """
    Load train and validation Parquet splits.

    Returns
    -------
    X_train, y_train, ids_train, X_val, y_val, ids_val
        Feature arrays as float32 numpy arrays.
        y arrays shaped (N, 1).
    """
    train_path = PROCESSED_DATA_DIR / "train_split.parquet"
    val_path   = PROCESSED_DATA_DIR / "val_split.parquet"

    for p in (train_path, val_path):
        if not p.exists():
            raise FileNotFoundError(f"Parquet file not found: {p}")

    df_train = pd.read_parquet(train_path)
    df_val   = pd.read_parquet(val_path)

    print(f"[OK] Loaded train: {df_train.shape}, val: {df_val.shape}")

    # Verify expected feature columns are present
    missing_train = [c for c in EXPECTED_FEATURES if c not in df_train.columns]
    missing_val   = [c for c in EXPECTED_FEATURES if c not in df_val.columns]
    if missing_train or missing_val:
        raise ValueError(
            f"Missing features — train: {missing_train}, val: {missing_val}"
        )

    # Separate metadata
    ids_train = df_train["customerID"].values
    ids_val   = df_val["customerID"].values

    y_train = df_train["Churn"].values.astype(np.float32).reshape(-1, 1)
    y_val   = df_val["Churn"].values.astype(np.float32).reshape(-1, 1)

    # Features only (exact 46, exact order)
    X_train = df_train[EXPECTED_FEATURES].values.astype(np.float32)
    X_val   = df_val[EXPECTED_FEATURES].values.astype(np.float32)

    assert X_train.shape[1] == 46, f"Expected 46 features, got {X_train.shape[1]}"
    assert X_val.shape[1] == 46,   f"Expected 46 features, got {X_val.shape[1]}"

    print(f"[OK] X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"[OK] X_val:   {X_val.shape},   y_val:   {y_val.shape}")

    return X_train, y_train, ids_train, X_val, y_val, ids_val


# ===========================================================================
# 3. Class weight computation
# ===========================================================================
def compute_class_weights(y_train: np.ndarray) -> dict:
    """
    Compute balanced class weights from training labels.

    Parameters
    ----------
    y_train : ndarray of shape (N, 1)

    Returns
    -------
    dict  {0: w0, 1: w1}
    """
    y_flat = y_train.ravel().astype(int)
    classes = np.array([0, 1])
    weights = compute_class_weight("balanced", classes=classes, y=y_flat)
    cw = {0: float(weights[0]), 1: float(weights[1])}
    print(f"[OK] Class weights: {cw}")
    return cw


# ===========================================================================
# 4. Model architecture
# ===========================================================================
def build_mlp_model(
    n_features: int = 46,
    config: MLPHyperparameters = DEFAULT_MLP_CONFIG,
) -> keras.Model:
    """
    Build the ChurnIQ MLP architecture.

    Architecture (approved, locked):
      Input(46) → Dense(64,ReLU,L2) → Dropout(0.25)
               → Dense(32,ReLU,L2) → Dropout(0.25)
               → Dense(16,ReLU,L2)
               → Dense(1,sigmoid)
    """
    l2_reg = config.l2_regularization
    dr     = config.dropout_rate
    units  = config.hidden_units  # [64, 32, 16]

    model = keras.Sequential(
        [
            layers.Input(shape=(n_features,), name="input"),
            layers.Dense(
                units[0], activation="relu",
                kernel_regularizer=regularizers.l2(l2_reg),
                name="dense_1",
            ),
            layers.Dropout(dr, name="dropout_1"),
            layers.Dense(
                units[1], activation="relu",
                kernel_regularizer=regularizers.l2(l2_reg),
                name="dense_2",
            ),
            layers.Dropout(dr, name="dropout_2"),
            layers.Dense(
                units[2], activation="relu",
                kernel_regularizer=regularizers.l2(l2_reg),
                name="dense_3",
            ),
            layers.Dense(1, activation="sigmoid", name="output"),
        ],
        name="ChurnIQ_MLP",
    )
    model.summary()
    return model


# ===========================================================================
# 5. Training
# ===========================================================================
def train_mlp(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_weights: dict,
    config: MLPHyperparameters = DEFAULT_MLP_CONFIG,
):
    """
    Compile and train the MLP.

    EarlyStopping patience=3 (per explicit user requirement).
    Class weights applied ONLY in model.fit(), NOT in compile.

    Returns
    -------
    history : keras History object
    """
    optimizer = keras.optimizers.Adam(learning_rate=config.learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=config.loss_function,
        metrics=["accuracy"],
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,                   # explicit user requirement (overrides config=10)
        restore_best_weights=True,
        verbose=1,
    )

    print("\n[INFO] Starting MLP training...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=config.max_epochs,
        batch_size=config.batch_size,
        callbacks=[early_stop],
        class_weight=class_weights,
        verbose=1,
    )
    print("[OK] Training complete.")
    return history


# ===========================================================================
# 6. Evaluation
# ===========================================================================
def evaluate_mlp_validation(
    model: keras.Model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    threshold: float = 0.50,
) -> tuple:
    """
    Evaluate MLP on validation set.

    Parameters
    ----------
    model     : trained Keras model
    X_val     : float32 (N, 46)
    y_val     : float32 (N, 1)
    threshold : decision threshold (default 0.50)

    Returns
    -------
    (metrics_dict, y_proba, y_pred)
    """
    y_proba_raw = model.predict(X_val, verbose=0)
    y_proba = y_proba_raw.ravel()

    # Sanity checks
    assert np.all(np.isfinite(y_proba)), "Probabilities contain non-finite values!"
    assert np.all((y_proba >= 0) & (y_proba <= 1)), "Probabilities outside [0,1]!"

    y_true = y_val.ravel().astype(int)
    y_pred = (y_proba >= threshold).astype(int)

    roc_auc  = roc_auc_score(y_true, y_proba)
    pr_auc   = average_precision_score(y_true, y_proba)
    prec     = precision_score(y_true, y_pred, zero_division=0)
    rec      = recall_score(y_true, y_pred, zero_division=0)
    f1       = f1_score(y_true, y_pred, zero_division=0)
    cm       = confusion_matrix(y_true, y_pred)

    metrics = {
        "threshold":  threshold,
        "roc_auc":    round(float(roc_auc), 4),
        "pr_auc":     round(float(pr_auc),  4),
        "precision":  round(float(prec),    4),
        "recall":     round(float(rec),     4),
        "f1":         round(float(f1),      4),
        "confusion_matrix": cm.tolist(),
    }

    return metrics, y_proba, y_pred


# ===========================================================================
# 7. Save artifacts
# ===========================================================================
def save_artifacts(
    model: keras.Model,
    history,
    metrics: dict,
    ids_val: np.ndarray,
    y_val: np.ndarray,
    y_proba: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """Save model, metrics, training history, and validation predictions."""

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Model ---
    model_path = MODELS_DIR / "mlp_churn_model.keras"
    model.save(str(model_path))
    print(f"[OK] Model saved → {model_path}")

    # --- Validation metrics ---
    metrics_path = REPORTS_DIR / "mlp_validation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[OK] Metrics saved → {metrics_path}")

    # --- Training history ---
    hist_dict = {}
    for key, values in history.history.items():
        hist_dict[key] = [round(float(v), 6) for v in values]

    history_path = REPORTS_DIR / "mlp_training_history.json"
    with open(history_path, "w") as f:
        json.dump(hist_dict, f, indent=2)
    print(f"[OK] Training history saved → {history_path}")

    # --- Validation predictions parquet ---
    threshold = metrics["threshold"]
    preds_df = pd.DataFrame(
        {
            "customerID":              ids_val,
            "actual_churn":            y_val.ravel().astype(int),
            "churn_probability":       y_proba,
            "predicted_churn_at_0.50": y_pred,
        }
    )
    preds_path = REPORTS_DIR / "mlp_val_predictions.parquet"
    preds_df.to_parquet(preds_path, index=False)
    print(f"[OK] Val predictions saved → {preds_path}")


# ===========================================================================
# 8. Verification
# ===========================================================================
def verify_saved_mlp(
    model_path: Path,
    X_val: np.ndarray,
    expected_proba: np.ndarray,
    tol: float = 1e-5,
) -> None:
    """Reload the saved model and check predictions match within tolerance."""
    reloaded = keras.models.load_model(str(model_path))
    reloaded_proba = reloaded.predict(X_val, verbose=0).ravel()
    max_diff = float(np.max(np.abs(reloaded_proba - expected_proba)))
    if max_diff >= tol:
        raise AssertionError(
            f"Save/reload verification FAILED: max abs diff = {max_diff:.2e} (tol={tol:.0e})"
        )
    print(f"[OK] Save/reload verification passed (max diff = {max_diff:.2e})")


# ===========================================================================
# 9. Comparison report
# ===========================================================================
def print_comparison_report(mlp_metrics: dict) -> None:
    """Print side-by-side comparison with Logistic Regression baseline."""
    b = BASELINE_METRICS
    m = mlp_metrics

    def delta(key):
        d = m[key] - b[key]
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:+.4f}"

    print("\n" + "=" * 65)
    print("  MODULE 3  —  MLP vs. Logistic Regression Baseline")
    print("=" * 65)
    print(f"  {'Metric':<18} {'LR Baseline':>12} {'MLP (val)':>12} {'Delta':>10}")
    print("-" * 65)
    for key in ("roc_auc", "pr_auc", "precision", "recall", "f1"):
        label = key.replace("_", " ").upper()
        print(
            f"  {label:<18} {b[key]:>12.4f} {m[key]:>12.4f} {delta(key):>10}"
        )
    print("-" * 65)
    cm = m["confusion_matrix"]
    print(f"  Confusion Matrix  TN={cm[0][0]:4d}  FP={cm[0][1]:4d}")
    print(f"                    FN={cm[1][0]:4d}  TP={cm[1][1]:4d}")
    print("=" * 65)

    # Honest verdict
    if m["roc_auc"] > b["roc_auc"]:
        print("  VERDICT: MLP IMPROVES on ROC-AUC vs. baseline.")
    elif m["roc_auc"] == b["roc_auc"]:
        print("  VERDICT: MLP MATCHES baseline ROC-AUC.")
    else:
        print("  VERDICT: MLP is below baseline on ROC-AUC.")
    print("=" * 65 + "\n")


# ===========================================================================
# 10. Orchestrator
# ===========================================================================
def run_mlp_pipeline() -> None:
    """End-to-end MLP pipeline: verify → load → build → train → evaluate → save → report."""

    print("\n" + "=" * 65)
    print("  ChurnIQ — Module 3: MLP Neural Network")
    print("=" * 65)

    # 0. Backend check
    verify_backend()

    # 1. Seed everything
    set_all_seeds()

    # 2. Load data
    X_train, y_train, ids_train, X_val, y_val, ids_val = load_train_val_data()

    # 3. Class weights
    class_weights = compute_class_weights(y_train)

    # 4. Build model
    model = build_mlp_model(n_features=X_train.shape[1], config=DEFAULT_MLP_CONFIG)

    # 5. Train
    history = train_mlp(model, X_train, y_train, X_val, y_val, class_weights, DEFAULT_MLP_CONFIG)

    # 6. Evaluate
    metrics, y_proba, y_pred = evaluate_mlp_validation(model, X_val, y_val, threshold=0.50)

    # 7. Save
    save_artifacts(model, history, metrics, ids_val, y_val, y_proba, y_pred)

    # 8. Verify saved model
    verify_saved_mlp(MODELS_DIR / "mlp_churn_model.keras", X_val, y_proba)

    # 9. Comparison report
    print_comparison_report(metrics)

    print("[DONE] Module 3 complete. Awaiting user review before next module.")


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    run_mlp_pipeline()

