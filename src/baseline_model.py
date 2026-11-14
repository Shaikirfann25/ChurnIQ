"""Baseline model module for ChurnIQ.

Trains and evaluates a Logistic Regression baseline model on the processed
training and validation partitions using balanced class weights, a fixed
random seed, and standardized features.
"""

from dataclasses import asdict
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import (
    BASELINE_MODEL_PATH,
    DEFAULT_BASELINE_CONFIG,
    ID_COLUMN,
    RANDOM_SEED,
    REPORTS_DIR,
    TARGET_COLUMN,
    TRAIN_DATA_FILE,
    VAL_DATA_FILE,
    BaselineHyperparameters,
    set_all_seeds,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ChurnIQ.BaselineModel")

# Default report output paths
BASELINE_METRICS_PATH: Path = REPORTS_DIR / "baseline_validation_metrics.json"
BASELINE_VAL_PRED_PATH: Path = (
    REPORTS_DIR / "baseline_val_predictions.parquet"
)


# ==============================================================================
# 1. DATA LOADING & FEATURE ISOLATION
# ==============================================================================
def load_train_and_val_data(
    train_path: Path = TRAIN_DATA_FILE,
    val_path: Path = VAL_DATA_FILE,
) -> Tuple[
    pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series, pd.Series
]:
    """Load training and validation Parquet partitions and isolate features.

    Strictly removes customerID and Churn from the feature matrix X.
    Does NOT touch the quarantined test split.

    Args:
        train_path: Path to processed training parquet.
        val_path: Path to processed validation parquet.

    Returns:
        Tuple:
            - X_train: Training features (46 columns).
            - y_train: Training target labels (0 or 1).
            - ids_train: Training customer IDs.
            - X_val: Validation features (46 columns).
            - y_val: Validation target labels (0 or 1).
            - ids_val: Validation customer IDs.

    Raises:
        FileNotFoundError: If partition files do not exist.
        ValueError: If ID or Target columns are missing.
    """
    if not train_path.exists():
        raise FileNotFoundError(
            f"Training split not found at: {train_path}. "
            "Please run src/data_pipeline.py first."
        )
    if not val_path.exists():
        raise FileNotFoundError(
            f"Validation split not found at: {val_path}. "
            "Please run src/data_pipeline.py first."
        )

    logger.info(f"Loading training partition from: {train_path}")
    train_df = pd.read_parquet(train_path)

    logger.info(f"Loading validation partition from: {val_path}")
    val_df = pd.read_parquet(val_path)

    for name, df in [("Train", train_df), ("Val", val_df)]:
        if ID_COLUMN not in df.columns or TARGET_COLUMN not in df.columns:
            raise ValueError(
                f"{name} partition is missing required metadata/target columns."
            )

    # Separate metadata and target from features
    ids_train = train_df[ID_COLUMN].copy()
    y_train = train_df[TARGET_COLUMN].copy().astype(int)
    X_train = train_df.drop(columns=[ID_COLUMN, TARGET_COLUMN])

    ids_val = val_df[ID_COLUMN].copy()
    y_val = val_df[TARGET_COLUMN].copy().astype(int)
    X_val = val_df.drop(columns=[ID_COLUMN, TARGET_COLUMN])

    # Integrity assertions
    assert (
        X_train.columns.tolist() == X_val.columns.tolist()
    ), "Feature column mismatch between train and val partitions!"
    assert (
        ID_COLUMN not in X_train.columns
    ), f"Leakage alert: {ID_COLUMN} is present in X_train!"
    assert (
        TARGET_COLUMN not in X_train.columns
    ), f"Leakage alert: {TARGET_COLUMN} is present in X_train!"

    logger.info(
        f"Data loaded successfully. "
        f"X_train: {X_train.shape}, y_train: {y_train.shape} | "
        f"X_val: {X_val.shape}, y_val: {y_val.shape}"
    )

    return X_train, y_train, ids_train, X_val, y_val, ids_val


# ==============================================================================
# 2. MODEL TRAINING
# ==============================================================================
def train_baseline_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: BaselineHyperparameters = DEFAULT_BASELINE_CONFIG,
) -> LogisticRegression:
    """Train a Logistic Regression baseline model on training data only.

    Configured with class_weight='balanced', fixed seed, and lbfgs solver.

    Args:
        X_train: Training feature matrix.
        y_train: Training target labels.
        config: Hyperparameter configuration dataclass.

    Returns:
        LogisticRegression: Trained model instance.

    Raises:
        RuntimeWarning: If the optimizer fails to reach convergence within max_iter.
    """
    logger.info("Initializing Logistic Regression baseline model...")
    logger.info(
        f"Hyperparameters: max_iter={config.max_iter}, C={config.c_regularization}, "
        f"class_weight='{config.class_weight}', solver='{config.solver}', "
        f"random_state={config.random_state}"
    )

    model = LogisticRegression(
        max_iter=config.max_iter,
        C=config.c_regularization,
        class_weight=config.class_weight,
        solver=config.solver,
        random_state=config.random_state,
    )

    logger.info("Fitting Logistic Regression strictly on X_train...")
    model.fit(X_train, y_train)

    # Convergence check
    iterations_run = int(model.n_iter_[0])
    logger.info(
        f"Model converged in {iterations_run} iterations "
        f"(max_iter limit: {config.max_iter})."
    )
    if iterations_run >= config.max_iter:
        logger.warning(
            f"Model reached maximum iteration limit ({config.max_iter}). "
            "Consider increasing max_iter or adjusting solver tolerance."
        )

    return model


# ==============================================================================
# 3. VALIDATION EVALUATION (Default Threshold 0.50)
# ==============================================================================
def evaluate_baseline_validation(
    model: LogisticRegression,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    threshold: float = 0.50,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """Compute validation metrics at the default threshold of 0.50.

    Computes:
    - ROC-AUC (threshold-independent)
    - PR-AUC (Average Precision, threshold-independent)
    - Precision (@ threshold)
    - Recall (@ threshold)
    - F1-Score (@ threshold)
    - Confusion Matrix (@ threshold)

    Note:
        Does NOT perform threshold optimization. Optimal threshold search
        is reserved for Module 4 (threshold_optimizer.py).

    Args:
        model: Trained LogisticRegression instance.
        X_val: Validation features.
        y_val: Validation target labels.
        threshold: Decision threshold for binary classification (default 0.50).

    Returns:
        Tuple:
            - metrics: Dictionary of evaluation metrics.
            - y_val_proba: Continuous churn probabilities array.
            - y_val_pred: Binary predicted classes array.
    """
    logger.info("Evaluating baseline model on validation partition...")

    # Predict continuous probabilities
    val_proba_matrix = model.predict_proba(X_val)
    y_val_proba = val_proba_matrix[:, 1]

    # Validate probability bounds [0.0, 1.0]
    if np.any(y_val_proba < 0.0) or np.any(y_val_proba > 1.0):
        raise ValueError(
            "Model produced invalid probabilities outside the [0, 1] interval!"
        )

    # Threshold-based predictions at default 0.50
    y_val_pred = (y_val_proba >= threshold).astype(int)

    # Metric computations
    roc_auc = float(roc_auc_score(y_val, y_val_proba))
    pr_auc = float(average_precision_score(y_val, y_val_proba))
    precision = float(precision_score(y_val, y_val_pred, zero_division=0))
    recall = float(recall_score(y_val, y_val_pred, zero_division=0))
    f1 = float(f1_score(y_val, y_val_pred, zero_division=0))

    cm = confusion_matrix(y_val, y_val_pred)
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    metrics: Dict[str, Any] = {
        "model_name": "Logistic Regression Baseline",
        "evaluation_partition": "validation",
        "threshold": float(threshold),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": {
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "matrix_2x2": [[tn, fp], [fn, tp]],
        },
        "sample_counts": {
            "total_samples": len(y_val),
            "actual_churn_positives": int(y_val.sum()),
            "actual_churn_negatives": int((y_val == 0).sum()),
            "predicted_positives": int(y_val_pred.sum()),
            "predicted_negatives": int((y_val_pred == 0).sum()),
        },
        "hyperparameters": {
            "max_iter": model.max_iter,
            "C": model.C,
            "class_weight": model.class_weight,
            "solver": model.solver,
            "iterations_to_converge": int(model.n_iter_[0]),
        },
    }

    logger.info(
        f"Validation Results (@ threshold={threshold:.2f}): "
        f"ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}, "
        f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}"
    )
    logger.info(
        f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}"
    )

    return metrics, y_val_proba, y_val_pred


# ==============================================================================
# 4. ARTIFACT PERSISTENCE
# ==============================================================================
def save_baseline_artifacts(
    model: LogisticRegression,
    metrics: Dict[str, Any],
    val_predictions_df: pd.DataFrame,
    model_path: Path = BASELINE_MODEL_PATH,
    metrics_path: Path = BASELINE_METRICS_PATH,
    predictions_path: Path = BASELINE_VAL_PRED_PATH,
) -> None:
    """Serialize the trained baseline model, validation metrics, and predictions.

    Args:
        model: Trained LogisticRegression model.
        metrics: Dictionary of validation metrics.
        val_predictions_df: DataFrame with predictions and customerID.
        model_path: Destination for baseline model joblib.
        metrics_path: Destination for validation metrics JSON.
        predictions_path: Destination for validation predictions Parquet.
    """
    logger.info(f"Saving baseline model artifact to: {model_path}")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    logger.info(f"Saving validation metrics to: {metrics_path}")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Saving validation predictions to: {predictions_path}")
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    val_predictions_df.to_parquet(
        predictions_path, index=False, engine="pyarrow"
    )

    logger.info("All baseline artifacts saved successfully.")


# ==============================================================================
# 5. MODEL INTEGRITY & REPRODUCIBILITY VERIFICATION
# ==============================================================================
def verify_saved_baseline(
    model_path: Path,
    X_val: pd.DataFrame,
    expected_proba: np.ndarray,
) -> None:
    """Verify that the saved model artifact reloads and reproduces exact predictions.

    Args:
        model_path: Path to serialized model.
        X_val: Validation feature matrix.
        expected_proba: Probabilities generated before saving.

    Raises:
        AssertionError: If loaded model predictions differ from expected probabilities.
    """
    logger.info(f"Verifying model reloadability from: {model_path}...")
    assert model_path.exists(), f"Saved model file not found at: {model_path}"

    reloaded_model: LogisticRegression = joblib.load(model_path)
    reloaded_proba = reloaded_model.predict_proba(X_val)[:, 1]

    # Check identical probability outputs
    max_abs_diff = float(np.max(np.abs(reloaded_proba - expected_proba)))
    assert max_abs_diff < 1e-9, (
        f"Reloaded model outputs differ from in-memory model! "
        f"Max absolute diff: {max_abs_diff}"
    )

    logger.info(
        f"Model reload verified successfully. Max diff = {max_abs_diff:.2e}"
    )


# ==============================================================================
# 6. PIPELINE ORCHESTRATOR
# ==============================================================================
def run_baseline_pipeline(
    train_path: Path = TRAIN_DATA_FILE,
    val_path: Path = VAL_DATA_FILE,
    config: BaselineHyperparameters = DEFAULT_BASELINE_CONFIG,
    threshold: float = 0.50,
) -> Tuple[LogisticRegression, Dict[str, Any], pd.DataFrame]:
    """Execute the end-to-end baseline training and validation pipeline.

    Args:
        train_path: Training partition path.
        val_path: Validation partition path.
        config: Hyperparameters dataclass.
        threshold: Classification threshold (default 0.50).

    Returns:
        Tuple:
            - model: Trained LogisticRegression model.
            - metrics: Evaluated validation metrics dict.
            - val_predictions_df: Enriched validation DataFrame.
    """
    set_all_seeds(config.random_state)

    # 1. Load data
    X_train, y_train, ids_train, X_val, y_val, ids_val = load_train_and_val_data(
        train_path=train_path, val_path=val_path
    )

    # 2. Train baseline model
    model = train_baseline_model(X_train, y_train, config=config)

    # 3. Evaluate on validation set
    metrics, y_val_proba, y_val_pred = evaluate_baseline_validation(
        model=model, X_val=X_val, y_val=y_val, threshold=threshold
    )

    # 4. Construct validation predictions record
    val_predictions_df = pd.DataFrame(
        {
            ID_COLUMN: ids_val.values,
            "actual_churn": y_val.values,
            "churn_probability": y_val_proba,
            f"predicted_churn_at_{threshold:.2f}": y_val_pred,
        }
    )

    # 5. Save artifacts
    save_baseline_artifacts(
        model=model,
        metrics=metrics,
        val_predictions_df=val_predictions_df,
    )

    # 6. Verify reloadability
    verify_saved_baseline(
        model_path=BASELINE_MODEL_PATH,
        X_val=X_val,
        expected_proba=y_val_proba,
    )

    return model, metrics, val_predictions_df


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("CHURNIQ - EXECUTING BASELINE LOGISTIC REGRESSION PIPELINE")
    print("=" * 70 + "\n")
    trained_model, eval_metrics, val_preds = run_baseline_pipeline()
    print("\n" + "=" * 70)
    print("BASELINE PIPELINE EXECUTION COMPLETED")
    print("=" * 70 + "\n")

