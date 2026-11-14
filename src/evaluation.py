"""
ChurnIQ - Module 5: Final Test Evaluation
==========================================
Unbiased benchmark of both models on the quarantined test partition.

Rules enforced in this module:
  - Test partition loaded exactly once, read-only.
  - No training, fine-tuning, recalibration, or threshold changes.
  - Thresholds read exclusively from models/threshold_metadata.json.
  - Neither model nor preprocessor is modified.
  - customerID is metadata only — never passed to a model.
"""

import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for file output
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Project path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
)

import keras

# ---------------------------------------------------------------------------
# Expected 46 features (locked — must match preprocessor and both models)
# ---------------------------------------------------------------------------
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

# Validation metrics (ground truth from prior modules — for generalization analysis)
VALIDATION_METRICS = {
    "LogisticRegression": {
        "roc_auc":  0.8449,
        "pr_auc":   0.6314,
        "f1_default":  0.6247,
        "f1_optimal":  0.6329,
        "precision_optimal": 0.5220,
        "recall_optimal":    0.8036,
    },
    "MLP": {
        "roc_auc":  0.8454,
        "pr_auc":   0.6321,
        "f1_default":  0.6289,
        "f1_optimal":  0.6319,
        "precision_optimal": 0.5134,
        "recall_optimal":    0.8214,
    },
}


# ===========================================================================
# 1. Load artifacts
# ===========================================================================
def load_threshold_metadata() -> dict:
    path = MODELS_DIR / "threshold_metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Threshold metadata not found: {path}")
    with open(path) as f:
        meta = json.load(f)
    lr_thr  = meta["models"]["logistic_regression"]["optimal_threshold"]
    mlp_thr = meta["models"]["mlp"]["optimal_threshold"]
    print(f"[OK] Thresholds loaded from metadata — LR: {lr_thr}, MLP: {mlp_thr}")
    return meta


def load_test_data() -> tuple:
    """
    Load test_split.parquet exactly once.

    Returns
    -------
    X_test  : float32 ndarray (N, 46)
    y_test  : int ndarray (N,)
    ids_test: str ndarray (N,)
    """
    test_path = PROCESSED_DATA_DIR / "test_split.parquet"
    if not test_path.exists():
        raise FileNotFoundError(f"Test partition not found: {test_path}")

    df = pd.read_parquet(test_path)
    print(f"[OK] Test partition loaded: {df.shape}")

    # Verify feature columns present
    missing = [c for c in EXPECTED_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing features in test partition: {missing}")

    ids_test = df["customerID"].values
    y_test   = df["Churn"].values.astype(int)
    X_test   = df[EXPECTED_FEATURES].values.astype(np.float32)

    assert X_test.shape[1] == 46, f"Expected 46 features, got {X_test.shape[1]}"
    print(f"[OK] X_test: {X_test.shape}  Churn rate: {y_test.mean():.4f}")
    return X_test, y_test, ids_test


def load_lr_model():
    path = MODELS_DIR / "baseline_logistic.joblib"
    if not path.exists():
        raise FileNotFoundError(f"LR model not found: {path}")
    model = joblib.load(path)
    print(f"[OK] LR model loaded from {path}")
    return model


def load_mlp_model():
    path = MODELS_DIR / "mlp_churn_model.keras"
    if not path.exists():
        raise FileNotFoundError(f"MLP model not found: {path}")
    model = keras.models.load_model(str(path))
    print(f"[OK] MLP model loaded from {path}")
    return model


# ===========================================================================
# 2. Predict
# ===========================================================================
def predict_lr(lr_model, X_test: np.ndarray) -> np.ndarray:
    proba = lr_model.predict_proba(X_test)[:, 1]
    assert np.all(np.isfinite(proba)), "LR probabilities contain non-finite values!"
    assert np.all((proba >= 0) & (proba <= 1)), "LR probabilities outside [0,1]!"
    print(f"[OK] LR predictions: min={proba.min():.4f}, max={proba.max():.4f}, mean={proba.mean():.4f}")
    return proba


def predict_mlp(mlp_model, X_test: np.ndarray) -> np.ndarray:
    proba = mlp_model.predict(X_test, verbose=0).ravel()
    assert np.all(np.isfinite(proba)), "MLP probabilities contain non-finite values!"
    assert np.all((proba >= 0) & (proba <= 1)), "MLP probabilities outside [0,1]!"
    print(f"[OK] MLP predictions: min={proba.min():.4f}, max={proba.max():.4f}, mean={proba.mean():.4f}")
    return proba


# ===========================================================================
# 3. Compute metrics
# ===========================================================================
def full_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    model_name: str,
    threshold_label: str,
) -> dict:
    """
    Compute the full metric suite at the given threshold.

    Metrics computed:
      Threshold-independent: ROC-AUC, PR-AUC
      Threshold-dependent:   Accuracy, Precision, Recall, F1, Specificity,
                             FPR, FNR, Confusion Matrix
    """
    y_pred = (y_proba >= threshold).astype(int)
    cm     = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    total = int(tn + fp + fn + tp)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr_val     = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr_val     = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "model":           model_name,
        "threshold_label": threshold_label,
        "threshold":       round(threshold, 2),
        "n_test":          total,
        # Threshold-independent
        "roc_auc":         round(float(roc_auc_score(y_true, y_proba)),           4),
        "pr_auc":          round(float(average_precision_score(y_true, y_proba)), 4),
        # Threshold-dependent
        "accuracy":        round(float(accuracy_score(y_true, y_pred)),           4),
        "precision":       round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":          round(float(recall_score(y_true, y_pred, zero_division=0)),    4),
        "f1":              round(float(f1_score(y_true, y_pred, zero_division=0)),        4),
        "specificity":     round(float(specificity), 4),
        "fpr":             round(float(fpr_val),     4),
        "fnr":             round(float(fnr_val),     4),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp),
        },
    }


def evaluate_model(
    model_name: str,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    default_threshold: float,
    optimal_threshold: float,
) -> dict:
    """Evaluate at both default (0.50) and optimal threshold."""
    m_default = full_metrics(y_true, y_proba, default_threshold, model_name, "default")
    m_optimal = full_metrics(y_true, y_proba, optimal_threshold, model_name, "optimal")
    return {"default": m_default, "optimal": m_optimal}


# ===========================================================================
# 4. Generalization analysis
# ===========================================================================
def generalization_analysis(
    model_name: str,
    val_metrics: dict,
    test_metrics_optimal: dict,
) -> dict:
    """
    Compare validation vs test on key metrics.

    Returns a dict with per-metric deltas and a plain-English finding.
    """
    keys_independent = ["roc_auc", "pr_auc"]
    keys_dependent   = ["f1", "precision", "recall"]

    deltas = {}
    for k in keys_independent + keys_dependent:
        val_key = f"{k}_optimal" if k in keys_dependent else k
        v_val   = val_metrics.get(val_key, val_metrics.get(k))
        t_val   = test_metrics_optimal[k]
        deltas[k] = {
            "validation": round(float(v_val), 4),
            "test":       round(float(t_val), 4),
            "delta":      round(float(t_val - v_val), 4),
        }

    # Determine overfitting signal
    roc_drop  = abs(deltas["roc_auc"]["delta"])
    f1_drop   = abs(deltas["f1"]["delta"])

    if roc_drop < 0.01 and f1_drop < 0.02:
        finding = "No meaningful generalization degradation detected."
    elif roc_drop < 0.03 and f1_drop < 0.04:
        finding = "Minor generalization gap; acceptable for a production model."
    else:
        finding = (
            f"Moderate generalization gap detected: "
            f"ROC-AUC dropped {deltas['roc_auc']['delta']:.4f}, "
            f"F1 dropped {deltas['f1']['delta']:.4f}. "
            "Consider additional regularization or more training data."
        )

    return {"model": model_name, "deltas": deltas, "finding": finding}


def compare_models(
    lr_test_optimal: dict,
    mlp_test_optimal: dict,
) -> str:
    """
    Determine which model performs better on unseen test data.

    Judgement criteria (in order):
      1. ROC-AUC difference > 0.005 → decisive.
      2. F1 difference > 0.005 → decisive.
      3. Otherwise → no meaningful winner; both models are equivalent on test data.
    """
    roc_diff = abs(mlp_test_optimal["roc_auc"] - lr_test_optimal["roc_auc"])
    f1_diff  = abs(mlp_test_optimal["f1"]      - lr_test_optimal["f1"])

    if roc_diff > 0.005:
        winner = "MLP" if mlp_test_optimal["roc_auc"] > lr_test_optimal["roc_auc"] else "LogisticRegression"
        return (
            f"{winner} leads on ROC-AUC by {roc_diff:.4f} — a meaningful difference. "
            f"LR ROC-AUC={lr_test_optimal['roc_auc']:.4f}, "
            f"MLP ROC-AUC={mlp_test_optimal['roc_auc']:.4f}."
        )
    if f1_diff > 0.005:
        winner = "MLP" if mlp_test_optimal["f1"] > lr_test_optimal["f1"] else "LogisticRegression"
        return (
            f"{winner} leads on F1 by {f1_diff:.4f} — a meaningful difference. "
            f"LR F1={lr_test_optimal['f1']:.4f}, MLP F1={mlp_test_optimal['f1']:.4f}."
        )
    return (
        f"No meaningful winner: ROC-AUC gap={roc_diff:.4f}, F1 gap={f1_diff:.4f} "
        f"(both below the 0.005 decisiveness threshold). "
        f"LR and MLP are equivalent on unseen test data at their optimized thresholds."
    )


# ===========================================================================
# 5. Visualizations
# ===========================================================================
def plot_confusion_matrices(
    y_true: np.ndarray,
    lr_proba: np.ndarray,
    mlp_proba: np.ndarray,
    lr_threshold: float,
    mlp_threshold: float,
) -> Path:
    """Plot 2x2 grid of confusion matrices (default + optimal) for each model."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("ChurnIQ — Confusion Matrices (Test Set)", fontsize=14, fontweight="bold")

    configs = [
        (axes[0, 0], lr_proba,  0.50,          "LR @ 0.50 (default)"),
        (axes[0, 1], lr_proba,  lr_threshold,   f"LR @ {lr_threshold:.2f} (optimal)"),
        (axes[1, 0], mlp_proba, 0.50,           "MLP @ 0.50 (default)"),
        (axes[1, 1], mlp_proba, mlp_threshold,  f"MLP @ {mlp_threshold:.2f} (optimal)"),
    ]

    for ax, proba, thr, title in configs:
        y_pred = (proba >= thr).astype(int)
        cm     = confusion_matrix(y_true, y_pred, labels=[0, 1])
        im     = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["No Churn", "Churn"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["No Churn", "Churn"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                        fontsize=14, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    out_path = FIGURES_DIR / "confusion_matrices_test.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Confusion matrices → {out_path}")
    return out_path


def plot_roc_pr_curves(
    y_true: np.ndarray,
    lr_proba: np.ndarray,
    mlp_proba: np.ndarray,
    lr_threshold: float,
    mlp_threshold: float,
) -> Path:
    """Plot ROC and PR curves for both models on the test set."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("ChurnIQ — ROC & PR Curves (Test Set)", fontsize=14, fontweight="bold")

    # --- ROC Curve ---
    for proba, label, color in (
        (lr_proba,  f"LR  (AUC={roc_auc_score(y_true, lr_proba):.4f})",  "#1f77b4"),
        (mlp_proba, f"MLP (AUC={roc_auc_score(y_true, mlp_proba):.4f})", "#ff7f0e"),
    ):
        fpr_arr, tpr_arr, _ = roc_curve(y_true, proba)
        ax1.plot(fpr_arr, tpr_arr, color=color, lw=2, label=label)
    ax1.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax1.set_xlabel("False Positive Rate"); ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve"); ax1.legend(loc="lower right"); ax1.grid(alpha=0.3)

    # --- PR Curve ---
    for proba, thr, label, color in (
        (lr_proba,  lr_threshold,
         f"LR  (AP={average_precision_score(y_true, lr_proba):.4f})",  "#1f77b4"),
        (mlp_proba, mlp_threshold,
         f"MLP (AP={average_precision_score(y_true, mlp_proba):.4f})", "#ff7f0e"),
    ):
        prec_arr, rec_arr, _ = precision_recall_curve(y_true, proba)
        ax2.plot(rec_arr, prec_arr, color=color, lw=2, label=label)
        # Mark optimal threshold operating point
        y_pred_thr = (proba >= thr).astype(int)
        op_prec = precision_score(y_true, y_pred_thr, zero_division=0)
        op_rec  = recall_score(y_true, y_pred_thr, zero_division=0)
        ax2.scatter(op_rec, op_prec, color=color, s=100, zorder=5,
                    marker="*", label=f"{label.split('(')[0].strip()} opt. thr={thr:.2f}")

    # Baseline (random)
    base_rate = y_true.mean()
    ax2.axhline(base_rate, color="k", linestyle="--", lw=1,
                label=f"Random (base rate={base_rate:.3f})")
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve"); ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out_path = FIGURES_DIR / "roc_pr_curves_test.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] ROC/PR curves      → {out_path}")
    return out_path


# ===========================================================================
# 6. Save artifacts
# ===========================================================================
def save_final_artifacts(
    all_metrics: list[dict],
    ids_test: np.ndarray,
    y_true: np.ndarray,
    lr_proba: np.ndarray,
    mlp_proba: np.ndarray,
    lr_optimal_threshold: float,
    mlp_optimal_threshold: float,
    gen_lr: dict,
    gen_mlp: dict,
    model_comparison_verdict: str,
) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- final_test_metrics.json ---
    final_metrics_payload = {
        "note": (
            "Final unbiased evaluation on quarantined test partition. "
            "Models and thresholds were NOT modified based on test results."
        ),
        "results": all_metrics,
        "generalization_analysis": {
            "LogisticRegression": gen_lr,
            "MLP":                gen_mlp,
        },
        "model_comparison_verdict": model_comparison_verdict,
    }
    metrics_path = REPORTS_DIR / "final_test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(final_metrics_payload, f, indent=2)
    print(f"[OK] Final test metrics → {metrics_path}")

    # --- final_test_predictions.parquet ---
    preds_df = pd.DataFrame(
        {
            "customerID":                ids_test,
            "actual_churn":              y_true,
            "lr_churn_probability":      lr_proba,
            "lr_pred_default_0.50":      (lr_proba  >= 0.50).astype(int),
            "lr_pred_optimal":           (lr_proba  >= lr_optimal_threshold).astype(int),
            "mlp_churn_probability":     mlp_proba,
            "mlp_pred_default_0.50":     (mlp_proba >= 0.50).astype(int),
            "mlp_pred_optimal":          (mlp_proba >= mlp_optimal_threshold).astype(int),
        }
    )
    preds_path = REPORTS_DIR / "final_test_predictions.parquet"
    preds_df.to_parquet(preds_path, index=False)
    print(f"[OK] Test predictions   → {preds_path}")

    # --- model_comparison.csv ---
    rows = []
    for m in all_metrics:
        rows.append(
            {
                "model":           m["model"],
                "threshold_label": m["threshold_label"],
                "threshold":       m["threshold"],
                "roc_auc":         m["roc_auc"],
                "pr_auc":          m["pr_auc"],
                "accuracy":        m["accuracy"],
                "precision":       m["precision"],
                "recall":          m["recall"],
                "f1":              m["f1"],
                "specificity":     m["specificity"],
                "fpr":             m["fpr"],
                "fnr":             m["fnr"],
                "tn":              m["confusion_matrix"]["tn"],
                "fp":              m["confusion_matrix"]["fp"],
                "fn":              m["confusion_matrix"]["fn"],
                "tp":              m["confusion_matrix"]["tp"],
            }
        )
    comp_df = pd.DataFrame(rows)
    comp_path = REPORTS_DIR / "model_comparison.csv"
    comp_df.to_csv(comp_path, index=False)
    print(f"[OK] Model comparison   → {comp_path}")


# ===========================================================================
# 7. Verification
# ===========================================================================
def verify_final_evaluation(
    y_true: np.ndarray,
    lr_proba: np.ndarray,
    mlp_proba: np.ndarray,
    all_metrics: list[dict],
) -> None:
    """Comprehensive correctness checks."""
    print("\n[Verification]")

    n = len(y_true)

    # 1. Probabilities finite and in [0,1]
    for name, proba in (("LR", lr_proba), ("MLP", mlp_proba)):
        assert np.all(np.isfinite(proba)), f"{name}: non-finite probabilities!"
        assert np.all((proba >= 0) & (proba <= 1)), f"{name}: probabilities outside [0,1]!"
    print("  [OK] All probabilities finite and within [0, 1].")

    # 2. Confusion matrices sum to test sample count
    for m in all_metrics:
        cm = m["confusion_matrix"]
        total = cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"]
        assert total == n, (
            f"{m['model']} @ {m['threshold_label']}: CM sum={total} != n={n}"
        )
    print(f"  [OK] All confusion matrices sum to n={n}.")

    # 3. Reproduce metrics from saved predictions parquet
    preds_df = pd.read_parquet(REPORTS_DIR / "final_test_predictions.parquet")
    repro_lr_roc  = roc_auc_score(preds_df["actual_churn"], preds_df["lr_churn_probability"])
    repro_mlp_roc = roc_auc_score(preds_df["actual_churn"], preds_df["mlp_churn_probability"])
    # Find saved optimal-threshold metrics
    lr_opt_m  = next(m for m in all_metrics
                     if m["model"] == "LogisticRegression" and m["threshold_label"] == "optimal")
    mlp_opt_m = next(m for m in all_metrics
                     if m["model"] == "MLP" and m["threshold_label"] == "optimal")
    assert abs(repro_lr_roc  - lr_opt_m["roc_auc"])  < 1e-4, "LR ROC-AUC mismatch on reproduction!"
    assert abs(repro_mlp_roc - mlp_opt_m["roc_auc"]) < 1e-4, "MLP ROC-AUC mismatch on reproduction!"
    print("  [OK] Metrics reproduced from saved predictions parquet.")

    # 4. All output files readable
    for path in (
        REPORTS_DIR / "final_test_metrics.json",
        REPORTS_DIR / "final_test_predictions.parquet",
        REPORTS_DIR / "model_comparison.csv",
    ):
        assert path.exists(), f"Missing output: {path}"
        if path.suffix == ".json":
            with open(path) as f: json.load(f)
        elif path.suffix == ".parquet":
            pd.read_parquet(path)
        elif path.suffix == ".csv":
            pd.read_csv(path)
    print("  [OK] All output artifacts reopened successfully.")

    # 5. Confirm figures exist
    for fig_name in ("confusion_matrices_test.png", "roc_pr_curves_test.png"):
        assert (FIGURES_DIR / fig_name).exists(), f"Missing figure: {fig_name}"
    print("  [OK] Figure files present.")

    print("[OK] All verifications passed.\n")


# ===========================================================================
# 8. Print final report
# ===========================================================================
def print_final_report(
    all_metrics: list[dict],
    gen_lr: dict,
    gen_mlp: dict,
    model_comparison_verdict: str,
    lr_optimal_threshold: float,
    mlp_optimal_threshold: float,
) -> None:

    def block(model_name, metrics_list):
        default_m = next(m for m in metrics_list
                         if m["model"] == model_name and m["threshold_label"] == "default")
        optimal_m = next(m for m in metrics_list
                         if m["model"] == model_name and m["threshold_label"] == "optimal")
        thr_label = (
            f"{optimal_m['threshold']:.2f} (val-optimised)"
        )

        print(f"\n  ── {model_name} ──")
        print(f"  {'Metric':<14} {'Default 0.50':>13} {f'Optimal {thr_label}':>22}")
        print(f"  {'-'*52}")
        for key in ("roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1",
                    "specificity", "fpr", "fnr"):
            label = key.upper().replace("_", "-")
            d = default_m[key]
            o = optimal_m[key]
            delta = o - d
            sign  = "+" if delta >= 0 else ""
            print(f"  {label:<14} {d:>13.4f} {o:>13.4f}   {sign}{delta:+.4f}")
        cm_d = default_m["confusion_matrix"]
        cm_o = optimal_m["confusion_matrix"]
        print(f"  {'CM Default':<14} TN={cm_d['tn']:4d} FP={cm_d['fp']:4d} FN={cm_d['fn']:4d} TP={cm_d['tp']:4d}")
        print(f"  {'CM Optimal':<14} TN={cm_o['tn']:4d} FP={cm_o['fp']:4d} FN={cm_o['fn']:4d} TP={cm_o['tp']:4d}")

    print("\n" + "=" * 70)
    print("  MODULE 5  —  Final Test Evaluation (Unseen Test Partition)")
    print("=" * 70)
    block("LogisticRegression", all_metrics)
    block("MLP", all_metrics)

    print("\n" + "─" * 70)
    print("  GENERALIZATION ANALYSIS  (Validation → Test)")
    print("─" * 70)
    for gen in (gen_lr, gen_mlp):
        print(f"\n  {gen['model']}")
        for k, v in gen["deltas"].items():
            d = v["delta"]
            sign = "+" if d >= 0 else ""
            print(f"    {k:<12}  val={v['validation']:.4f}  test={v['test']:.4f}  "
                  f"Δ={sign}{d:+.4f}")
        print(f"  → {gen['finding']}")

    print("\n" + "─" * 70)
    print("  MODEL COMPARISON VERDICT")
    print("─" * 70)
    print(f"\n  {model_comparison_verdict}")
    print("\n" + "=" * 70 + "\n")


# ===========================================================================
# 9. Orchestrator
# ===========================================================================
def run_final_evaluation() -> None:
    """
    End-to-end final evaluation pipeline.

    Sequence:
      1. Load threshold metadata (read thresholds — do not modify).
      2. Load test partition (once, read-only).
      3. Load trained models from saved artifacts.
      4. Generate predictions.
      5. Compute metrics at default and optimal thresholds.
      6. Generalization analysis (val vs test).
      7. Model comparison verdict.
      8. Save artifacts.
      9. Generate visualizations.
      10. Verify.
      11. Print report.
    """
    print("\n" + "=" * 70)
    print("  ChurnIQ — Module 5: Final Test Evaluation")
    print("=" * 70)

    # 1. Thresholds
    threshold_meta   = load_threshold_metadata()
    lr_opt_thr       = threshold_meta["models"]["logistic_regression"]["optimal_threshold"]
    mlp_opt_thr      = threshold_meta["models"]["mlp"]["optimal_threshold"]

    # 2. Test data (loaded exactly once)
    X_test, y_test, ids_test = load_test_data()

    # 3. Models
    lr_model  = load_lr_model()
    mlp_model = load_mlp_model()

    # 4. Predictions
    print("\n[INFO] Generating test predictions...")
    lr_proba  = predict_lr(lr_model,  X_test)
    mlp_proba = predict_mlp(mlp_model, X_test)

    # 5. Metrics at default and optimal thresholds
    print("\n[INFO] Computing test metrics...")
    lr_evals  = evaluate_model("LogisticRegression", y_test, lr_proba,
                               0.50, lr_opt_thr)
    mlp_evals = evaluate_model("MLP",                y_test, mlp_proba,
                               0.50, mlp_opt_thr)

    all_metrics = [
        lr_evals["default"],
        lr_evals["optimal"],
        mlp_evals["default"],
        mlp_evals["optimal"],
    ]

    # 6. Generalization analysis
    gen_lr  = generalization_analysis("LogisticRegression",
                                      VALIDATION_METRICS["LogisticRegression"],
                                      lr_evals["optimal"])
    gen_mlp = generalization_analysis("MLP",
                                      VALIDATION_METRICS["MLP"],
                                      mlp_evals["optimal"])

    # 7. Model comparison
    verdict = compare_models(lr_evals["optimal"], mlp_evals["optimal"])

    # 8. Save
    print("\n[INFO] Saving artifacts...")
    save_final_artifacts(
        all_metrics, ids_test, y_test,
        lr_proba, mlp_proba,
        lr_opt_thr, mlp_opt_thr,
        gen_lr, gen_mlp, verdict,
    )

    # 9. Visualizations
    print("\n[INFO] Generating figures...")
    plot_confusion_matrices(y_test, lr_proba, mlp_proba, lr_opt_thr, mlp_opt_thr)
    plot_roc_pr_curves(y_test, lr_proba, mlp_proba, lr_opt_thr, mlp_opt_thr)

    # 10. Verify
    verify_final_evaluation(y_test, lr_proba, mlp_proba, all_metrics)

    # 11. Report
    print_final_report(
        all_metrics, gen_lr, gen_mlp, verdict,
        lr_opt_thr, mlp_opt_thr,
    )

    print("[DONE] Module 5 complete. Awaiting user review before next module.")


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    run_final_evaluation()

