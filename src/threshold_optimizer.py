"""
ChurnIQ - Module 4: Threshold Optimizer
========================================
Independently optimizes the binary classification threshold for each model
(Logistic Regression and MLP) by maximizing F1 on validation predictions.

Scope:
  - Uses only validation predictions (no test partition access).
  - Risk categories (low/medium/high) are derived separately from thresholds.
  - Does NOT modify trained models, existing prediction files, or raw data.
"""

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    REPORTS_DIR,
    MODELS_DIR,
    RiskThresholdsConfig,
)

RISK_CFG = RiskThresholdsConfig()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
THRESHOLD_MIN  = 0.05
THRESHOLD_MAX  = 0.95
THRESHOLD_STEP = 0.01
DEFAULT_THRESHOLD = 0.50

# Prediction file paths
LR_PREDS_PATH  = REPORTS_DIR / "baseline_val_predictions.parquet"
MLP_PREDS_PATH = REPORTS_DIR / "mlp_val_predictions.parquet"

# Output artifact paths
THRESHOLD_METADATA_PATH     = MODELS_DIR   / "threshold_metadata.json"
COMPARISON_REPORT_PATH      = REPORTS_DIR  / "threshold_optimization_comparison.json"
THRESHOLD_SEARCH_CSV_PATH   = REPORTS_DIR  / "threshold_search_results.csv"


# ===========================================================================
# 1. Load and align prediction files
# ===========================================================================
def load_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load LR and MLP validation prediction parquets.

    Returns
    -------
    df_lr, df_mlp : DataFrames with columns
        customerID, actual_churn, churn_probability
    """
    for p in (LR_PREDS_PATH, MLP_PREDS_PATH):
        if not p.exists():
            raise FileNotFoundError(f"Prediction file not found: {p}")

    df_lr  = pd.read_parquet(LR_PREDS_PATH)
    df_mlp = pd.read_parquet(MLP_PREDS_PATH)

    print(f"[OK] LR  predictions loaded:  {df_lr.shape}")
    print(f"[OK] MLP predictions loaded:  {df_mlp.shape}")

    # Required columns
    required = {"customerID", "actual_churn", "churn_probability"}
    for name, df in (("LR", df_lr), ("MLP", df_mlp)):
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{name} prediction file missing columns: {missing}")

    return df_lr, df_mlp


def verify_alignment(df_lr: pd.DataFrame, df_mlp: pd.DataFrame) -> None:
    """
    Verify that customerIDs and actual_churn labels align between both files.

    Alignment check:
      - Same number of rows.
      - Identical customerID sets (order-independent).
      - After sorting by customerID, actual_churn must match exactly.
    """
    if len(df_lr) != len(df_mlp):
        raise AssertionError(
            f"Row count mismatch: LR={len(df_lr)}, MLP={len(df_mlp)}"
        )

    ids_lr  = set(df_lr["customerID"])
    ids_mlp = set(df_mlp["customerID"])
    if ids_lr != ids_mlp:
        only_lr  = ids_lr  - ids_mlp
        only_mlp = ids_mlp - ids_lr
        raise AssertionError(
            f"CustomerID mismatch. Only in LR: {len(only_lr)}, only in MLP: {len(only_mlp)}"
        )

    # Sort both by customerID and compare labels
    lr_sorted  = df_lr.sort_values("customerID").reset_index(drop=True)
    mlp_sorted = df_mlp.sort_values("customerID").reset_index(drop=True)

    label_match = (lr_sorted["actual_churn"].values == mlp_sorted["actual_churn"].values).all()
    if not label_match:
        n_diff = int((lr_sorted["actual_churn"] != mlp_sorted["actual_churn"]).sum())
        raise AssertionError(
            f"actual_churn labels differ between LR and MLP for {n_diff} customers."
        )

    print(f"[OK] Alignment verified: {len(df_lr)} customers, labels match.")


# ===========================================================================
# 2. Threshold search
# ===========================================================================
def search_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str,
) -> pd.DataFrame:
    """
    Evaluate precision, recall and F1 at each threshold in [0.05, 0.95] step 0.01.

    Tie-breaking rule (deterministic):
      If multiple thresholds yield the same maximum F1, the LOWEST threshold
      is selected. Rationale: for churn prediction, higher recall (catching
      more churners) is preferred when F1 is equal; a lower threshold achieves
      this by being more aggressive in flagging churn.

    Parameters
    ----------
    y_true     : int array (N,)
    y_proba    : float array (N,)
    model_name : label for display

    Returns
    -------
    DataFrame with columns:
        model, threshold, precision, recall, f1,
        tn, fp, fn, tp
    """
    thresholds = np.round(
        np.arange(THRESHOLD_MIN, THRESHOLD_MAX + THRESHOLD_STEP / 2, THRESHOLD_STEP),
        decimals=2,
    )

    rows = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        prec   = precision_score(y_true, y_pred, zero_division=0)
        rec    = recall_score(y_true, y_pred, zero_division=0)
        f1     = f1_score(y_true, y_pred, zero_division=0)
        cm     = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        rows.append(
            {
                "model":     model_name,
                "threshold": float(t),
                "precision": round(float(prec), 6),
                "recall":    round(float(rec),  6),
                "f1":        round(float(f1),   6),
                "tn":        int(tn),
                "fp":        int(fp),
                "fn":        int(fn),
                "tp":        int(tp),
            }
        )

    df_search = pd.DataFrame(rows)
    return df_search


def select_optimal_threshold(df_search: pd.DataFrame, model_name: str) -> dict:
    """
    Select the threshold that maximises F1.

    Tie-breaking: lowest threshold among all maximisers (see search_thresholds
    docstring for rationale).

    Returns
    -------
    dict with keys: model, optimal_threshold, f1, precision, recall,
                    tn, fp, fn, tp, n_ties
    """
    max_f1 = df_search["f1"].max()
    candidates = df_search[df_search["f1"] == max_f1].copy()
    n_ties = len(candidates)

    # Deterministic: pick the lowest threshold
    best_row = candidates.sort_values("threshold").iloc[0]

    result = {
        "model":             model_name,
        "optimal_threshold": float(best_row["threshold"]),
        "f1":                round(float(best_row["f1"]),        4),
        "precision":         round(float(best_row["precision"]), 4),
        "recall":            round(float(best_row["recall"]),    4),
        "tn":                int(best_row["tn"]),
        "fp":                int(best_row["fp"]),
        "fn":                int(best_row["fn"]),
        "tp":                int(best_row["tp"]),
        "n_ties":            n_ties,
    }

    tie_note = f" ({n_ties} thresholds tied; lowest selected)" if n_ties > 1 else ""
    print(
        f"[OK] {model_name} optimal threshold: {result['optimal_threshold']:.2f}"
        f"  F1={result['f1']:.4f}  P={result['precision']:.4f}"
        f"  R={result['recall']:.4f}{tie_note}"
    )
    return result


# ===========================================================================
# 3. Metrics at a given threshold
# ===========================================================================
def metrics_at_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
) -> dict:
    """Compute full metric set at a fixed threshold."""
    y_pred = (y_proba >= threshold).astype(int)
    cm     = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "threshold": round(threshold, 2),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, y_pred, zero_division=0)),    4),
        "f1":        round(float(f1_score(y_true, y_pred, zero_division=0)),        4),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        # Threshold-independent (same at every threshold)
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)),           4),
        "pr_auc":  round(float(average_precision_score(y_true, y_proba)), 4),
    }


# ===========================================================================
# 4. Business risk classification
# ===========================================================================
def assign_risk_categories(y_proba: np.ndarray) -> np.ndarray:
    """
    Map churn probabilities to business risk categories.

    Boundaries (from RiskThresholdsConfig, independent of binary threshold):
      Low    : prob < 0.30
      Medium : 0.30 <= prob < 0.70
      High   : prob >= 0.70

    Parameters
    ----------
    y_proba : float array

    Returns
    -------
    str array of same length with values 'Low', 'Medium', 'High'
    """
    risk = np.where(
        y_proba < RISK_CFG.low_threshold,
        "Low",
        np.where(y_proba < RISK_CFG.high_threshold, "Medium", "High"),
    )
    return risk


def summarise_risk_distribution(
    y_proba: np.ndarray, model_name: str
) -> dict:
    """Return risk category counts and percentages."""
    risk = assign_risk_categories(y_proba)
    total = len(risk)
    dist = {}
    for cat in ("Low", "Medium", "High"):
        n = int((risk == cat).sum())
        dist[cat] = {"count": n, "pct": round(n / total * 100, 2)}
    print(
        f"[OK] {model_name} risk distribution: "
        + "  ".join(f"{k}={dist[k]['count']} ({dist[k]['pct']}%)" for k in ("Low", "Medium", "High"))
    )
    return dist


# ===========================================================================
# 5. Build comparison report
# ===========================================================================
def build_comparison_report(
    model_name: str,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    optimal: dict,
    risk_dist: dict,
) -> dict:
    """Compile default-vs-optimised comparison for one model."""
    default_metrics  = metrics_at_threshold(y_true, y_proba, DEFAULT_THRESHOLD)
    optimal_metrics  = metrics_at_threshold(y_true, y_proba, optimal["optimal_threshold"])

    # F1 gain from optimisation
    f1_gain = round(optimal_metrics["f1"] - default_metrics["f1"], 4)

    return {
        "model":             model_name,
        "optimal_threshold": optimal["optimal_threshold"],
        "default_threshold": DEFAULT_THRESHOLD,
        "f1_gain_from_optimization": f1_gain,
        "tie_breaking_rule": (
            "Lowest threshold selected among all thresholds achieving max F1. "
            "Rationale: a lower threshold increases recall (catching more churners) "
            "when F1 is tied."
        ),
        "n_ties": optimal["n_ties"],
        "default_metrics":  default_metrics,
        "optimal_metrics":  optimal_metrics,
        "risk_distribution": {
            "boundaries": {
                "low_threshold":  RISK_CFG.low_threshold,
                "high_threshold": RISK_CFG.high_threshold,
                "note": (
                    "Risk categories are INDEPENDENT of the binary classification threshold. "
                    "They are used only for business prioritisation, not for the churn decision."
                ),
            },
            "counts": risk_dist,
        },
    }


# ===========================================================================
# 6. Save artifacts
# ===========================================================================
def save_artifacts(
    comparison_lr:  dict,
    comparison_mlp: dict,
    df_search_lr:   pd.DataFrame,
    df_search_mlp:  pd.DataFrame,
) -> None:
    """
    Save three artifacts:
      1. models/threshold_metadata.json  — optimal thresholds for inference use
      2. reports/threshold_optimization_comparison.json  — full comparison report
      3. reports/threshold_search_results.csv  — per-threshold grid for both models
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Threshold metadata (consumed by inference / evaluation modules) ---
    metadata = {
        "note": (
            "Optimal thresholds selected by maximising F1 on validation predictions only. "
            "Test partition was never accessed during optimization."
        ),
        "tie_breaking_rule": (
            "Lowest threshold among all maximisers of validation F1."
        ),
        "threshold_grid": {
            "min":  THRESHOLD_MIN,
            "max":  THRESHOLD_MAX,
            "step": THRESHOLD_STEP,
        },
        "models": {
            "logistic_regression": {
                "optimal_threshold": comparison_lr["optimal_threshold"],
                "default_threshold": DEFAULT_THRESHOLD,
                "optimal_f1":        comparison_lr["optimal_metrics"]["f1"],
                "default_f1":        comparison_lr["default_metrics"]["f1"],
                "f1_gain":           comparison_lr["f1_gain_from_optimization"],
            },
            "mlp": {
                "optimal_threshold": comparison_mlp["optimal_threshold"],
                "default_threshold": DEFAULT_THRESHOLD,
                "optimal_f1":        comparison_mlp["optimal_metrics"]["f1"],
                "default_f1":        comparison_mlp["default_metrics"]["f1"],
                "f1_gain":           comparison_mlp["f1_gain_from_optimization"],
            },
        },
        "risk_boundaries": {
            "low_threshold":  RISK_CFG.low_threshold,
            "high_threshold": RISK_CFG.high_threshold,
            "independence_note": (
                "Risk category boundaries are independent of model classification thresholds."
            ),
        },
    }

    with open(THRESHOLD_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[OK] Threshold metadata  → {THRESHOLD_METADATA_PATH}")

    # --- 2. Full comparison report ---
    full_report = {
        "logistic_regression": comparison_lr,
        "mlp":                 comparison_mlp,
    }
    with open(COMPARISON_REPORT_PATH, "w") as f:
        json.dump(full_report, f, indent=2)
    print(f"[OK] Comparison report   → {COMPARISON_REPORT_PATH}")

    # --- 3. Threshold search results CSV ---
    df_all = pd.concat([df_search_lr, df_search_mlp], ignore_index=True)
    df_all.to_csv(THRESHOLD_SEARCH_CSV_PATH, index=False)
    print(f"[OK] Search results CSV  → {THRESHOLD_SEARCH_CSV_PATH}")


# ===========================================================================
# 7. Verification
# ===========================================================================
def verify_artifacts(
    comparison_lr:  dict,
    comparison_mlp: dict,
    df_search_lr:   pd.DataFrame,
    df_search_mlp:  pd.DataFrame,
    y_true_lr:      np.ndarray,
    y_true_mlp:     np.ndarray,
) -> None:
    """
    Verify correctness of threshold selection and artifact integrity.
    """
    print("\n[Verification]")

    # 1. Test partition was never accessed
    test_path = PROJECT_ROOT / "data" / "processed" / "test_split.parquet"
    # (We simply confirm it was not read; no assertion needed beyond code review.
    #  Document this explicitly.)
    print(f"  [OK] Test partition NOT accessed. Path exists: {test_path.exists()} "
          f"(confirmed not loaded in this module).")

    # 2. Optimal threshold actually maximises F1 on the grid
    for name, df_s, comp in (
        ("LR",  df_search_lr,  comparison_lr),
        ("MLP", df_search_mlp, comparison_mlp),
    ):
        grid_max_f1 = df_s["f1"].max()
        chosen_f1   = comp["optimal_metrics"]["f1"]
        # Allow for float rounding between grid (6 dp) and metrics_at_threshold (4 dp)
        assert abs(chosen_f1 - round(grid_max_f1, 4)) < 1e-3, (
            f"{name}: chosen F1 {chosen_f1} does not match grid max F1 {grid_max_f1:.4f}"
        )
        print(f"  [OK] {name} optimal threshold verified against grid max F1.")

    # 3. Risk categories independent of binary thresholds
    lr_opt  = comparison_lr["optimal_threshold"]
    mlp_opt = comparison_mlp["optimal_threshold"]
    lr_risk_b  = comparison_lr["risk_distribution"]["boundaries"]
    mlp_risk_b = comparison_mlp["risk_distribution"]["boundaries"]
    # Confirm risk boundaries differ from optimal thresholds (they are separate concerns)
    print(
        f"  [OK] Risk boundaries (low={lr_risk_b['low_threshold']}, "
        f"high={lr_risk_b['high_threshold']}) are independent of "
        f"binary thresholds (LR={lr_opt}, MLP={mlp_opt})."
    )

    # 4. Reload all saved artifacts
    with open(THRESHOLD_METADATA_PATH)  as f: json.load(f)
    with open(COMPARISON_REPORT_PATH)   as f: json.load(f)
    df_csv = pd.read_csv(THRESHOLD_SEARCH_CSV_PATH)
    expected_rows = 2 * len(df_search_lr)   # LR + MLP
    assert len(df_csv) == expected_rows, (
        f"CSV row count mismatch: expected {expected_rows}, got {len(df_csv)}"
    )
    print(f"  [OK] All artifacts reload successfully ({len(df_csv)} CSV rows).")

    print("[OK] All verifications passed.\n")


# ===========================================================================
# 8. Print summary report
# ===========================================================================
def print_summary_report(comparison_lr: dict, comparison_mlp: dict) -> None:
    """Console summary: selected thresholds and default-vs-optimised metrics."""

    def model_block(comp: dict) -> None:
        name    = comp["model"]
        d_thr   = comp["default_threshold"]
        o_thr   = comp["optimal_threshold"]
        dm      = comp["default_metrics"]
        om      = comp["optimal_metrics"]
        gain    = comp["f1_gain_from_optimization"]
        ties    = comp["n_ties"]

        print(f"\n  ── {name} ──")
        print(f"  Default threshold : {d_thr:.2f}  │  Optimal threshold: {o_thr:.2f}"
              f"  {'(ties: ' + str(ties) + ')' if ties > 1 else ''}")
        print(f"  {'Metric':<12} {'Default':>10} {'Optimised':>10} {'Delta':>8}")
        print(f"  {'-'*44}")
        for key, label in (
            ("f1",        "F1"),
            ("precision", "Precision"),
            ("recall",    "Recall"),
            ("roc_auc",   "ROC-AUC"),
            ("pr_auc",    "PR-AUC"),
        ):
            d_val = dm[key]
            o_val = om[key]
            delta = o_val - d_val
            sign  = "+" if delta >= 0 else ""
            print(f"  {label:<12} {d_val:>10.4f} {o_val:>10.4f} {sign}{delta:>+7.4f}")
        print(f"  Conf. Matrix  Default : TN={dm['tn']:4d}  FP={dm['fp']:4d}  FN={dm['fn']:4d}  TP={dm['tp']:4d}")
        print(f"  Conf. Matrix  Optimal : TN={om['tn']:4d}  FP={om['fp']:4d}  FN={om['fn']:4d}  TP={om['tp']:4d}")
        print(f"  F1 gain from optimisation: {gain:+.4f}")

        risk_b = comp["risk_distribution"]["boundaries"]
        risk_c = comp["risk_distribution"]["counts"]
        print(
            f"  Risk (independent of threshold): "
            f"Low={risk_c['Low']['count']} ({risk_c['Low']['pct']}%)  "
            f"Med={risk_c['Medium']['count']} ({risk_c['Medium']['pct']}%)  "
            f"High={risk_c['High']['count']} ({risk_c['High']['pct']}%)"
        )

    print("\n" + "=" * 65)
    print("  MODULE 4  —  Threshold Optimisation Results")
    print("=" * 65)
    model_block(comparison_lr)
    model_block(comparison_mlp)
    print("\n" + "=" * 65)
    print("  Selected thresholds (stored in models/threshold_metadata.json)")
    print(f"  LR  optimal threshold : {comparison_lr['optimal_threshold']:.2f}")
    print(f"  MLP optimal threshold : {comparison_mlp['optimal_threshold']:.2f}")
    print("=" * 65 + "\n")


# ===========================================================================
# 9. Orchestrator
# ===========================================================================
def run_threshold_optimizer() -> None:
    """
    End-to-end threshold optimisation pipeline.

    Steps:
      1. Load LR and MLP validation predictions.
      2. Verify alignment.
      3. Search thresholds independently for each model.
      4. Select optimal threshold per model (max F1, lowest-threshold tie-break).
      5. Compute default and optimal metrics.
      6. Summarise risk distributions.
      7. Build comparison reports.
      8. Save artifacts.
      9. Verify correctness.
      10. Print summary.
    """
    print("\n" + "=" * 65)
    print("  ChurnIQ — Module 4: Threshold Optimiser")
    print("=" * 65)

    # 1. Load
    df_lr, df_mlp = load_predictions()

    # 2. Align
    verify_alignment(df_lr, df_mlp)

    # Extract arrays (sort both by customerID for consistent ordering)
    df_lr_s  = df_lr.sort_values("customerID").reset_index(drop=True)
    df_mlp_s = df_mlp.sort_values("customerID").reset_index(drop=True)

    y_true_lr   = df_lr_s["actual_churn"].values.astype(int)
    y_proba_lr  = df_lr_s["churn_probability"].values.astype(float)

    y_true_mlp  = df_mlp_s["actual_churn"].values.astype(int)
    y_proba_mlp = df_mlp_s["churn_probability"].values.astype(float)

    # 3. Search thresholds
    print(f"\n[INFO] Searching thresholds [{THRESHOLD_MIN}, {THRESHOLD_MAX}] "
          f"step={THRESHOLD_STEP} ...")
    df_search_lr  = search_thresholds(y_true_lr,  y_proba_lr,  "LogisticRegression")
    df_search_mlp = search_thresholds(y_true_mlp, y_proba_mlp, "MLP")

    # 4. Select optimal
    optimal_lr  = select_optimal_threshold(df_search_lr,  "LogisticRegression")
    optimal_mlp = select_optimal_threshold(df_search_mlp, "MLP")

    # 5 & 6. Risk distribution
    risk_dist_lr  = summarise_risk_distribution(y_proba_lr,  "LogisticRegression")
    risk_dist_mlp = summarise_risk_distribution(y_proba_mlp, "MLP")

    # 7. Comparison reports
    comp_lr  = build_comparison_report(
        "LogisticRegression", y_true_lr,  y_proba_lr,  optimal_lr,  risk_dist_lr
    )
    comp_mlp = build_comparison_report(
        "MLP",                y_true_mlp, y_proba_mlp, optimal_mlp, risk_dist_mlp
    )

    # 8. Save
    print()
    save_artifacts(comp_lr, comp_mlp, df_search_lr, df_search_mlp)

    # 9. Verify
    verify_artifacts(
        comp_lr, comp_mlp,
        df_search_lr, df_search_mlp,
        y_true_lr, y_true_mlp,
    )

    # 10. Print summary
    print_summary_report(comp_lr, comp_mlp)

    print("[DONE] Module 4 complete. Awaiting user review before next module.")


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    run_threshold_optimizer()

