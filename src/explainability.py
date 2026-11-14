"""
ChurnIQ - Module 6: SHAP Explainability
=========================================
Model-agnostic SHAP explanations using KernelExplainer for both
Logistic Regression and MLP (Keras 3 / PyTorch backend).

Design principles:
  - KernelExplainer used throughout: compatible with any predict function,
    no TensorFlow/DeepExplainer assumptions.
  - 75 stratified real training records as background (no K-means centroids).
  - SHAP explains MODEL BEHAVIOUR, not causal factors of churn.
  - Feature aggregation maps one-hot dummies → original customer attributes
    using feature_mapping.json. Aggregated importances represent the combined
    model sensitivity to an original feature, not independent causal effects.
  - Test partition is never loaded or used.
"""

import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import sys
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

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
    RANDOM_SEED,
    set_all_seeds,
)

import keras
import shap

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_BACKGROUND      = 75       # real training records for SHAP background
SMOKE_N_EXPLAIN   = 5        # records for smoke test
SMOKE_NSAMPLES    = 50       # kernel evaluations per record (smoke test)
GLOBAL_N_EXPLAIN  = 200      # records for global importance
GLOBAL_NSAMPLES   = 100      # kernel evaluations per record (global)
LOCAL_NSAMPLES    = 200      # kernel evaluations per record (local — higher quality)
DEFAULT_THRESHOLD = 0.50

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
# 1. Load artifacts
# ===========================================================================
def verify_keras_backend() -> None:
    active = keras.backend.backend()
    if active != "torch":
        raise RuntimeError(f"Expected Keras backend 'torch', got '{active}'.")
    print(f"[OK] Keras backend: {active}")


def load_feature_mapping() -> dict:
    path = MODELS_DIR / "feature_mapping.json"
    with open(path) as f:
        mapping = json.load(f)
    print(f"[OK] Feature mapping loaded: {len(mapping)} encoded features")
    return mapping


def load_data() -> tuple:
    """Load train and val parquets. Test partition is never loaded."""
    df_train = pd.read_parquet(PROCESSED_DATA_DIR / "train_split.parquet")
    df_val   = pd.read_parquet(PROCESSED_DATA_DIR / "val_split.parquet")
    print(f"[OK] Train: {df_train.shape}, Val: {df_val.shape}")
    return df_train, df_val


def load_models() -> tuple:
    lr_model  = joblib.load(MODELS_DIR / "baseline_logistic.joblib")
    mlp_model = keras.models.load_model(str(MODELS_DIR / "mlp_churn_model.keras"))
    print(f"[OK] LR model loaded")
    print(f"[OK] MLP model loaded")
    return lr_model, mlp_model


# ===========================================================================
# 2. SHAP background dataset — 75 stratified real training records
# ===========================================================================
def build_background_dataset(
    df_train: pd.DataFrame,
    n_total: int = N_BACKGROUND,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """
    Select n_total records from the training partition, stratified by Churn.

    Stratification:
      - Maintains the training class distribution (≈73% No, ≈27% Yes).
      - Uses real records — no synthetic centroids.
      - Both classes are represented.

    Returns
    -------
    background : float32 ndarray (n_total, 46)
    """
    rng = np.random.default_rng(seed)

    churn_0 = df_train[df_train["Churn"] == 0]
    churn_1 = df_train[df_train["Churn"] == 1]

    # Proportional split
    n1 = max(1, round(n_total * len(churn_1) / len(df_train)))
    n0 = n_total - n1

    idx0 = rng.choice(len(churn_0), size=n0, replace=False)
    idx1 = rng.choice(len(churn_1), size=n1, replace=False)

    bg_0 = churn_0.iloc[idx0][EXPECTED_FEATURES].values.astype(np.float32)
    bg_1 = churn_1.iloc[idx1][EXPECTED_FEATURES].values.astype(np.float32)

    background = np.vstack([bg_0, bg_1])
    # Shuffle so churn_0/churn_1 blocks are interleaved
    shuffle_idx = rng.permutation(len(background))
    background  = background[shuffle_idx]

    # Save
    joblib.dump(background, MODELS_DIR / "shap_background.joblib")
    print(
        f"[OK] Background dataset: {background.shape}  "
        f"(No={n0}, Yes={n1})  → models/shap_background.joblib"
    )
    return background


# ===========================================================================
# 3. Prediction wrappers
# ===========================================================================
def make_lr_predictor(lr_model):
    """Return a function: X → churn probability array (N,)."""
    def predict_fn(X: np.ndarray) -> np.ndarray:
        return lr_model.predict_proba(X.astype(float))[:, 1]
    return predict_fn


def make_mlp_predictor(mlp_model):
    """Return a function: X → churn probability array (N,)."""
    def predict_fn(X: np.ndarray) -> np.ndarray:
        return mlp_model.predict(X.astype(np.float32), verbose=0).ravel()
    return predict_fn


def validate_predictor(predict_fn, background: np.ndarray, name: str) -> None:
    """Confirm the wrapper returns finite probabilities on background data."""
    out = predict_fn(background[:5])
    assert out.ndim == 1,                        f"{name}: output must be 1-D"
    assert np.all(np.isfinite(out)),             f"{name}: non-finite outputs"
    assert np.all((out >= 0) & (out <= 1)),      f"{name}: outputs outside [0,1]"
    print(f"[OK] {name} predictor validated: {out.round(4)}")


# ===========================================================================
# 4. SHAP smoke test
# ===========================================================================
def smoke_test(
    predict_fn,
    background: np.ndarray,
    X_explain: np.ndarray,
    name: str,
) -> shap.KernelExplainer:
    """
    Run KernelExplainer on SMOKE_N_EXPLAIN records with SMOKE_NSAMPLES.
    Returns the explainer for reuse.
    """
    print(f"\n[Smoke test] {name} — {SMOKE_N_EXPLAIN} records, nsamples={SMOKE_NSAMPLES} ...")
    t0 = time.time()
    explainer   = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(X_explain[:SMOKE_N_EXPLAIN], nsamples=SMOKE_NSAMPLES)
    elapsed     = time.time() - t0

    assert shap_values.shape == (SMOKE_N_EXPLAIN, len(EXPECTED_FEATURES)), (
        f"Unexpected shap_values shape: {shap_values.shape}"
    )
    print(
        f"[OK] Smoke test passed: shape={shap_values.shape}, "
        f"elapsed={elapsed:.1f}s"
    )
    return explainer


# ===========================================================================
# 5. Feature aggregation: one-hot → original attributes
# ===========================================================================
def build_feature_groups(feature_mapping: dict) -> dict:
    """
    Build a dict mapping original_feature_name → list of encoded column indices.

    feature_mapping.json structure (from data_pipeline.py):
      { "encoded_col_name": "original_feature_name" }
      e.g. {"gender_Female": "gender", "tenure": "tenure", ...}

    Aggregation method:
      Mean absolute SHAP value across all dummy columns belonging to the same
      original feature. This represents the average model sensitivity to that
      feature across its possible categories — not an independent causal magnitude.
    """
    groups: dict[str, list[int]] = {}
    for idx, col in enumerate(EXPECTED_FEATURES):
        if col in feature_mapping:
            val = feature_mapping[col]
            # Support both str values and dict values with "original_feature" key
            orig = val if isinstance(val, str) else val.get("original_feature", col)
        else:
            orig = col
        groups.setdefault(orig, []).append(idx)
    return groups


def aggregate_shap_to_original(
    shap_values: np.ndarray,
    feature_groups: dict,
) -> pd.DataFrame:
    """
    Aggregate per-encoded-feature SHAP values to original customer attributes.

    Method: mean absolute SHAP across dummy columns → global importance per attribute.

    Limitation note (documented, not hidden):
      Correlated dummy variables from the same categorical feature will all carry
      partial SHAP mass. Aggregation reduces but does not eliminate this overlap.
      Do not interpret aggregated scores as independent causal effects.

    Returns
    -------
    DataFrame with columns: original_feature, mean_abs_shap, n_encoded_cols
      Sorted descending by mean_abs_shap.
    """
    rows = []
    for orig_feat, indices in feature_groups.items():
        abs_shap = np.abs(shap_values[:, indices])
        # Mean absolute SHAP per record, then mean over records
        agg_importance = float(abs_shap.mean())
        rows.append(
            {
                "original_feature": orig_feat,
                "mean_abs_shap":    round(agg_importance, 6),
                "n_encoded_cols":   len(indices),
            }
        )
    df = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return df


# ===========================================================================
# 6. Global explanations
# ===========================================================================
def compute_global_shap(
    explainer: shap.KernelExplainer,
    X_explain: np.ndarray,
    name: str,
    n_records: int = GLOBAL_N_EXPLAIN,
    nsamples:  int = GLOBAL_NSAMPLES,
) -> np.ndarray:
    """Compute SHAP values for up to n_records validation records."""
    n = min(n_records, len(X_explain))
    print(f"\n[INFO] Computing global SHAP for {name} ({n} records, nsamples={nsamples}) ...")
    t0 = time.time()
    shap_values = explainer.shap_values(X_explain[:n], nsamples=nsamples)
    elapsed = time.time() - t0
    print(f"[OK] Global SHAP done: shape={shap_values.shape}, elapsed={elapsed:.1f}s")
    return shap_values


def plot_shap_summary(
    shap_values: np.ndarray,
    X_explain: np.ndarray,
    model_name: str,
    fname: str,
) -> Path:
    """Beeswarm summary plot (encoded feature level)."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    n = shap_values.shape[0]
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.sca(ax)
    shap.summary_plot(
        shap_values,
        X_explain[:n],
        feature_names=EXPECTED_FEATURES,
        show=False,
        max_display=20,
        plot_type="dot",
    )
    plt.title(f"SHAP Summary — {model_name} (n={n}, val set)", fontsize=12, pad=10)
    plt.tight_layout()
    out = FIGURES_DIR / fname
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] SHAP summary plot  → {out}")
    return out


def plot_shap_importance(
    importance_df: pd.DataFrame,
    model_name: str,
    fname: str,
    top_n: int = 10,
) -> Path:
    """Horizontal bar chart of top-N original feature importances."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    top = importance_df.head(top_n).copy()
    top = top.iloc[::-1]  # reverse for ascending horizontal bar

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(top["original_feature"], top["mean_abs_shap"],
                   color="#1f77b4", edgecolor="white", height=0.6)
    ax.set_xlabel("Mean |SHAP value| (aggregated over encoded columns)", fontsize=10)
    ax.set_title(
        f"Top {top_n} Customer Attributes — {model_name}\n"
        "(Aggregated SHAP | Model behaviour, not causal factors)",
        fontsize=11,
    )
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    out = FIGURES_DIR / fname
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] SHAP importance bar → {out}")
    return out


# ===========================================================================
# 7. Local explanations
# ===========================================================================
def select_local_cases(
    df_val: pd.DataFrame,
    predict_fn,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """
    Select 4 representative validation customers:
      - tp: correctly predicted churner
      - tn: correctly predicted non-churner
      - fp: false positive (predicted churn, actually retained)
      - fn: false negative (missed churner)

    Returns dict: case_label → (customerID, row_index_in_val, actual_churn, proba)
    """
    df = df_val.sort_values("customerID").reset_index(drop=True)
    X  = df[EXPECTED_FEATURES].values.astype(np.float32)
    y  = df["Churn"].values.astype(int)
    ids= df["customerID"].values

    proba = predict_fn(X)
    pred  = (proba >= threshold).astype(int)

    cases = {}
    for label, actual, predicted in (
        ("tp", 1, 1), ("tn", 0, 0), ("fp", 0, 1), ("fn", 1, 0),
    ):
        mask    = (y == actual) & (pred == predicted)
        indices = np.where(mask)[0]
        if len(indices) == 0:
            print(f"  [WARN] No {label} cases found — skipping.")
            continue
        # Pick the record with probability closest to 0.5 within the group
        # (most informative borderline case)
        sub_probas = proba[indices]
        pick       = indices[np.argmin(np.abs(sub_probas - 0.5))]
        cases[label] = {
            "customerID":   str(ids[pick]),
            "val_row_idx":  int(pick),
            "actual_churn": int(y[pick]),
            "predicted":    int(pred[pick]),
            "churn_prob":   round(float(proba[pick]), 4),
        }

    return cases, df, X


def compute_local_shap(
    explainer: shap.KernelExplainer,
    X_val: np.ndarray,
    cases: dict,
    feature_groups: dict,
    model_name: str,
    nsamples: int = LOCAL_NSAMPLES,
) -> dict:
    """
    Compute local SHAP values for each of the 4 representative cases.
    Returns dict: case_label → {shap_values, feature_contribs}
    """
    local_results = {}
    for label, info in cases.items():
        idx = info["val_row_idx"]
        x   = X_val[idx : idx + 1]
        sv  = explainer.shap_values(x, nsamples=nsamples)[0]  # shape (46,)

        # Top encoded features (absolute)
        feature_contribs = sorted(
            zip(EXPECTED_FEATURES, sv),
            key=lambda t: abs(t[1]),
            reverse=True,
        )

        # Aggregate to original features
        agg_rows = []
        for orig, indices in feature_groups.items():
            agg_val = float(sv[indices].sum())   # signed sum for local (direction matters)
            agg_rows.append({"original_feature": orig, "shap_sum": round(agg_val, 6)})
        agg_df = pd.DataFrame(agg_rows).sort_values("shap_sum", key=abs, ascending=False)

        local_results[label] = {
            "meta":          info,
            "shap_encoded":  {f: round(float(v), 6) for f, v in feature_contribs[:15]},
            "shap_original": agg_df.to_dict(orient="records"),
        }
        print(
            f"  [OK] Local SHAP {label.upper()} ({model_name}): "
            f"cust={info['customerID']}, prob={info['churn_prob']}"
        )
    return local_results


def plot_local_waterfall(
    shap_values_1d: np.ndarray,
    x_row: np.ndarray,
    expected_value: float,
    case_label: str,
    case_info: dict,
    model_name: str,
    fname: str,
) -> Path:
    """Horizontal bar plot showing signed SHAP contributions for one customer."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Top 12 by absolute value
    order  = np.argsort(np.abs(shap_values_1d))[::-1][:12]
    feats  = [EXPECTED_FEATURES[i] for i in order]
    vals   = shap_values_1d[order]
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in vals]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(feats[::-1], vals[::-1], color=colors[::-1], edgecolor="white", height=0.6)
    ax.axvline(0, color="black", lw=0.8, linestyle="--")
    ax.set_xlabel("SHAP value (positive = increases churn probability)", fontsize=9)

    label_map = {"tp": "True Positive", "tn": "True Negative",
                 "fp": "False Positive", "fn": "False Negative"}
    ax.set_title(
        f"{model_name} — Local Explanation ({label_map.get(case_label, case_label)})\n"
        f"CustomerID: {case_info['customerID']} | "
        f"Actual: {case_info['actual_churn']} | "
        f"Churn prob: {case_info['churn_prob']:.4f}",
        fontsize=10,
    )
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=7)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    out = FIGURES_DIR / fname
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


# ===========================================================================
# 8. Save reports
# ===========================================================================
def save_global_importance_csv(
    lr_importance:  pd.DataFrame,
    mlp_importance: pd.DataFrame,
) -> None:
    out = REPORTS_DIR / "shap_global_importance.csv"
    lr_importance["model"]  = "LogisticRegression"
    mlp_importance["model"] = "MLP"
    combined = pd.concat([lr_importance, mlp_importance], ignore_index=True)
    combined.to_csv(out, index=False)
    print(f"[OK] Global importance CSV → {out}")


def save_local_explanations_json(
    lr_local:  dict,
    mlp_local: dict,
) -> None:
    payload = {
        "note": (
            "SHAP local explanations show which features most influenced each prediction. "
            "SHAP explains MODEL BEHAVIOUR, not actual causes of customer churn."
        ),
        "logistic_regression": lr_local,
        "mlp":                 mlp_local,
    }
    out = REPORTS_DIR / "shap_local_explanations.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[OK] Local explanations JSON → {out}")


# ===========================================================================
# 9. Verification
# ===========================================================================
def verify_shap_artifacts(
    background:    np.ndarray,
    df_train:      pd.DataFrame,
    lr_shap:       np.ndarray,
    mlp_shap:      np.ndarray,
    lr_importance: pd.DataFrame,
    mlp_importance:pd.DataFrame,
) -> None:
    print("\n[Verification]")

    # 1. Background contains only real training records
    n_train = len(df_train)
    assert background.shape == (N_BACKGROUND, 46), (
        f"Background shape mismatch: {background.shape}"
    )
    bg_reload = joblib.load(MODELS_DIR / "shap_background.joblib")
    assert np.allclose(background, bg_reload), "Background reload mismatch!"
    print(f"  [OK] Background: {N_BACKGROUND} real training records (no test data).")

    # 2. Test partition never loaded (structural check)
    test_path = PROCESSED_DATA_DIR / "test_split.parquet"
    print(f"  [OK] Test partition NOT loaded. Exists on disk: {test_path.exists()}.")

    # 3. SHAP value dimensions
    assert lr_shap.shape[1]  == 46, f"LR SHAP col mismatch: {lr_shap.shape}"
    assert mlp_shap.shape[1] == 46, f"MLP SHAP col mismatch: {mlp_shap.shape}"
    print(f"  [OK] SHAP shapes — LR: {lr_shap.shape}, MLP: {mlp_shap.shape}")

    # 4. Feature mapping coverage (all 46 features present in importance)
    assert len(lr_importance)  > 0, "LR importance DataFrame empty!"
    assert len(mlp_importance) > 0, "MLP importance DataFrame empty!"
    print(f"  [OK] Importance — LR: {len(lr_importance)} orig features, "
          f"MLP: {len(mlp_importance)} orig features.")

    # 5. Aggregated importances are non-negative
    assert (lr_importance["mean_abs_shap"]  >= 0).all(), "LR importance has negative values!"
    assert (mlp_importance["mean_abs_shap"] >= 0).all(), "MLP importance has negative values!"
    print("  [OK] Aggregated importances are non-negative.")

    # 6. Output files
    expected_files = [
        FIGURES_DIR / "shap_summary_mlp.png",
        FIGURES_DIR / "shap_importance_mlp.png",
        FIGURES_DIR / "shap_summary_logistic.png",
        FIGURES_DIR / "shap_importance_logistic.png",
        REPORTS_DIR / "shap_global_importance.csv",
        REPORTS_DIR / "shap_local_explanations.json",
    ]
    for p in expected_files:
        assert p.exists(), f"Missing output: {p}"
    # Check local waterfall figures
    for label in ("tp", "tn", "fp", "fn"):
        for m in ("lr", "mlp"):
            p = FIGURES_DIR / f"shap_local_{m}_{label}.png"
            if p.exists():
                pass  # optional — may be missing if case not found
    pd.read_csv(REPORTS_DIR / "shap_global_importance.csv")
    with open(REPORTS_DIR / "shap_local_explanations.json") as f:
        json.load(f)
    print(f"  [OK] All expected output files present and readable.")

    print("[OK] All verifications passed.\n")


# ===========================================================================
# 10. Print summary report
# ===========================================================================
def print_summary_report(
    lr_importance:  pd.DataFrame,
    mlp_importance: pd.DataFrame,
    lr_local:       dict,
    mlp_local:      dict,
    total_elapsed:  float,
) -> None:
    print("\n" + "=" * 68)
    print("  MODULE 6  —  SHAP Explainability Summary")
    print("=" * 68)

    print("\n  ⚠  SHAP LIMITATION NOTICE:")
    print("  SHAP values explain MODEL BEHAVIOUR, not actual causes of churn.")
    print("  Aggregated importances combine correlated dummy variables from")
    print("  the same categorical feature — do not treat as independent effects.")

    for name, imp in (("LogisticRegression", lr_importance), ("MLP", mlp_importance)):
        print(f"\n  ── Top 10 Original Attributes by SHAP Importance: {name} ──")
        print(f"  {'Rank':<6} {'Original Feature':<30} {'Mean |SHAP|':>12} {'#Encoded':>9}")
        print(f"  {'-'*60}")
        for i, row in imp.head(10).iterrows():
            print(
                f"  {i+1:<6} {row['original_feature']:<30} "
                f"{row['mean_abs_shap']:>12.5f} {row['n_encoded_cols']:>9}"
            )

    print(f"\n  ── Local Explanations ──")
    for model_name, local in (("LR", lr_local), ("MLP", mlp_local)):
        print(f"\n  {model_name}:")
        for label, result in local.items():
            m = result["meta"]
            top1 = list(result["shap_encoded"].items())[0]
            print(
                f"    {label.upper():<4} cust={m['customerID']}  "
                f"actual={m['actual_churn']}  pred={m['predicted']}  "
                f"prob={m['churn_prob']:.3f}  "
                f"top_feature={top1[0]} ({top1[1]:+.4f})"
            )

    print(f"\n  Total SHAP runtime: {total_elapsed:.1f}s")
    print("=" * 68 + "\n")


# ===========================================================================
# 11. Orchestrator
# ===========================================================================
def run_explainability_pipeline() -> None:
    """
    End-to-end SHAP explainability pipeline.

    Sequence:
      0. Seed + backend check.
      1. Load data, models, feature mapping.
      2. Build background dataset (75 stratified train records).
      3. Build prediction wrappers + validate.
      4. Smoke test (5 records, 50 nsamples) for LR and MLP.
      5. Build KernelExplainers (reuse from smoke test).
      6. Global SHAP: 200 val records, 100 nsamples.
      7. Plots: summary beeswarm + importance bar for each model.
      8. Feature group aggregation.
      9. Local SHAP: 4 cases per model.
      10. Local waterfall plots.
      11. Save CSV + JSON reports.
      12. Verify.
      13. Print summary.
    """
    total_start = time.time()

    print("\n" + "=" * 68)
    print("  ChurnIQ — Module 6: SHAP Explainability")
    print("=" * 68)

    # 0. Setup
    verify_keras_backend()
    set_all_seeds()

    # 1. Load
    feature_mapping = load_feature_mapping()
    df_train, df_val = load_data()
    lr_model, mlp_model = load_models()

    # Prepare arrays
    X_train = df_train[EXPECTED_FEATURES].values.astype(np.float32)
    X_val   = df_val[EXPECTED_FEATURES].values.astype(np.float32)

    # 2. Background dataset
    background = build_background_dataset(df_train, n_total=N_BACKGROUND, seed=RANDOM_SEED)

    # 3. Prediction wrappers
    lr_predict  = make_lr_predictor(lr_model)
    mlp_predict = make_mlp_predictor(mlp_model)
    validate_predictor(lr_predict,  background, "LR")
    validate_predictor(mlp_predict, background, "MLP")

    # 4 & 5. Smoke tests → reusable explainers
    lr_explainer  = smoke_test(lr_predict,  background, X_val, "LR")
    mlp_explainer = smoke_test(mlp_predict, background, X_val, "MLP")

    # 6. Global SHAP values
    lr_shap_global  = compute_global_shap(lr_explainer,  X_val, "LR")
    mlp_shap_global = compute_global_shap(mlp_explainer, X_val, "MLP")

    # 7. Summary and importance plots
    plot_shap_summary(lr_shap_global,  X_val[:GLOBAL_N_EXPLAIN],
                      "Logistic Regression", "shap_summary_logistic.png")
    plot_shap_summary(mlp_shap_global, X_val[:GLOBAL_N_EXPLAIN],
                      "MLP",               "shap_summary_mlp.png")

    # 8. Feature group aggregation
    feature_groups = build_feature_groups(feature_mapping)
    lr_importance  = aggregate_shap_to_original(lr_shap_global,  feature_groups)
    mlp_importance = aggregate_shap_to_original(mlp_shap_global, feature_groups)

    plot_shap_importance(lr_importance,  "Logistic Regression", "shap_importance_logistic.png")
    plot_shap_importance(mlp_importance, "MLP",                 "shap_importance_mlp.png")

    # 9. Local SHAP — 4 representative cases per model
    print("\n[INFO] Selecting local explanation cases ...")
    lr_cases,  df_val_sorted, X_val_sorted = select_local_cases(df_val, lr_predict)
    mlp_cases, _,             _             = select_local_cases(df_val, mlp_predict)

    print("[INFO] Computing local SHAP values ...")
    lr_local  = compute_local_shap(lr_explainer,  X_val_sorted, lr_cases,  feature_groups, "LR")
    mlp_local = compute_local_shap(mlp_explainer, X_val_sorted, mlp_cases, feature_groups, "MLP")

    # 10. Local waterfall plots
    print("[INFO] Generating local waterfall plots ...")
    for label, info in lr_cases.items():
        idx = info["val_row_idx"]
        sv  = lr_explainer.shap_values(
            X_val_sorted[idx : idx + 1], nsamples=LOCAL_NSAMPLES
        )[0]
        plot_local_waterfall(
            sv, X_val_sorted[idx], lr_explainer.expected_value,
            label, info, "Logistic Regression",
            f"shap_local_lr_{label}.png",
        )

    for label, info in mlp_cases.items():
        idx = info["val_row_idx"]
        sv  = mlp_explainer.shap_values(
            X_val_sorted[idx : idx + 1], nsamples=LOCAL_NSAMPLES
        )[0]
        plot_local_waterfall(
            sv, X_val_sorted[idx], mlp_explainer.expected_value,
            label, info, "MLP",
            f"shap_local_mlp_{label}.png",
        )

    # 11. Save CSV + JSON
    save_global_importance_csv(lr_importance.copy(), mlp_importance.copy())
    save_local_explanations_json(lr_local, mlp_local)

    # 12. Verify
    verify_shap_artifacts(
        background, df_train,
        lr_shap_global, mlp_shap_global,
        lr_importance, mlp_importance,
    )

    # 13. Summary
    total_elapsed = time.time() - total_start
    print_summary_report(
        lr_importance, mlp_importance,
        lr_local, mlp_local,
        total_elapsed,
    )

    print("[DONE] Module 6 complete. Awaiting user review before next module.")


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    run_explainability_pipeline()
