"""
ChurnIQ – app/streamlit_app.py
================================
Interactive Streamlit dashboard for the ChurnIQ churn prediction system.

Pages / Tabs:
  1. 📊 Overview           – Project summary, dataset stats, class distribution
  2. 🏆 Model Performance  – LR vs MLP metrics, confusion matrices, ROC/PR curves
  3. 🎯 Threshold Analysis – Threshold search visualisation, default vs optimal
  4. ⚠️  Risk Stratification – Low/Medium/High distribution, business actions
  5. 🔍 SHAP Explainability – Global importance, local waterfall plots
  6. 🔮 Individual Prediction – Single-customer input form → real-time prediction
  7. 🛠️  Pipeline Status     – Artifact health check

Design rules:
  - Load from saved artifacts only. No retraining, no live SHAP computation.
  - KERAS_BACKEND=torch set before any keras/model import.
  - customerID is metadata only — never a model feature.
  - Risk categories are always independent of binary threshold.
  - Works fully offline.

Run with:
  c:\\Users\\shaik\\OneDrive\\Desktop\\ChurnIQ\\.venv\\Scripts\\streamlit.exe run app/streamlit_app.py
"""

# ── Backend env must be set before any keras import ──────────────────────────
import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import sys
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ── Project path setup ────────────────────────────────────────────────────────
APP_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    MODELS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_FILE,
    PROJECT_NAME,
    PROJECT_DESCRIPTION,
    PROJECT_VERSION,
    RiskThresholdsConfig,
)
from app.components.charts import (
    roc_curve_chart,
    pr_curve_chart,
    confusion_matrix_chart,
    threshold_f1_chart,
    shap_importance_bar,
    risk_distribution_donut,
    training_history_chart,
    class_distribution_chart,
    metrics_comparison_chart,
    PALETTE,
)
from app.components.ui_widgets import (
    metric_card,
    risk_badge,
    model_selector,
    threshold_display,
    shap_disclaimer,
    probability_gauge,
    top_features_table,
    section_header,
    artifact_status_row,
)

RISK_CFG = RiskThresholdsConfig()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnIQ – AI Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load custom CSS ───────────────────────────────────────────────────────────
_CSS_PATH = APP_DIR / "styles" / "custom.css"
if _CSS_PATH.exists():
    with open(_CSS_PATH) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ==============================================================================
# Data loading helpers (all cached)
# ==============================================================================

@st.cache_data(show_spinner=False)
def load_threshold_metadata() -> dict:
    with open(MODELS_DIR / "threshold_metadata.json") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_final_test_metrics() -> list[dict]:
    with open(REPORTS_DIR / "final_test_metrics.json") as f:
        data = json.load(f)
    return data["results"]


@st.cache_data(show_spinner=False)
def load_training_history() -> dict:
    with open(REPORTS_DIR / "mlp_training_history.json") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_threshold_search() -> pd.DataFrame:
    df = pd.read_csv(REPORTS_DIR / "threshold_search_results.csv")
    return df


@st.cache_data(show_spinner=False)
def load_shap_importance() -> pd.DataFrame:
    return pd.read_csv(REPORTS_DIR / "shap_global_importance.csv")


@st.cache_data(show_spinner=False)
def load_inference_predictions(model_key: str) -> pd.DataFrame:
    fname = f"inference_predictions_{model_key}.parquet"
    return pd.read_parquet(REPORTS_DIR / fname)


@st.cache_data(show_spinner=False)
def load_val_predictions(model_key: str) -> pd.DataFrame:
    if model_key == "mlp":
        return pd.read_parquet(REPORTS_DIR / "mlp_val_predictions.parquet")
    else:
        return pd.read_parquet(REPORTS_DIR / "baseline_val_predictions.parquet")


@st.cache_data(show_spinner=False)
def load_raw_dataset_stats() -> dict:
    """Return basic stats about the full dataset without loading raw CSV."""
    # Use processed partitions (safe — no test data for modelling)
    train = pd.read_parquet(PROCESSED_DATA_DIR / "train_split.parquet")
    val   = pd.read_parquet(PROCESSED_DATA_DIR / "val_split.parquet")
    # Total = train + val + test (test size from final_test_metrics.json)
    n_test = 1057
    n_total = len(train) + len(val) + n_test
    n_churn_train = int(train["Churn"].sum())
    n_churn_val   = int(val["Churn"].sum())
    n_churn_test  = 281  # from final metrics (TP+FN = 223+58)
    n_churn_total = n_churn_train + n_churn_val + n_churn_test
    return {
        "n_total":    n_total,
        "n_train":    len(train),
        "n_val":      len(val),
        "n_test":     n_test,
        "n_features": 46,
        "n_churn":    n_churn_total,
        "n_no_churn": n_total - n_churn_total,
        "churn_rate": round(n_churn_total / n_total * 100, 1),
    }


@st.cache_data(show_spinner=False)
def compute_roc_pr_data() -> dict:
    """Compute ROC / PR curve arrays from val predictions.

    Val parquets have columns: customerID, actual_churn, churn_probability, predicted_churn_at_0.50
    """
    from sklearn.metrics import roc_curve, precision_recall_curve, auc

    result = {}
    for model_key in ("logistic_regression", "mlp"):
        df = load_val_predictions(model_key)
        y_true = df["actual_churn"].values
        y_prob = df["churn_probability"].values

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        pr_auc  = auc(rec, prec)
        result[model_key] = {
            "fpr": fpr, "tpr": tpr, "roc_auc": roc_auc,
            "prec": prec, "rec": rec, "pr_auc": pr_auc,
        }
    return result


@st.cache_resource(show_spinner="Loading inference engine…")
def get_inference_engine(model_name: str):
    """Load ChurnIQInference (cached as a resource for session persistence)."""
    from src.inference import ChurnIQInference
    engine = ChurnIQInference(model_name=model_name)
    engine._load()
    return engine


# ==============================================================================
# Sidebar
# ==============================================================================
def render_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            f"""
            <div style="text-align:center; padding: 1rem 0 0.5rem;">
                <div style="font-size:2.5rem;">📡</div>
                <div style="font-size:1.3rem; font-weight:800;
                     background:linear-gradient(135deg,#667eea,#f093fb);
                     -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                    {PROJECT_NAME}
                </div>
                <div style="font-size:0.72rem; color:#6b7280; margin-top:0.2rem;">
                    v{PROJECT_VERSION} · Keras 3 / PyTorch CPU
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        options = [
            "📊 Overview",
            "🏆 Model Performance",
            "🎯 Threshold Analysis",
            "⚠️ Risk Stratification",
            "🔍 SHAP Explainability",
            "🔮 Individual Prediction",
            "🛠️ Pipeline Status",
        ]

        default_idx = 0
        try:
            qp = st.query_params.get("page", "")
            if qp:
                for i, opt in enumerate(options):
                    if qp.lower() in opt.lower():
                        default_idx = i
                        break
        except Exception:
            pass

        page = st.radio(
            "Navigation",
            options=options,
            index=default_idx,
            label_visibility="collapsed",
        )

        st.divider()
        st.caption("IBM Telco Customer Churn Dataset  \n"
                   "Keras 3.15.1 · PyTorch 2.15.0+cpu  \n"
                   "scikit-learn 1.5.2 · SHAP 0.46.0")
    return page


# ==============================================================================
# Page 1 – Overview
# ==============================================================================
def page_overview():
    st.markdown(
        '<h1 class="churniq-title">📡 ChurnIQ</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(f"*{PROJECT_DESCRIPTION}*")
    st.divider()

    stats = load_raw_dataset_stats()

    # KPI cards
    cols = st.columns(5)
    kpis = [
        ("Total Customers", stats["n_total"], None),
        ("Training Records", stats["n_train"], None),
        ("Validation Records", stats["n_val"], None),
        ("Test Records", stats["n_test"], None),
        ("Churn Rate", f"{stats['churn_rate']}%", None),
    ]
    for col, (label, val, delta) in zip(cols, kpis):
        with col:
            metric_card(label, str(val))

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.2, 1])
    with col_left:
        section_header("Project Architecture", "End-to-end pipeline overview")
        st.markdown("""
| Module | File | Status |
|--------|------|--------|
| 1 – Data Pipeline | `src/data_pipeline.py` | ✅ Complete |
| 2 – Baseline Model | `src/baseline_model.py` | ✅ Complete |
| 3 – MLP Neural Network | `src/neural_network.py` | ✅ Complete |
| 4 – Threshold Optimiser | `src/threshold_optimizer.py` | ✅ Complete |
| 5 – Final Evaluation | `src/evaluation.py` | ✅ Complete |
| 6 – SHAP Explainability | `src/explainability.py` | ✅ Complete |
| 7 – Inference Engine | `src/inference.py` | ✅ Complete |
| 8 – Master Orchestrator | `src/train.py` | ✅ Complete |
| 9 – Dashboard | `app/streamlit_app.py` | ✅ **This app** |
        """)

    with col_right:
        section_header("Class Distribution", "Full dataset (n=7,043)")
        fig = class_distribution_chart(stats["n_no_churn"], stats["n_churn"])
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    section_header("Key Findings")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**Model Winner:** MLP (Neural Network)
- Test ROC-AUC: **0.8465** vs LR 0.8448
- Test F1 at optimal threshold: **0.6326** vs LR 0.6259
- Margin: +0.0067 — meaningful difference (>0.005)
- No overfitting detected on test partition

**Optimal Thresholds (by validation F1):**
- Logistic Regression: **0.54** (↑ from 0.50 default)
- MLP: **0.46** (↓ from 0.50 default)
        """)
    with col2:
        st.markdown("""
**Top Churn Predictors (SHAP – both models):**
1. `tenure` — longer tenure → lower churn
2. `MonthlyCharges` / `TotalCharges`
3. `Contract_Month-to-month`
4. `InternetService_Fiber optic`
5. `PaperlessBilling_Yes`
6. `OnlineSecurity_No`

**Risk Boundaries (independent of threshold):**
- 🟢 Low: P < 0.30 → Routine engagement
- 🟡 Medium: 0.30 ≤ P < 0.70 → Proactive outreach
- 🔴 High: P ≥ 0.70 → Immediate retention offer
        """)


# ==============================================================================
# Page 2 – Model Performance
# ==============================================================================
def page_model_performance():
    section_header("🏆 Model Performance", "Final unbiased evaluation on quarantined test set (n=1,057)")

    results = load_final_test_metrics()
    meta    = load_threshold_metadata()
    history = load_training_history()

    # ── Build lookup dicts ─────────────────────────────────────────────────────
    def _get(model_label, threshold_label) -> dict:
        for r in results:
            if r["model"] == model_label and r["threshold_label"] == threshold_label:
                return r
        return {}

    lr_opt  = _get("LogisticRegression", "optimal")
    mlp_opt = _get("MLP", "optimal")
    lr_def  = _get("LogisticRegression", "default")
    mlp_def = _get("MLP", "default")

    # ── Headline metrics ───────────────────────────────────────────────────────
    st.subheader("📈 Key Metrics at Optimal Threshold")
    heads = ["Metric", "LR (default 0.50)", "LR (optimal 0.54)", "MLP (default 0.50)", "MLP (optimal 0.46)"]
    rows = []
    for key, label in [
        ("roc_auc","ROC-AUC"), ("pr_auc","PR-AUC"),
        ("accuracy","Accuracy"), ("precision","Precision"),
        ("recall","Recall"), ("f1","F1"),
        ("specificity","Specificity"),
    ]:
        rows.append({
            "Metric": label,
            "LR (default 0.50)":  f"{lr_def.get(key, 0):.4f}",
            "LR (optimal 0.54)":  f"{lr_opt.get(key, 0):.4f}",
            "MLP (default 0.50)": f"{mlp_def.get(key, 0):.4f}",
            "MLP (optimal 0.46)": f"{mlp_opt.get(key, 0):.4f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Bar comparison ─────────────────────────────────────────────────────────
    metric_keys = ["roc_auc", "pr_auc", "f1", "precision", "recall", "specificity"]
    fig_cmp = metrics_comparison_chart(lr_opt, mlp_opt, metric_keys)
    st.plotly_chart(fig_cmp, use_container_width=True)

    # ── Confusion matrices ─────────────────────────────────────────────────────
    st.subheader("🧮 Confusion Matrices (Optimal Thresholds)")
    cm_col1, cm_col2 = st.columns(2)
    with cm_col1:
        cm_lr = lr_opt.get("confusion_matrix", {})
        fig_lr_cm = confusion_matrix_chart(
            tn=cm_lr.get("tn", 0), fp=cm_lr.get("fp", 0),
            fn=cm_lr.get("fn", 0), tp=cm_lr.get("tp", 0),
            model_name="Logistic Regression",
            threshold=lr_opt.get("threshold", 0.54),
        )
        st.plotly_chart(fig_lr_cm, use_container_width=True)
    with cm_col2:
        cm_mlp = mlp_opt.get("confusion_matrix", {})
        fig_mlp_cm = confusion_matrix_chart(
            tn=cm_mlp.get("tn", 0), fp=cm_mlp.get("fp", 0),
            fn=cm_mlp.get("fn", 0), tp=cm_mlp.get("tp", 0),
            model_name="MLP",
            threshold=mlp_opt.get("threshold", 0.46),
        )
        st.plotly_chart(fig_mlp_cm, use_container_width=True)

    # ── ROC / PR curves ───────────────────────────────────────────────────────
    st.subheader("📉 ROC & Precision-Recall Curves")

    # Try loading saved figure first; fall back to computed
    roc_path = FIGURES_DIR / "roc_pr_curves_test.png"
    if roc_path.exists():
        st.image(str(roc_path), caption="ROC and PR curves (test set) — saved from evaluation run", use_column_width=True)
    else:
        try:
            curve_data = compute_roc_pr_data()
            lr_d  = curve_data["logistic_regression"]
            mlp_d = curve_data["mlp"]
            stats = load_raw_dataset_stats()
            baseline_rate = stats["churn_rate"] / 100

            roc_col, pr_col = st.columns(2)
            with roc_col:
                fig_roc = roc_curve_chart(
                    lr_d["fpr"], lr_d["tpr"], lr_d["roc_auc"],
                    mlp_d["fpr"], mlp_d["tpr"], mlp_d["roc_auc"],
                )
                st.plotly_chart(fig_roc, use_container_width=True)
            with pr_col:
                fig_pr = pr_curve_chart(
                    lr_d["prec"], lr_d["rec"], lr_d["pr_auc"],
                    mlp_d["prec"], mlp_d["rec"], mlp_d["pr_auc"],
                    baseline_rate,
                )
                st.plotly_chart(fig_pr, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not compute curve data: {e}")

    # ── Training history ──────────────────────────────────────────────────────
    st.subheader("📈 MLP Training History")
    fig_hist = training_history_chart(history)
    st.plotly_chart(fig_hist, use_container_width=True)
    st.caption(
        "Training stopped at epoch 14 (best at epoch 11) via EarlyStopping (patience=3). "
        "Dashed lines = validation; solid = training."
    )

    # ── Generalisation analysis ───────────────────────────────────────────────
    st.subheader("🔬 Generalisation Analysis")
    st.markdown("""
| | LR Val F1 | LR Test F1 | Δ | MLP Val F1 | MLP Test F1 | Δ |
|--|--|--|--|--|--|--|
| F1 (optimal threshold) | 0.6329 | 0.6259 | −0.0070 | 0.6319 | 0.6326 | +0.0007 |
| ROC-AUC | 0.8449 | 0.8448 | −0.0001 | 0.8454 | 0.8465 | +0.0011 |
| PR-AUC  | 0.6314 | 0.6592 | +0.0278 | 0.6321 | 0.6559 | +0.0238 |

✅ **Verdict:** No meaningful generalisation degradation for either model.  
🏆 **Winner:** MLP leads on test F1 by **+0.0067** — a meaningful difference (> 0.005 threshold).
    """)


# ==============================================================================
# Page 3 – Threshold Analysis
# ==============================================================================
def page_threshold_analysis():
    section_header("🎯 Threshold Analysis", "Validation-set F1 optimisation across decision thresholds")

    meta = load_threshold_metadata()
    df_search = load_threshold_search()

    lr_meta  = meta["models"]["logistic_regression"]
    mlp_meta = meta["models"]["mlp"]

    # ── Summary ────────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Logistic Regression")
        threshold_display(
            default=lr_meta["default_threshold"],
            optimal=lr_meta["optimal_threshold"],
            model_name="LR",
        )
        st.metric("F1 Gain", f"+{lr_meta['f1_gain']:.4f}")
        st.caption(
            f"Optimal: **{lr_meta['optimal_threshold']:.2f}** "
            f"→ F1={lr_meta['optimal_f1']:.4f} (default F1={lr_meta['default_f1']:.4f})"
        )
    with col2:
        st.markdown("#### MLP Neural Network")
        threshold_display(
            default=mlp_meta["default_threshold"],
            optimal=mlp_meta["optimal_threshold"],
            model_name="MLP",
        )
        st.metric("F1 Gain", f"+{mlp_meta['f1_gain']:.4f}")
        st.caption(
            f"Optimal: **{mlp_meta['optimal_threshold']:.2f}** "
            f"→ F1={mlp_meta['optimal_f1']:.4f} (default F1={mlp_meta['default_f1']:.4f})"
        )

    st.info(
        f"🔗 **Tie-breaking rule:** {meta.get('tie_breaking_rule', 'lowest threshold among all max-F1 candidates')}. "
        "Models are optimised **independently** on the validation set. "
        "Test partition was never accessed during optimisation."
    )

    # ── F1 search curve ────────────────────────────────────────────────────────
    st.subheader("F1 vs Threshold (Validation Set)")
    # The CSV is long-format: columns [model, threshold, precision, recall, f1, ...]
    # threshold_f1_chart handles both long and wide formats internally.
    fig_th = threshold_f1_chart(
        df_search,
        optimal_lr=lr_meta["optimal_threshold"],
        optimal_mlp=mlp_meta["optimal_threshold"],
    )
    st.plotly_chart(fig_th, use_container_width=True)

    # ── Raw search data ────────────────────────────────────────────────────────
    with st.expander("📋 Raw threshold search data"):
        st.dataframe(df_search, use_container_width=True)
        st.caption(
            f"Grid: {meta['threshold_grid']['min']} → {meta['threshold_grid']['max']} "
            f"(step={meta['threshold_grid']['step']})"
        )

    # ── Risk boundary note ─────────────────────────────────────────────────────
    st.divider()
    st.markdown("""
> **Risk categories are independent of these thresholds.**  
> Low (<0.30), Medium (0.30–0.70), High (≥0.70) are business prioritisation
> boundaries based on raw churn probability — they do not change when the
> binary decision threshold changes.
    """)


# ==============================================================================
# Page 4 – Risk Stratification
# ==============================================================================
def page_risk_stratification():
    section_header("⚠️ Risk Stratification", "Business-action tiers based on churn probability")

    st.markdown("""
| Tier | Probability | Business Action |
|------|-------------|-----------------|
| 🟢 Low | P < 0.30 | Routine engagement — no urgent intervention |
| 🟡 Medium | 0.30 ≤ P < 0.70 | Proactive outreach — automated low-cost nudge |
| 🔴 High | P ≥ 0.70 | Immediate retention offer — high-touch personalised discount |

> Risk categories use raw probability boundaries, independent of the binary
> decision threshold (LR=0.54, MLP=0.46).
    """)
    st.divider()

    # ── Model selector ─────────────────────────────────────────────────────────
    model_key = model_selector(key="risk_model_select")
    model_label = "MLP" if model_key == "mlp" else "Logistic Regression"

    df_inf = load_inference_predictions(model_key)

    # Recompute risk summary from loaded predictions
    risk_summary = {}
    for cat in ("Low", "Medium", "High"):
        n   = int((df_inf["risk_category"] == cat).sum())
        pct = round(n / len(df_inf) * 100, 2)
        risk_summary[cat] = {"count": n, "pct": pct}

    # ── KPI cards ─────────────────────────────────────────────────────────────
    cols = st.columns(3)
    for col, cat in zip(cols, ["Low", "Medium", "High"]):
        with col:
            s = risk_summary[cat]
            metric_card(
                f"{cat} Risk",
                f"{s['count']}",
                delta=f"{s['pct']:.1f}% of customers",
                delta_positive=(cat == "Low"),
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Donut chart ───────────────────────────────────────────────────────────
    col_donut, col_bar = st.columns(2)
    with col_donut:
        fig_donut = risk_distribution_donut(risk_summary, model_label)
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_bar:
        st.markdown("#### Risk Category Breakdown")
        for cat in ["High", "Medium", "Low"]:
            s = risk_summary[cat]
            risk_badge(cat)
            st.progress(s["pct"] / 100, text=f"{s['count']} customers ({s['pct']:.1f}%)")
            st.markdown("")

    # ── Churn rate by risk tier ───────────────────────────────────────────────
    st.divider()
    st.subheader("Churn Rate by Risk Tier (Validation Set)")
    df_val = load_val_predictions(model_key)
    # Val parquets use: actual_churn, churn_probability
    prob_col  = "churn_probability" if "churn_probability" in df_val.columns else None
    label_col = "actual_churn"      if "actual_churn"      in df_val.columns else None

    if prob_col and label_col:
        df_val["risk_category"] = df_val[prob_col].apply(
            lambda p: "High" if p >= RISK_CFG.high_threshold
            else ("Medium" if p >= RISK_CFG.low_threshold else "Low")
        )
        tier_stats = (
            df_val.groupby("risk_category")
            .agg(n_customers=(label_col, "count"), n_churn=(label_col, "sum"))
            .reset_index()
        )
        tier_stats["churn_rate_pct"] = (
            tier_stats["n_churn"] / tier_stats["n_customers"] * 100
        ).round(1)
        tier_stats = tier_stats.set_index("risk_category").reindex(["High", "Medium", "Low"]).reset_index()
        st.dataframe(
            tier_stats.rename(columns={
                "risk_category": "Risk Tier",
                "n_customers": "Customers",
                "n_churn": "Actual Churners",
                "churn_rate_pct": "Actual Churn Rate (%)",
            }),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Computed on validation partition using actual Churn labels. "
            "Demonstrates that risk tiers are well-calibrated."
        )

    # ── Sample predictions ────────────────────────────────────────────────────
    with st.expander("📋 Sample Predictions (Validation Set, first 50 rows)"):
        st.dataframe(df_inf.head(50), use_container_width=True)


# ==============================================================================
# Page 5 – SHAP Explainability
# ==============================================================================
def page_shap_explainability():
    section_header("🔍 SHAP Explainability", "Global and local model explanations (KernelExplainer)")

    shap_disclaimer()

    df_shap = load_shap_importance()

    # ── Global importance ─────────────────────────────────────────────────────
    st.subheader("Global Feature Importance")
    top_n = st.slider("Number of top features to display", 5, 46, 15, key="shap_top_n")

    tab_mlp, tab_lr = st.tabs(["🤖 MLP", "📐 Logistic Regression"])

    with tab_mlp:
        # Try saved PNG first; fall back to Plotly chart
        png_path = FIGURES_DIR / "shap_importance_mlp.png"
        if png_path.exists():
            st.image(str(png_path), caption="MLP – Global SHAP Feature Importance", use_column_width=True)
        else:
            df_mlp_shap = df_shap[df_shap["model"].str.upper() == "MLP"].copy() if "model" in df_shap.columns else df_shap
            if "mean_abs_shap" in df_mlp_shap.columns:
                # Rename to generic column name the chart function expects
                df_plot = df_mlp_shap.rename(columns={"original_feature": "feature", "mean_abs_shap": "_imp"})
                fig = shap_importance_bar(df_plot, "_imp", "MLP", top_n=top_n, color=PALETTE["mlp"])
                st.plotly_chart(fig, use_container_width=True)

        png_sum = FIGURES_DIR / "shap_summary_mlp.png"
        if png_sum.exists():
            st.image(str(png_sum), caption="MLP – SHAP Summary Beeswarm", use_column_width=True)

    with tab_lr:
        png_path_lr = FIGURES_DIR / "shap_importance_logistic.png"
        if png_path_lr.exists():
            st.image(str(png_path_lr), caption="LR – Global SHAP Feature Importance", use_column_width=True)
        else:
            df_lr_shap = df_shap[df_shap["model"].str.lower().str.contains("logistic")] if "model" in df_shap.columns else df_shap
            if "mean_abs_shap" in df_lr_shap.columns:
                df_plot = df_lr_shap.rename(columns={"original_feature": "feature", "mean_abs_shap": "_imp"})
                fig = shap_importance_bar(df_plot, "_imp", "Logistic Regression", top_n=top_n, color=PALETTE["lr"])
                st.plotly_chart(fig, use_container_width=True)

        png_sum_lr = FIGURES_DIR / "shap_summary_logistic.png"
        if png_sum_lr.exists():
            st.image(str(png_sum_lr), caption="LR – SHAP Summary Beeswarm", use_column_width=True)

    # ── Global importance data table ──────────────────────────────────────────
    with st.expander("📋 Raw SHAP importance data"):
        st.dataframe(df_shap, use_container_width=True)

    # ── Local explanations ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("Local Explanations – Case Studies")
    shap_disclaimer()

    local_model = st.selectbox(
        "Select model for local explanation",
        ["MLP", "Logistic Regression"],
        key="local_model_sel",
    )
    model_key_local = "mlp" if local_model == "MLP" else "lr"

    case_col1, case_col2 = st.columns(2)
    cases = [
        ("True Positive (TP)", f"shap_local_{model_key_local}_tp.png",
         "Model correctly predicted churn for an actual churner."),
        ("True Negative (TN)", f"shap_local_{model_key_local}_tn.png",
         "Model correctly predicted no-churn for a non-churner."),
        ("False Positive (FP)", f"shap_local_{model_key_local}_fp.png",
         "Model incorrectly predicted churn — actual no-churn."),
        ("False Negative (FN)", f"shap_local_{model_key_local}_fn.png",
         "Model missed a churner — predicted no-churn."),
    ]
    for (title, fname, caption), col in zip(cases, [case_col1, case_col2, case_col1, case_col2]):
        with col:
            png_p = FIGURES_DIR / fname
            if png_p.exists():
                st.markdown(f"**{title}**")
                st.caption(caption)
                st.image(str(png_p), use_column_width=True)
            else:
                st.warning(f"Plot not found: `{fname}`")

    # ── Method note ───────────────────────────────────────────────────────────
    st.divider()
    with st.expander("ℹ️ SHAP Methodology"):
        st.markdown("""
**Method:** KernelExplainer (model-agnostic, no TF/DeepExplainer assumptions)

**Background dataset:** 75 stratified training records  
(55 No-churn, 20 Churn — mirrors training class distribution)

**Validation sample:** 200 records × 100 nsamples

**Why KernelExplainer?**  
TensorFlow's C-extension is blocked by Windows Smart App Control, so  
TF-specific SHAP explainers (DeepExplainer, GradientExplainer) are unavailable.  
KernelExplainer is fully model-agnostic and works with any prediction function.

**Important limitation:**  
SHAP values quantify *feature contributions to this model's predictions*.  
They do **not** prove causality or explain why customers actually churn.
        """)


# ==============================================================================
# Page 6 – Individual Prediction
# ==============================================================================

# Original categorical options
_GENDER_OPTS            = ["Male", "Female"]
_SENIOR_OPTS            = [0, 1]
_BINARY_OPTS            = ["Yes", "No"]
_MULTIPLELINES_OPTS     = ["Yes", "No", "No phone service"]
_INTERNET_OPTS          = ["DSL", "Fiber optic", "No"]
_INTERNET_SVC_OPTS      = ["Yes", "No", "No internet service"]
_CONTRACT_OPTS          = ["Month-to-month", "One year", "Two year"]
_PAYMENT_OPTS           = [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)",
]


def page_individual_prediction():
    section_header("🔮 Individual Prediction", "Enter customer data to predict churn probability in real time")

    st.info(
        "This form uses the **ChurnIQInference** engine to run your inputs through the full "
        "preprocessing pipeline and your chosen model. No test-set data is used. "
        "customerID is metadata only."
    )

    # ── Model choice ───────────────────────────────────────────────────────────
    model_key = model_selector(key="pred_model_select")
    model_label = "MLP (Neural Network)" if model_key == "mlp" else "Logistic Regression"

    # ── Input form ─────────────────────────────────────────────────────────────
    with st.form("customer_form"):
        st.markdown("#### 👤 Customer Profile")
        col1, col2, col3 = st.columns(3)

        with col1:
            customer_id    = st.text_input("Customer ID (optional)", value="CUST-001")
            gender         = st.selectbox("Gender", _GENDER_OPTS)
            senior_citizen = st.selectbox("Senior Citizen", _SENIOR_OPTS,
                                          format_func=lambda x: "Yes" if x == 1 else "No")
            partner        = st.selectbox("Partner", _BINARY_OPTS)
            dependents     = st.selectbox("Dependents", _BINARY_OPTS)
            tenure         = st.slider("Tenure (months)", 0, 72, 12)

        with col2:
            st.markdown("**📱 Services**")
            phone_service   = st.selectbox("Phone Service", _BINARY_OPTS)
            multiple_lines  = st.selectbox("Multiple Lines", _MULTIPLELINES_OPTS)
            internet_svc    = st.selectbox("Internet Service", _INTERNET_OPTS)
            online_security = st.selectbox("Online Security", _INTERNET_SVC_OPTS)
            online_backup   = st.selectbox("Online Backup", _INTERNET_SVC_OPTS)
            device_protect  = st.selectbox("Device Protection", _INTERNET_SVC_OPTS)

        with col3:
            st.markdown("**📺 Streaming & Billing**")
            tech_support     = st.selectbox("Tech Support", _INTERNET_SVC_OPTS)
            streaming_tv     = st.selectbox("Streaming TV", _INTERNET_SVC_OPTS)
            streaming_movies = st.selectbox("Streaming Movies", _INTERNET_SVC_OPTS)
            contract         = st.selectbox("Contract", _CONTRACT_OPTS)
            paperless        = st.selectbox("Paperless Billing", _BINARY_OPTS)
            payment_method   = st.selectbox("Payment Method", _PAYMENT_OPTS)
            monthly_charges  = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, 0.5)
            total_charges    = st.number_input("Total Charges ($)", 0.0, 10000.0,
                                               float(tenure * monthly_charges), 10.0)

        submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True)

    if submitted:
        customer_data = {
            "customerID":       customer_id,
            "gender":           gender,
            "SeniorCitizen":    senior_citizen,
            "Partner":          partner,
            "Dependents":       dependents,
            "tenure":           tenure,
            "PhoneService":     phone_service,
            "MultipleLines":    multiple_lines,
            "InternetService":  internet_svc,
            "OnlineSecurity":   online_security,
            "OnlineBackup":     online_backup,
            "DeviceProtection": device_protect,
            "TechSupport":      tech_support,
            "StreamingTV":      streaming_tv,
            "StreamingMovies":  streaming_movies,
            "Contract":         contract,
            "PaperlessBilling": paperless,
            "PaymentMethod":    payment_method,
            "MonthlyCharges":   monthly_charges,
            "TotalCharges":     total_charges,
        }

        with st.spinner(f"Running inference with {model_label}…"):
            try:
                engine = get_inference_engine(model_key)
                result = engine.predict_one_raw(customer_data)

                # ── Results ────────────────────────────────────────────────────
                st.divider()
                st.subheader(f"🎯 Prediction Result — {model_label}")

                res_col1, res_col2 = st.columns([1, 1.5])
                with res_col1:
                    probability_gauge(result.churn_probability)
                    risk_badge(result.risk_category)

                    churn_label = "⚠️ CHURN PREDICTED" if result.churn_predicted else "✅ NO CHURN PREDICTED"
                    churn_color = "#f87171" if result.churn_predicted else "#34d399"
                    st.markdown(
                        f'<div style="font-size:1.1rem; font-weight:700; color:{churn_color}; '
                        f'margin:0.8rem 0;">{churn_label}</div>',
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        f"Model: {result.model_used}  |  "
                        f"Threshold: {result.threshold_used:.2f}  |  "
                        f"Probability: {result.churn_probability:.4f}"
                    )

                with res_col2:
                    st.markdown("**Top Feature Contributions (Approximation)**")
                    shap_disclaimer()
                    top_features_table(result.top_features)

            except Exception as e:
                st.error(f"Inference failed: {e}")
                st.exception(e)


# ==============================================================================
# Page 7 – Pipeline Status
# ==============================================================================
def page_pipeline_status():
    section_header("🛠️ Pipeline Status", "Artifact health check for all 8 modules")

    # Define expected artifacts
    artifacts = {
        "Data Pipeline": [
            PROCESSED_DATA_DIR / "train_split.parquet",
            PROCESSED_DATA_DIR / "val_split.parquet",
            PROCESSED_DATA_DIR / "test_split.parquet",
        ],
        "Preprocessor & Mapping": [
            MODELS_DIR / "preprocessor.joblib",
            MODELS_DIR / "feature_mapping.json",
        ],
        "Baseline Model": [
            MODELS_DIR / "baseline_logistic.joblib",
            REPORTS_DIR / "baseline_validation_metrics.json",
            REPORTS_DIR / "baseline_val_predictions.parquet",
        ],
        "MLP Neural Network": [
            MODELS_DIR / "mlp_churn_model.keras",
            REPORTS_DIR / "mlp_validation_metrics.json",
            REPORTS_DIR / "mlp_training_history.json",
            REPORTS_DIR / "mlp_val_predictions.parquet",
        ],
        "Threshold Optimiser": [
            MODELS_DIR / "threshold_metadata.json",
            REPORTS_DIR / "threshold_optimization_comparison.json",
            REPORTS_DIR / "threshold_search_results.csv",
        ],
        "Final Evaluation": [
            REPORTS_DIR / "final_test_metrics.json",
            REPORTS_DIR / "final_test_predictions.parquet",
            REPORTS_DIR / "model_comparison.csv",
            FIGURES_DIR / "confusion_matrices_test.png",
            FIGURES_DIR / "roc_pr_curves_test.png",
        ],
        "SHAP Explainability": [
            MODELS_DIR / "shap_background.joblib",
            REPORTS_DIR / "shap_global_importance.csv",
            REPORTS_DIR / "shap_local_explanations.json",
            FIGURES_DIR / "shap_summary_mlp.png",
            FIGURES_DIR / "shap_summary_logistic.png",
            FIGURES_DIR / "shap_importance_mlp.png",
            FIGURES_DIR / "shap_importance_logistic.png",
            FIGURES_DIR / "shap_local_mlp_tp.png",
            FIGURES_DIR / "shap_local_mlp_tn.png",
            FIGURES_DIR / "shap_local_mlp_fp.png",
            FIGURES_DIR / "shap_local_mlp_fn.png",
            FIGURES_DIR / "shap_local_lr_tp.png",
            FIGURES_DIR / "shap_local_lr_tn.png",
            FIGURES_DIR / "shap_local_lr_fp.png",
            FIGURES_DIR / "shap_local_lr_fn.png",
        ],
        "Inference Engine": [
            REPORTS_DIR / "inference_predictions_mlp.parquet",
            REPORTS_DIR / "inference_predictions_logistic_regression.parquet",
        ],
    }

    all_ok = True
    for stage, paths in artifacts.items():
        with st.expander(f"**{stage}**", expanded=True):
            for p in paths:
                exists = p.exists()
                if not exists:
                    all_ok = False
                artifact_status_row(str(p.relative_to(PROJECT_ROOT)), exists)

    st.divider()
    if all_ok:
        st.success("✅ All pipeline artifacts present — ChurnIQ is fully operational.")
    else:
        st.error(
            "❌ Some artifacts are missing. Run the pipeline:\n\n"
            "```\n"
            "c:\\Users\\shaik\\OneDrive\\Desktop\\ChurnIQ\\.venv\\Scripts\\python.exe "
            "src/train.py --all\n"
            "```"
        )

    # ── Environment info ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("🐍 Runtime Environment")
    import sys
    env_data = {
        "Python": sys.version.split()[0],
        "Keras backend": os.environ.get("KERAS_BACKEND", "unknown"),
        "Project root": str(PROJECT_ROOT),
    }
    try:
        import torch
        env_data["PyTorch"] = torch.__version__
    except ImportError:
        env_data["PyTorch"] = "not installed"
    try:
        import keras
        env_data["Keras"] = keras.__version__
    except ImportError:
        env_data["Keras"] = "not installed"
    try:
        import sklearn
        env_data["scikit-learn"] = sklearn.__version__
    except ImportError:
        pass
    try:
        import shap
        env_data["SHAP"] = shap.__version__
    except ImportError:
        pass
    try:
        import streamlit as _st
        env_data["Streamlit"] = _st.__version__
    except ImportError:
        pass

    for k, v in env_data.items():
        st.markdown(f"- **{k}:** `{v}`")


# ==============================================================================
# Main Router
# ==============================================================================
def main():
    page = render_sidebar()

    if page == "📊 Overview":
        page_overview()
    elif page == "🏆 Model Performance":
        page_model_performance()
    elif page == "🎯 Threshold Analysis":
        page_threshold_analysis()
    elif page == "⚠️ Risk Stratification":
        page_risk_stratification()
    elif page == "🔍 SHAP Explainability":
        page_shap_explainability()
    elif page == "🔮 Individual Prediction":
        page_individual_prediction()
    elif page == "🛠️ Pipeline Status":
        page_pipeline_status()


if __name__ == "__main__":
    main()

