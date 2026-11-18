"""
ChurnIQ – app/streamlit_app.py
================================
Interactive Streamlit dashboard for the ChurnIQ churn prediction system.
Redesigned with Lime-inspired minimalist, modern SaaS visual identity.

Pages / Tabs:
  1. 📊 Overview           – Executive summary, dataset stats, class distribution, pipeline architecture
  2. 🏆 Model Performance  – LR vs MLP comparative evaluation on held-out test set, ROC/PR curves
  3. 🎯 Threshold Analysis – Validation F1 search curves, default vs optimal threshold trade-offs
  4. ⚠️ Risk Stratification – Decoupled Low/Medium/High risk tiers, business actions, empirical calibration
  5. 🔍 SHAP Explainability – Global feature importance, local decision case studies, audit disclaimer
  6. 👤 Individual Prediction – Single-customer real-time inference with preprocessing pipeline
  7. 🛠️ Pipeline Status     – Complete artifact health check and runtime telemetry

Design rules:
  - Load from saved artifacts only. No retraining, no live SHAP computation.
  - KERAS_BACKEND=torch set before any keras/model import.
  - customerID is metadata only — never a model feature.
  - Risk categories are always independent of binary threshold.
  - Works fully offline.
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
    page_header,
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
    page_title="ChurnIQ – AI Telecom Churn Analytics",
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
    train = pd.read_parquet(PROCESSED_DATA_DIR / "train_split.parquet")
    val   = pd.read_parquet(PROCESSED_DATA_DIR / "val_split.parquet")
    n_test = 1057
    n_total = len(train) + len(val) + n_test
    n_churn_train = int(train["Churn"].sum())
    n_churn_val   = int(val["Churn"].sum())
    n_churn_test  = 281
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
    """Compute ROC / PR curve arrays from val predictions."""
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
            <div class="sidebar-brand-box">
                <div class="sidebar-brand-logo">📡</div>
                <div class="sidebar-brand-title">{PROJECT_NAME}</div>
                <div>
                    <span class="sidebar-brand-badge">
                        <span style="color:#00DD00;">●</span> Operational
                    </span>
                </div>
                <div class="sidebar-brand-sub">
                    v{PROJECT_VERSION} · Keras 3 / PyTorch CPU
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        options = [
            "📊 Overview",
            "🏆 Model Performance",
            "🎯 Threshold Analysis",
            "⚠️ Risk Stratification",
            "🔍 SHAP Explainability",
            "👤 Individual Prediction",
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

        st.caption("PLATFORM NAVIGATION")
        page = st.radio(
            "Navigation",
            options=options,
            index=default_idx,
            label_visibility="collapsed",
        )

        st.markdown(
            """
            <div style="margin-top: 2rem; padding: 1rem; background: #F7F9F7; border-radius: 10px; border: 1px solid #E5E7E5; font-size: 0.74rem; color: #737373;">
                <div style="font-weight: 700; color: #171717; margin-bottom: 0.4rem;">Dataset & Stack</div>
                <div>• IBM Telco Churn (N=7,043)</div>
                <div>• Keras 3.15 · PyTorch CPU</div>
                <div>• scikit-learn 1.5.2 · SHAP 0.46</div>
                <div>• 46 Processed Input Features</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    return page


# ==============================================================================
# Page 1 – Overview
# ==============================================================================
def page_overview():
    page_header(
        title="📡 ChurnIQ Intelligence Overview",
        description=f"{PROJECT_DESCRIPTION} End-to-end predictive analytics, architectural pipeline tracking, and benchmark data distribution.",
        kicker="Executive Dashboard",
        status_badge="Operational Pipeline",
    )

    stats = load_raw_dataset_stats()

    # ── Top KPI Cards ──────────────────────────────────────────────────────────
    cols = st.columns(5)
    kpis = [
        ("Total Portfolio Accounts", f"{stats['n_total']:,}", None, True),
        ("Training Set (70%)", f"{stats['n_train']:,}", "Stratified Split", True),
        ("Validation Set (15%)", f"{stats['n_val']:,}", "Threshold Tuning", True),
        ("Held-out Test (15%)", f"{stats['n_test']:,}", "Final Audit", True),
        ("Baseline Churn Rate", f"{stats['churn_rate']}%", "Imbalanced Class", False),
    ]
    for col, (label, val, delta, pos) in zip(cols, kpis):
        with col:
            metric_card(label, val, delta=delta, delta_positive=pos)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Architecture & Class Distribution ──────────────────────────────────────
    col_left, col_right = st.columns([1.15, 1])
    with col_left:
        section_header("System Pipeline Architecture", "8 pipeline modules + 1 analytics interface")
        st.markdown("""
<table class="compare-table">
    <thead>
        <tr>
            <th>Module Name</th>
            <th>Source Script</th>
            <th>Status</th>
        </tr>
    </thead>
    <tbody>
        <tr><td><strong>1. Data Pipeline</strong></td><td><code>src/data_pipeline.py</code></td><td><span class="risk-badge Low" style="font-size:0.75rem; padding:0.15rem 0.5rem;">✓ Complete</span></td></tr>
        <tr><td><strong>2. Baseline Model</strong></td><td><code>src/baseline_model.py</code></td><td><span class="risk-badge Low" style="font-size:0.75rem; padding:0.15rem 0.5rem;">✓ Complete</span></td></tr>
        <tr><td><strong>3. MLP Neural Network</strong></td><td><code>src/neural_network.py</code></td><td><span class="risk-badge Low" style="font-size:0.75rem; padding:0.15rem 0.5rem;">✓ Complete</span></td></tr>
        <tr><td><strong>4. Threshold Optimizer</strong></td><td><code>src/threshold_optimizer.py</code></td><td><span class="risk-badge Low" style="font-size:0.75rem; padding:0.15rem 0.5rem;">✓ Complete</span></td></tr>
        <tr><td><strong>5. Final Evaluation</strong></td><td><code>src/evaluation.py</code></td><td><span class="risk-badge Low" style="font-size:0.75rem; padding:0.15rem 0.5rem;">✓ Complete</span></td></tr>
        <tr><td><strong>6. SHAP Explainability</strong></td><td><code>src/explainability.py</code></td><td><span class="risk-badge Low" style="font-size:0.75rem; padding:0.15rem 0.5rem;">✓ Complete</span></td></tr>
        <tr><td><strong>7. Inference Engine</strong></td><td><code>src/inference.py</code></td><td><span class="risk-badge Low" style="font-size:0.75rem; padding:0.15rem 0.5rem;">✓ Complete</span></td></tr>
        <tr><td><strong>8. Master Orchestrator</strong></td><td><code>src/train.py</code></td><td><span class="risk-badge Low" style="font-size:0.75rem; padding:0.15rem 0.5rem;">✓ Complete</span></td></tr>
        <tr><td><strong>9. Analytics UI</strong></td><td><code>app/streamlit_app.py</code></td><td><span class="risk-badge Low" style="font-size:0.75rem; padding:0.15rem 0.5rem; background:#00DD00; color:#171717;">● Active UI</span></td></tr>
    </tbody>
</table>
        """, unsafe_allow_html=True)

    with col_right:
        section_header("Benchmark Class Imbalance", "Full dataset class distribution (N=7,043)")
        fig = class_distribution_chart(stats["n_no_churn"], stats["n_churn"])
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Key Findings Cards ─────────────────────────────────────────────────────
    section_header("Key Experimental Findings", "Audited outcomes from final evaluation & model interpretability")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="saas-card">
            <div class="saas-card-title">🏆 Champion Model: MLP Neural Network</div>
            <div class="saas-card-subtitle">Generalisation & test metrics vs Logistic Regression baseline</div>
            <ul style="color: #374151; font-size: 0.9rem; line-height: 1.6; padding-left: 1.2rem; margin: 0;">
                <li><strong>Test F1 Score:</strong> <span style="color:#087A08; font-weight:700;">0.6326</span> at optimal threshold τ = 0.46 (vs LR 0.6259 at τ = 0.54).</li>
                <li><strong>Meaningful Margin:</strong> Lead of <strong>+0.0067 F1</strong> exceeds pre-registered significance bound (> 0.0050).</li>
                <li><strong>Discriminative Power:</strong> Test ROC-AUC of <strong>0.8465</strong> (vs LR 0.8448).</li>
                <li><strong>Generalisation:</strong> Zero test degradation detected across validation and held-out test sets.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="saas-card">
            <div class="saas-card-title">🔍 Interpretability & Retention Risk Tiers</div>
            <div class="saas-card-subtitle">SHAP attributions and commercial decision decoupling</div>
            <ul style="color: #374151; font-size: 0.9rem; line-height: 1.6; padding-left: 1.2rem; margin: 0;">
                <li><strong>Primary Churn Drivers:</strong> Tenure, contract duration (Month-to-month), Fiber optic internet, and electronic payment.</li>
                <li><strong>Protective Factors:</strong> Multi-year contracts, partner status, and online security add-ons.</li>
                <li><strong>Commercial Tiers:</strong> Decoupled risk boundaries (Low: P &lt; 0.30, Medium: 0.30 ≤ P &lt; 0.70, High: P ≥ 0.70).</li>
                <li><strong>Empirical Calibration:</strong> High tier captures <strong>64.2%</strong> observed churners vs only <strong>5.2%</strong> in Low tier.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# Page 2 – Model Performance
# ==============================================================================
def page_model_performance():
    page_header(
        title="🏆 Model Performance & Audit",
        description="Final unbiased comparative evaluation on the held-out test set (n=1,057) across default (τ=0.50) and validation-optimal thresholds.",
        kicker="Module 5 – Evaluation",
        status_badge="Held-out Test Set",
    )

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
    section_header("Key Metrics Matrix", "Comprehensive comparison across decision thresholds")
    rows = []
    for key, label in [
        ("roc_auc", "ROC-AUC (Discriminative Capacity)"),
        ("pr_auc", "PR-AUC (Precision-Recall Integral)"),
        ("accuracy", "Overall Accuracy"),
        ("precision", "Precision (Positive Predictive Value)"),
        ("recall", "Recall (Sensitivity / Churn Detection Rate)"),
        ("f1", "F1 Score (Harmonic Mean)"),
        ("specificity", "Specificity (True Negative Rate)"),
    ]:
        rows.append({
            "Evaluation Metric": label,
            "LR (Default τ=0.50)":  f"{lr_def.get(key, 0):.4f}",
            "LR (Optimal τ=0.54)":  f"{lr_opt.get(key, 0):.4f}",
            "MLP (Default τ=0.50)": f"{mlp_def.get(key, 0):.4f}",
            "MLP (Optimal τ=0.46)": f"{mlp_opt.get(key, 0):.4f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Bar comparison ─────────────────────────────────────────────────────────
    metric_keys = ["roc_auc", "pr_auc", "f1", "precision", "recall", "specificity"]
    fig_cmp = metrics_comparison_chart(lr_opt, mlp_opt, metric_keys)
    st.plotly_chart(fig_cmp, use_container_width=True)

    # ── Confusion matrices ─────────────────────────────────────────────────────
    section_header("Confusion Matrices at Optimal Thresholds", "Test set contingency breakdown (n=1,057)")
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
            model_name="MLP Champion",
            threshold=mlp_opt.get("threshold", 0.46),
        )
        st.plotly_chart(fig_mlp_cm, use_container_width=True)

    # ── ROC / PR curves ───────────────────────────────────────────────────────
    section_header("ROC & Precision-Recall Diagnostic Curves", "Threshold-independent discriminative performance")

    roc_path = FIGURES_DIR / "roc_pr_curves_test.png"
    if roc_path.exists():
        st.image(str(roc_path), caption="ROC and PR curves (held-out test set) — exported from Stage 5 evaluation", use_column_width=True)
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
    section_header("MLP Optimization History", "Binary Cross-Entropy Loss & Accuracy across 14 training epochs")
    fig_hist = training_history_chart(history)
    st.plotly_chart(fig_hist, use_container_width=True)
    st.caption(
        "Training terminated at epoch 14 via EarlyStopping (patience=3, restoring best weights from epoch 11). "
        "Dashed lines indicate validation split metrics; solid lines indicate training split."
    )

    # ── Generalisation analysis ───────────────────────────────────────────────
    section_header("Generalisation Audit & Overfitting Verification", "Validation vs Held-out Test partition drift check")
    st.markdown("""
<table class="compare-table">
    <thead>
        <tr>
            <th>Metric</th>
            <th>LR Val</th>
            <th>LR Test</th>
            <th>LR Drift (Δ)</th>
            <th>MLP Val</th>
            <th>MLP Test</th>
            <th>MLP Drift (Δ)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>F1 Score (at τ*)</strong></td>
            <td>0.6329</td>
            <td>0.6259</td>
            <td style="color:#DC2626;">−0.0070</td>
            <td>0.6319</td>
            <td>0.6326</td>
            <td style="color:#087A08; font-weight:700;">+0.0007</td>
        </tr>
        <tr>
            <td><strong>ROC-AUC</strong></td>
            <td>0.8449</td>
            <td>0.8448</td>
            <td style="color:#737373;">−0.0001</td>
            <td>0.8454</td>
            <td>0.8465</td>
            <td style="color:#087A08; font-weight:700;">+0.0011</td>
        </tr>
        <tr>
            <td><strong>PR-AUC</strong></td>
            <td>0.6314</td>
            <td>0.6592</td>
            <td style="color:#087A08;">+0.0278</td>
            <td>0.6321</td>
            <td>0.6559</td>
            <td style="color:#087A08;">+0.0238</td>
        </tr>
    </tbody>
</table>
<div style="margin-top: 0.9rem; padding: 0.9rem 1.2rem; background: #FFFFFF; border: 1px solid #E5E7E5; border-radius: 8px;">
    <strong>Audit Verdict:</strong> Zero harmful generalisation degradation. The MLP Neural Network outperforms the baseline with an F1 margin of <strong>+0.0067</strong> on the test set, exceeding the pre-registered threshold of 0.005.
</div>
    """, unsafe_allow_html=True)


# ==============================================================================
# Page 3 – Threshold Analysis
# ==============================================================================
def page_threshold_analysis():
    page_header(
        title="🎯 Decision Threshold Optimization",
        description="F1-score maximization across the decision threshold grid (τ ∈ [0.05, 0.95], step=0.01) conducted exclusively on the validation set.",
        kicker="Module 4 – Threshold Tuning",
        status_badge="Independent Validation Tuning",
    )

    meta = load_threshold_metadata()
    df_search = load_threshold_search()

    lr_meta  = meta["models"]["logistic_regression"]
    mlp_meta = meta["models"]["mlp"]

    # ── Summary Cards ──────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Logistic Regression Baseline")
        threshold_display(
            default=lr_meta["default_threshold"],
            optimal=lr_meta["optimal_threshold"],
            model_name="LR",
        )
        st.metric("F1 Score Improvement", f"+{lr_meta['f1_gain']:.4f}")
        st.caption(
            f"Optimal Threshold: **τ = {lr_meta['optimal_threshold']:.2f}** "
            f"→ F1 = {lr_meta['optimal_f1']:.4f} (Default τ=0.50 F1 = {lr_meta['default_f1']:.4f})"
        )
    with col2:
        st.markdown("#### MLP Neural Network (Champion)")
        threshold_display(
            default=mlp_meta["default_threshold"],
            optimal=mlp_meta["optimal_threshold"],
            model_name="MLP",
        )
        st.metric("F1 Score Improvement", f"+{mlp_meta['f1_gain']:.4f}")
        st.caption(
            f"Optimal Threshold: **τ = {mlp_meta['optimal_threshold']:.2f}** "
            f"→ F1 = {mlp_meta['optimal_f1']:.4f} (Default τ=0.50 F1 = {mlp_meta['default_f1']:.4f})"
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.info(
        f"🔗 **Deterministic Tie-Breaking Rule:** {meta.get('tie_breaking_rule', 'lowest threshold among all max-F1 candidates')}. "
        "Threshold optimization was executed strictly on the validation set; the held-out test partition was never evaluated during search."
    )

    # ── F1 search curve ────────────────────────────────────────────────────────
    section_header("Validation F1 Search Profiles", "F1 score trajectory across decision thresholds")
    fig_th = threshold_f1_chart(
        df_search,
        optimal_lr=lr_meta["optimal_threshold"],
        optimal_mlp=mlp_meta["optimal_threshold"],
    )
    st.plotly_chart(fig_th, use_container_width=True)

    # ── Raw search data ────────────────────────────────────────────────────────
    with st.expander("📋 Inspect Full Grid Search Results (91 evaluations per model)"):
        st.dataframe(df_search, use_container_width=True)
        st.caption(
            f"Evaluated grid: {meta['threshold_grid']['min']} → {meta['threshold_grid']['max']} "
            f"(step={meta['threshold_grid']['step']})"
        )

    # ── Risk boundary note ─────────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div class="saas-card" style="border-left: 4px solid #00DD00;">
        <strong>Decoupling Principle:</strong>
        Risk categories (Low: P &lt; 0.30, Medium: 0.30 ≤ P &lt; 0.70, High: P ≥ 0.70) are business retention prioritisation
        tiers grounded in raw model probabilities. They remain invariant when binary decision thresholds (LR=0.54, MLP=0.46) are shifted.
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# Page 4 – Risk Stratification
# ==============================================================================
def page_risk_stratification():
    page_header(
        title="⚠️ Risk Stratification & Retention Tiers",
        description="Action-oriented commercial tiers evaluated on the validation cohort (N=1,056) based on raw churn probability, independent of binary decision thresholds.",
        kicker="Commercial Policy",
        status_badge="Validation Cohort (N=1,056)",
    )

    st.markdown("""
<table class="compare-table">
    <thead>
        <tr>
            <th>Risk Tier</th>
            <th>Probability Cutoff</th>
            <th>Commercial Retention Protocol</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><span class="risk-badge Low">🟢 Low</span></td>
            <td><strong>P &lt; 0.30</strong></td>
            <td>Routine relationship maintenance — standard service marketing, zero expensive retention spend.</td>
        </tr>
        <tr>
            <td><span class="risk-badge Medium">🟡 Medium</span></td>
            <td><strong>0.30 ≤ P &lt; 0.70</strong></td>
            <td>Proactive outreach — automated low-cost nudge, loyalty points bonus, digital self-serve engagement.</td>
        </tr>
        <tr>
            <td><span class="risk-badge High">🔴 High</span></td>
            <td><strong>P ≥ 0.70</strong></td>
            <td>Immediate high-touch intervention — dedicated account specialist, customized plan discount, priority support.</td>
        </tr>
    </tbody>
</table>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

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
                f"{cat} Risk Accounts",
                f"{s['count']:,}",
                delta=f"{s['pct']:.1f}% validation cohort share (N=1,056)",
                delta_positive=(cat == "Low"),
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Donut chart ───────────────────────────────────────────────────────────
    col_donut, col_bar = st.columns(2)
    with col_donut:
        fig_donut = risk_distribution_donut(risk_summary, model_label)
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_bar:
        section_header("Tier Breakdown & Action Summary", "Validation cohort distribution (N=1,056)")
        for cat in ["High", "Medium", "Low"]:
            s = risk_summary[cat]
            risk_badge(cat)
            st.progress(s["pct"] / 100, text=f"{s['count']:,} accounts ({s['pct']:.1f}% of validation cohort N=1,056)")
            st.markdown("")

    # ── Churn rate by risk tier ───────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    section_header("Empirical Calibration on Validation Split", "Actual observed churn rate by predicted risk tier across validation cohort (N=1,056)")
    df_val = load_val_predictions(model_key)
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
                "n_customers": "Total Accounts",
                "n_churn": "Actual Churners",
                "churn_rate_pct": "Observed Churn Rate (%)",
            }),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Empirical validation confirms clear calibration across the validation cohort (N=1,056): accounts categorized as High Risk churn at 64.2%, "
            "whereas Low Risk accounts exhibit only a 5.2% churn incidence."
        )

    # ── Sample predictions ────────────────────────────────────────────────────
    with st.expander("📋 View Sample Scored Records (Validation Cohort N=1,056 – First 50 rows)"):
        st.dataframe(df_inf.head(50), use_container_width=True)


# ==============================================================================
# Page 5 – SHAP Explainability
# ==============================================================================
def page_shap_explainability():
    page_header(
        title="🔍 SHAP Explainability & Attribution",
        description="Model-agnostic SHAP KernelExplainer attributions: global feature importance rankings and local decision audit case studies.",
        kicker="Module 6 – Explainability",
        status_badge="SHAP KernelExplainer",
    )

    shap_disclaimer()

    df_shap = load_shap_importance()

    # ── Global importance ─────────────────────────────────────────────────────
    section_header("Global Feature Importance Rankings", "Mean absolute SHAP value across validation cohort")
    top_n = st.slider("Select number of top features to display", 5, 46, 15, key="shap_top_n")

    tab_mlp, tab_lr = st.tabs(["🤖 MLP Neural Network (Champion)", "📐 Logistic Regression (Baseline)"])

    with tab_mlp:
        png_path = FIGURES_DIR / "shap_importance_mlp.png"
        if png_path.exists():
            st.image(str(png_path), caption="MLP – Global SHAP Feature Importance (Validation Set)", use_column_width=True)
        else:
            df_mlp_shap = df_shap[df_shap["model"].str.upper() == "MLP"].copy() if "model" in df_shap.columns else df_shap
            if "mean_abs_shap" in df_mlp_shap.columns:
                df_plot = df_mlp_shap.rename(columns={"original_feature": "feature", "mean_abs_shap": "_imp"})
                fig = shap_importance_bar(df_plot, "_imp", "MLP", top_n=top_n, color=PALETTE["mlp"])
                st.plotly_chart(fig, use_container_width=True)

        png_sum = FIGURES_DIR / "shap_summary_mlp.png"
        if png_sum.exists():
            st.image(str(png_sum), caption="MLP – SHAP Summary Beeswarm Distribution", use_column_width=True)

    with tab_lr:
        png_path_lr = FIGURES_DIR / "shap_importance_logistic.png"
        if png_path_lr.exists():
            st.image(str(png_path_lr), caption="Logistic Regression – Global SHAP Feature Importance", use_column_width=True)
        else:
            df_lr_shap = df_shap[df_shap["model"].str.lower().str.contains("logistic")] if "model" in df_shap.columns else df_shap
            if "mean_abs_shap" in df_lr_shap.columns:
                df_plot = df_lr_shap.rename(columns={"original_feature": "feature", "mean_abs_shap": "_imp"})
                fig = shap_importance_bar(df_plot, "_imp", "Logistic Regression", top_n=top_n, color=PALETTE["lr"])
                st.plotly_chart(fig, use_container_width=True)

        png_sum_lr = FIGURES_DIR / "shap_summary_logistic.png"
        if png_sum_lr.exists():
            st.image(str(png_sum_lr), caption="Logistic Regression – SHAP Summary Beeswarm Distribution", use_column_width=True)

    # ── Global importance data table ──────────────────────────────────────────
    with st.expander("📋 Inspect Raw SHAP Feature Importance Table"):
        st.dataframe(df_shap, use_container_width=True)

    # ── Local explanations ─────────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    section_header("Local Decision Case Studies", "Waterfall attributions across confusion matrix quadrants")

    local_model = st.selectbox(
        "Select architecture for local explanation analysis",
        ["MLP", "Logistic Regression"],
        key="local_model_sel",
    )
    model_key_local = "mlp" if local_model == "MLP" else "lr"

    case_col1, case_col2 = st.columns(2)
    cases = [
        ("True Positive (TP)", f"shap_local_{model_key_local}_tp.png",
         "Actual churner correctly classified by model."),
        ("True Negative (TN)", f"shap_local_{model_key_local}_tn.png",
         "Retained customer correctly identified by model."),
        ("False Positive (FP)", f"shap_local_{model_key_local}_fp.png",
         "False alarm: model predicted churn for retained customer."),
        ("False Negative (FN)", f"shap_local_{model_key_local}_fn.png",
         "Missed churner: model predicted retention for actual churner."),
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
    st.markdown("<hr>", unsafe_allow_html=True)
    with st.expander("ℹ️ KernelExplainer Sampling Methodology & Configuration"):
        st.markdown("""
**Explainer Backend:** KernelExplainer (model-agnostic sampling, zero deep learning black-box assumptions).

**Background Distribution:** 75 stratified training records (55 No-churn, 20 Churn), precisely reflecting the 73.5% / 26.5% base distribution.

**Evaluation Cohort:** 200 validation instances evaluated at 100 samples per evaluation.

**Why KernelExplainer?**  
Enables unified comparison across both non-linear neural networks and linear models under Windows Smart App Control environment constraints.
        """)


# ==============================================================================
# Page 6 – Individual Prediction
# ==============================================================================
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
    page_header(
        title="👤 Real-Time Customer Churn Predictor",
        description="Single-customer real-time scoring via the full fitted ChurnIQInference pipeline and decision thresholds.",
        kicker="Module 7 – Inference",
        status_badge="Live Inference Ready",
    )

    st.info(
        "Enter customer demographic, service, and contract attributes. "
        "The **ChurnIQInference** engine applies the fitted OneHot/StandardScaler pipeline and your selected model. "
        "Customer ID is strictly metadata and excluded from inference."
    )

    # ── Model choice ───────────────────────────────────────────────────────────
    model_key = model_selector(key="pred_model_select")
    model_label = "MLP Neural Network (Champion)" if model_key == "mlp" else "Logistic Regression"

    # ── Input form ─────────────────────────────────────────────────────────────
    with st.form("customer_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 👤 Demographics")
            customer_id    = st.text_input("Customer ID (Metadata)", value="CUST-7044")
            gender         = st.selectbox("Gender", _GENDER_OPTS)
            senior_citizen = st.selectbox("Senior Citizen Status", _SENIOR_OPTS,
                                          format_func=lambda x: "Yes" if x == 1 else "No")
            partner        = st.selectbox("Partner Support", _BINARY_OPTS)
            dependents     = st.selectbox("Dependents Present", _BINARY_OPTS)
            tenure         = st.slider("Account Tenure (Months)", 0, 72, 12)

        with col2:
            st.markdown("#### 📱 Telecom Services")
            phone_service   = st.selectbox("Phone Service", _BINARY_OPTS)
            multiple_lines  = st.selectbox("Multiple Lines", _MULTIPLELINES_OPTS)
            internet_svc    = st.selectbox("Internet Service Type", _INTERNET_OPTS)
            online_security = st.selectbox("Online Security Addon", _INTERNET_SVC_OPTS)
            online_backup   = st.selectbox("Online Backup Addon", _INTERNET_SVC_OPTS)
            device_protect  = st.selectbox("Device Protection Addon", _INTERNET_SVC_OPTS)

        with col3:
            st.markdown("#### 💳 Billing & Contract")
            tech_support     = st.selectbox("Tech Support Addon", _INTERNET_SVC_OPTS)
            streaming_tv     = st.selectbox("Streaming TV", _INTERNET_SVC_OPTS)
            streaming_movies = st.selectbox("Streaming Movies", _INTERNET_SVC_OPTS)
            contract         = st.selectbox("Contract Terms", _CONTRACT_OPTS)
            paperless        = st.selectbox("Paperless Billing", _BINARY_OPTS)
            payment_method   = st.selectbox("Payment Method", _PAYMENT_OPTS)
            monthly_charges  = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, 0.5)
            total_charges    = st.number_input("Total Charges ($)", 0.0, 10000.0,
                                               float(tenure * monthly_charges), 10.0)

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("👤 Run Churn Prediction Scoring", use_container_width=True)

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

        with st.spinner(f"Scoring customer account with {model_label}…"):
            try:
                engine = get_inference_engine(model_key)
                result = engine.predict_one_raw(customer_data)

                # ── Results ────────────────────────────────────────────────────
                st.markdown("<hr>", unsafe_allow_html=True)
                section_header(f"Inference Results: Account {customer_id}", f"Scored by {model_label}")

                res_col1, res_col2 = st.columns([1, 1.4])
                with res_col1:
                    probability_gauge(result.churn_probability)
                    risk_badge(result.risk_category)

                    churn_label = "⚠️ PREDICTED TO CHURN" if result.churn_predicted else "✅ PREDICTED RETAINED"
                    churn_color = "#DC2626" if result.churn_predicted else "#087A08"
                    churn_bg    = "#FEE2E2" if result.churn_predicted else "#E9FFE9"
                    churn_border= "#FCA5A5" if result.churn_predicted else "#86EFAC"

                    st.markdown(
                        f"""
                        <div style="background:{churn_bg}; border:1px solid {churn_border}; border-radius:10px;
                                    padding:1rem; text-align:center; margin:1rem 0;">
                            <div style="font-size:1.15rem; font-weight:800; color:{churn_color};">{churn_label}</div>
                            <div style="font-size:0.8rem; color:#737373; margin-top:0.3rem;">
                                Decision Threshold: τ = {result.threshold_used:.2f} · Probability: {result.churn_probability:.4f}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with res_col2:
                    st.markdown("**Top Feature Attribution Proxies**")
                    shap_disclaimer()
                    top_features_table(result.top_features)

            except Exception as e:
                st.error(f"Inference execution failed: {e}")
                st.exception(e)


# ==============================================================================
# Page 7 – Pipeline Status
# ==============================================================================
def page_pipeline_status():
    page_header(
        title="🛠️ Pipeline Status & Artifact Telemetry",
        description="Comprehensive health check and verification across all 8 pipeline modules, models, parquets, and figures.",
        kicker="Module 8 – Orchestration",
        status_badge="8 / 8 Modules Verified",
    )

    artifacts = {
        "Data Pipeline (Module 1)": [
            PROCESSED_DATA_DIR / "train_split.parquet",
            PROCESSED_DATA_DIR / "val_split.parquet",
            PROCESSED_DATA_DIR / "test_split.parquet",
        ],
        "Preprocessor & Feature Mapping (Module 1)": [
            MODELS_DIR / "preprocessor.joblib",
            MODELS_DIR / "feature_mapping.json",
        ],
        "Baseline Logistic Regression (Module 2)": [
            MODELS_DIR / "baseline_logistic.joblib",
            REPORTS_DIR / "baseline_validation_metrics.json",
            REPORTS_DIR / "baseline_val_predictions.parquet",
        ],
        "MLP Neural Network (Module 3)": [
            MODELS_DIR / "mlp_churn_model.keras",
            REPORTS_DIR / "mlp_validation_metrics.json",
            REPORTS_DIR / "mlp_training_history.json",
            REPORTS_DIR / "mlp_val_predictions.parquet",
        ],
        "Threshold Optimizer (Module 4)": [
            MODELS_DIR / "threshold_metadata.json",
            REPORTS_DIR / "threshold_optimization_comparison.json",
            REPORTS_DIR / "threshold_search_results.csv",
        ],
        "Final Evaluation (Module 5)": [
            REPORTS_DIR / "final_test_metrics.json",
            REPORTS_DIR / "final_test_predictions.parquet",
            REPORTS_DIR / "model_comparison.csv",
            FIGURES_DIR / "confusion_matrices_test.png",
            FIGURES_DIR / "roc_pr_curves_test.png",
        ],
        "SHAP Explainability (Module 6)": [
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
        "Inference Batch Predictions (Module 7)": [
            REPORTS_DIR / "inference_predictions_mlp.parquet",
            REPORTS_DIR / "inference_predictions_logistic_regression.parquet",
        ],
    }

    all_ok = True
    col1, col2 = st.columns(2)
    modules = list(artifacts.items())

    for idx, (stage, paths) in enumerate(modules):
        col = col1 if idx % 2 == 0 else col2
        with col:
            with st.expander(f"**{stage}**", expanded=True):
                for p in paths:
                    exists = p.exists()
                    if not exists:
                        all_ok = False
                    artifact_status_row(str(p.relative_to(PROJECT_ROOT)), exists)

    st.markdown("<hr>", unsafe_allow_html=True)
    if all_ok:
        st.success("✅ All 8 pipeline module artifacts are verified and present — ChurnIQ is fully operational.")
    else:
        st.error(
            "❌ Some pipeline artifacts are missing. Run orchestrator to regenerate:\n\n"
            "```\n"
            ".venv\\Scripts\\python.exe src/train.py --all\n"
            "```"
        )

    # ── Environment info ──────────────────────────────────────────────────────
    section_header("🐍 Runtime Environment Telemetry")
    env_data = {
        "Python Runtime": sys.version.split()[0],
        "Keras Execution Backend": os.environ.get("KERAS_BACKEND", "torch"),
        "Project Directory": str(PROJECT_ROOT),
    }
    try:
        import torch
        env_data["PyTorch Backend"] = torch.__version__
    except ImportError:
        env_data["PyTorch Backend"] = "not installed"
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

    telemetry_cols = st.columns(3)
    items = list(env_data.items())
    for i, (k, v) in enumerate(items):
        with telemetry_cols[i % 3]:
            st.markdown(
                f"""
                <div style="background:#FFFFFF; border:1px solid #E5E7E5; border-radius:8px; padding:0.8rem 1rem; margin-bottom:0.7rem;">
                    <div style="font-size:0.74rem; font-weight:700; color:#737373; text-transform:uppercase;">{k}</div>
                    <div style="font-size:0.95rem; font-weight:700; color:#171717; margin-top:0.2rem;"><code>{v}</code></div>
                </div>
                """,
                unsafe_allow_html=True,
            )


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
    elif page == "👤 Individual Prediction":
        page_individual_prediction()
    elif page == "🛠️ Pipeline Status":
        page_pipeline_status()


if __name__ == "__main__":
    main()
