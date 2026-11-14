"""
ChurnIQ – app/components/charts.py
===================================
Reusable Plotly chart factories for the Streamlit dashboard.

All functions accept pre-loaded data (DataFrames, dicts, lists) and return
Plotly Figure objects ready for `st.plotly_chart(fig, use_container_width=True)`.

No model loading or heavy computation occurs here.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# ── Colour palette ────────────────────────────────────────────────────────────
PALETTE = {
    "lr":        "#667eea",   # Logistic Regression – indigo
    "mlp":       "#f093fb",   # MLP – pink-purple
    "positive":  "#34d399",   # green
    "negative":  "#f87171",   # red
    "warn":      "#fbbf24",   # amber
    "bg":        "#0e1117",
    "grid":      "#2d3148",
    "text":      "#e5e7eb",
    "High":      "#f87171",
    "Medium":    "#fbbf24",
    "Low":       "#34d399",
}

_PLOTLY_TEMPLATE = "plotly_dark"

_LAYOUT_DEFAULTS = dict(
    template=_PLOTLY_TEMPLATE,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=PALETTE["text"], family="Inter, sans-serif"),
    margin=dict(l=40, r=20, t=50, b=40),
)


def _apply_defaults(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(title=dict(text=title, font=dict(size=15)), **_LAYOUT_DEFAULTS)
    fig.update_xaxes(gridcolor=PALETTE["grid"], zerolinecolor=PALETTE["grid"])
    fig.update_yaxes(gridcolor=PALETTE["grid"], zerolinecolor=PALETTE["grid"])
    return fig


# ── 1. ROC Curve ──────────────────────────────────────────────────────────────
def roc_curve_chart(
    fpr_lr: np.ndarray,
    tpr_lr: np.ndarray,
    auc_lr: float,
    fpr_mlp: np.ndarray,
    tpr_mlp: np.ndarray,
    auc_mlp: float,
) -> go.Figure:
    """Side-by-side ROC curves for LR and MLP on a single plot."""
    fig = go.Figure()
    # Diagonal reference
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dash", color="grey", width=1),
        name="Random (AUC=0.50)", showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=fpr_lr, y=tpr_lr, mode="lines",
        line=dict(color=PALETTE["lr"], width=2.5),
        name=f"Logistic Reg. (AUC={auc_lr:.4f})",
    ))
    fig.add_trace(go.Scatter(
        x=fpr_mlp, y=tpr_mlp, mode="lines",
        line=dict(color=PALETTE["mlp"], width=2.5),
        name=f"MLP (AUC={auc_mlp:.4f})",
    ))
    fig.update_xaxes(title="False Positive Rate", range=[0, 1])
    fig.update_yaxes(title="True Positive Rate",  range=[0, 1.02])
    return _apply_defaults(fig, "ROC Curve – Test Set")


# ── 2. PR Curve ───────────────────────────────────────────────────────────────
def pr_curve_chart(
    prec_lr:   np.ndarray,
    rec_lr:    np.ndarray,
    prauc_lr:  float,
    prec_mlp:  np.ndarray,
    rec_mlp:   np.ndarray,
    prauc_mlp: float,
    baseline_rate: float,
) -> go.Figure:
    """Precision-Recall curves for LR and MLP."""
    fig = go.Figure()
    # No-skill line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[baseline_rate, baseline_rate], mode="lines",
        line=dict(dash="dash", color="grey", width=1),
        name=f"No Skill (rate={baseline_rate:.2f})", showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=rec_lr, y=prec_lr, mode="lines",
        line=dict(color=PALETTE["lr"], width=2.5),
        name=f"Logistic Reg. (AP={prauc_lr:.4f})",
    ))
    fig.add_trace(go.Scatter(
        x=rec_mlp, y=prec_mlp, mode="lines",
        line=dict(color=PALETTE["mlp"], width=2.5),
        name=f"MLP (AP={prauc_mlp:.4f})",
    ))
    fig.update_xaxes(title="Recall", range=[0, 1])
    fig.update_yaxes(title="Precision", range=[0, 1.02])
    return _apply_defaults(fig, "Precision-Recall Curve – Test Set")


# ── 3. Confusion Matrix ───────────────────────────────────────────────────────
def confusion_matrix_chart(
    tn: int, fp: int, fn: int, tp: int,
    model_name: str,
    threshold: float,
) -> go.Figure:
    """Annotated confusion-matrix heatmap."""
    z    = [[tn, fp], [fn, tp]]
    text = [[f"TN<br>{tn}", f"FP<br>{fp}"], [f"FN<br>{fn}", f"TP<br>{tp}"]]

    fig = go.Figure(go.Heatmap(
        z=z,
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=18, color="white"),
        colorscale=[[0, "#1e2130"], [0.5, "#3a3f7a"], [1, "#667eea"]],
        showscale=False,
        xgap=3, ygap=3,
    ))
    fig.update_xaxes(
        tickvals=[0, 1], ticktext=["Predicted: No", "Predicted: Yes"],
        title="Predicted",
    )
    fig.update_yaxes(
        tickvals=[0, 1], ticktext=["Actual: No", "Actual: Yes"],
        title="Actual", autorange="reversed",
    )
    return _apply_defaults(
        fig, f"{model_name} – Confusion Matrix (threshold={threshold:.2f})"
    )


# ── 4. Threshold F1 Search ────────────────────────────────────────────────────
def threshold_f1_chart(
    df_search: pd.DataFrame,
    optimal_lr: float,
    optimal_mlp: float,
) -> go.Figure:
    """F1 vs threshold curves for both models with optimal markers.

    Accepts either:
      - wide format with columns  ['threshold', 'f1_lr', 'f1_mlp']
      - long format with columns  ['model', 'threshold', 'f1']  (actual saved format)
    """
    fig = go.Figure()

    # Normalise to wide format
    if "model" in df_search.columns and "f1_lr" not in df_search.columns:
        lr_rows  = df_search[df_search["model"].str.contains("Logistic", case=False)]
        mlp_rows = df_search[df_search["model"].str.contains("MLP",      case=False)]
        df_wide  = lr_rows[["threshold", "f1"]].rename(columns={"f1": "f1_lr"}).merge(
            mlp_rows[["threshold", "f1"]].rename(columns={"f1": "f1_mlp"}),
            on="threshold", how="inner",
        )
    else:
        df_wide = df_search.copy()

    for col, color, model_label, opt in [
        ("f1_lr",  PALETTE["lr"],  "Logistic Reg.", optimal_lr),
        ("f1_mlp", PALETTE["mlp"], "MLP",           optimal_mlp),
    ]:
        if col not in df_wide.columns:
            continue
        fig.add_trace(go.Scatter(
            x=df_wide["threshold"], y=df_wide[col],
            mode="lines", line=dict(color=color, width=2),
            name=f"{model_label} F1",
        ))
        best_rows = df_wide.loc[df_wide["threshold"].round(2) == round(opt, 2), col]
        if not best_rows.empty:
            fig.add_trace(go.Scatter(
                x=[opt], y=[best_rows.values[0]],
                mode="markers",
                marker=dict(symbol="star", size=14, color=color,
                            line=dict(color="white", width=1)),
                name=f"{model_label} optimal ({opt:.2f})",
            ))
    fig.update_xaxes(title="Decision Threshold", range=[0, 1])
    fig.update_yaxes(title="F1 Score")
    return _apply_defaults(fig, "Threshold vs F1 – Validation Set")


# ── 5. SHAP Feature Importance Bar ───────────────────────────────────────────
def shap_importance_bar(
    df_shap: pd.DataFrame,
    model_col: str,     # column name in df_shap, e.g. 'mean_abs_shap_mlp'
    model_label: str,
    top_n: int = 15,
    color: str | None = None,
) -> go.Figure:
    """Horizontal bar chart of global SHAP feature importances."""
    if model_col not in df_shap.columns:
        return go.Figure()

    df = (
        df_shap[["feature", model_col]]
        .rename(columns={model_col: "importance"})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .iloc[::-1]  # flip for horizontal bar (largest at top)
    )
    bar_color = color or PALETTE["mlp"]
    fig = go.Figure(go.Bar(
        x=df["importance"],
        y=df["feature"],
        orientation="h",
        marker_color=bar_color,
        marker_line=dict(width=0),
    ))
    fig.update_xaxes(title="Mean |SHAP Value|")
    fig.update_yaxes(tickfont=dict(size=11))
    return _apply_defaults(fig, f"Global Feature Importance – {model_label} (Top {top_n})")


# ── 6. Risk Distribution Donut ───────────────────────────────────────────────
def risk_distribution_donut(risk_summary: dict, model_label: str) -> go.Figure:
    """Donut chart of Low / Medium / High risk proportions."""
    labels  = ["Low", "Medium", "High"]
    values  = [risk_summary.get(k, {}).get("count", 0) for k in labels]
    colors  = [PALETTE["Low"], PALETTE["Medium"], PALETTE["High"]]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.52,
        marker=dict(colors=colors, line=dict(color=PALETTE["bg"], width=3)),
        textinfo="label+percent",
        textfont=dict(size=14),
        insidetextorientation="radial",
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"Risk Distribution – {model_label}", font=dict(size=15)),
        showlegend=True,
        legend=dict(orientation="h", y=-0.1),
        **_LAYOUT_DEFAULTS,
    )
    return fig


# ── 7. Training History ───────────────────────────────────────────────────────
def training_history_chart(history: dict) -> go.Figure:
    """Loss and accuracy training curves side by side."""
    epochs = list(range(1, len(history.get("loss", [])) + 1))
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Binary Cross-Entropy Loss", "Accuracy"),
        horizontal_spacing=0.12,
    )
    # Loss
    fig.add_trace(go.Scatter(
        x=epochs, y=history.get("loss", []),
        mode="lines", name="Train Loss",
        line=dict(color=PALETTE["mlp"], width=2),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=epochs, y=history.get("val_loss", []),
        mode="lines", name="Val Loss",
        line=dict(color=PALETTE["mlp"], width=2, dash="dot"),
    ), row=1, col=1)
    # Accuracy
    fig.add_trace(go.Scatter(
        x=epochs, y=history.get("accuracy", []),
        mode="lines", name="Train Acc",
        line=dict(color=PALETTE["lr"], width=2),
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=epochs, y=history.get("val_accuracy", []),
        mode="lines", name="Val Acc",
        line=dict(color=PALETTE["lr"], width=2, dash="dot"),
    ), row=1, col=2)

    fig.update_xaxes(title_text="Epoch")
    fig.update_layout(
        legend=dict(orientation="h", y=-0.18),
        **_LAYOUT_DEFAULTS,
    )
    fig.update_layout(title=dict(text="MLP Training History", font=dict(size=15)))
    return fig


# ── 8. Class distribution bar ────────────────────────────────────────────────
def class_distribution_chart(n_no: int, n_yes: int) -> go.Figure:
    """Simple bar chart showing churn class imbalance."""
    fig = go.Figure(go.Bar(
        x=["No Churn", "Churn"],
        y=[n_no, n_yes],
        marker_color=[PALETTE["Low"], PALETTE["High"]],
        text=[n_no, n_yes],
        textposition="outside",
        width=0.4,
    ))
    fig.update_yaxes(title="Customers")
    return _apply_defaults(fig, "Class Distribution – Full Dataset")


# ── 9. Metrics comparison bar chart ─────────────────────────────────────────
def metrics_comparison_chart(
    metrics_lr: dict, metrics_mlp: dict, metric_keys: list[str]
) -> go.Figure:
    """Side-by-side metric comparison for LR vs MLP at optimal threshold."""
    fig = go.Figure()
    vals_lr  = [metrics_lr.get(k, 0)  for k in metric_keys]
    vals_mlp = [metrics_mlp.get(k, 0) for k in metric_keys]

    fig.add_trace(go.Bar(
        name="Logistic Reg.", x=metric_keys, y=vals_lr,
        marker_color=PALETTE["lr"], text=[f"{v:.3f}" for v in vals_lr],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="MLP", x=metric_keys, y=vals_mlp,
        marker_color=PALETTE["mlp"], text=[f"{v:.3f}" for v in vals_mlp],
        textposition="outside",
    ))
    fig.update_layout(barmode="group", yaxis=dict(range=[0, 1.08]))
    return _apply_defaults(fig, "LR vs MLP – Key Metrics at Optimal Threshold (Test Set)")
