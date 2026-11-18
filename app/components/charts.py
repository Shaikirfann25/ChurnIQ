"""
ChurnIQ – app/components/charts.py
===================================
Reusable Plotly chart factories for the Streamlit dashboard.
Lime-inspired clean SaaS design system with high-contrast readability.

All functions accept pre-loaded data (DataFrames, dicts, lists) and return
Plotly Figure objects ready for `st.plotly_chart(fig, use_container_width=True)`.

No model loading or heavy computation occurs here.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# ── Colour palette (Lime-inspired clean SaaS) ────────────────────────────────
PALETTE = {
    "lr":        "#737373",   # Logistic Regression – Neutral Grey
    "mlp":       "#00DD00",   # MLP – Primary Lime Green (Champion Model)
    "lime":      "#00DD00",   # Primary Lime accent
    "positive":  "#087A08",   # Dark green (accessible text/emphasis)
    "negative":  "#DC2626",   # Red
    "warn":      "#D97706",   # Amber
    "bg":        "#FFFFFF",   # Pure surface white
    "grid":      "#F0F2F0",   # Very soft grid
    "border":    "#E5E7E5",   # Card border
    "text":      "#171717",   # Dark primary text
    "muted":     "#737373",   # Secondary text
    "High":      "#DC2626",
    "Medium":    "#D97706",
    "Low":       "#087A08",
}

_PLOTLY_TEMPLATE = "plotly_white"

_LAYOUT_DEFAULTS = dict(
    template=_PLOTLY_TEMPLATE,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=PALETTE["text"], family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"),
    margin=dict(l=45, r=25, t=55, b=45),
)


def _apply_defaults(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>" if title else "",
            font=dict(size=15, color=PALETTE["text"]),
            x=0.01,
            xanchor="left",
        ),
        **_LAYOUT_DEFAULTS,
    )
    fig.update_xaxes(
        gridcolor=PALETTE["grid"],
        zerolinecolor=PALETTE["border"],
        linecolor=PALETTE["border"],
        tickfont=dict(color=PALETTE["muted"], size=11),
        title_font=dict(color=PALETTE["text"], size=12),
    )
    fig.update_yaxes(
        gridcolor=PALETTE["grid"],
        zerolinecolor=PALETTE["border"],
        linecolor=PALETTE["border"],
        tickfont=dict(color=PALETTE["muted"], size=11),
        title_font=dict(color=PALETTE["text"], size=12),
    )
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
        line=dict(dash="dash", color="#9CA3AF", width=1.5),
        name="Random Guess (AUC=0.50)", showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=fpr_lr, y=tpr_lr, mode="lines",
        line=dict(color=PALETTE["lr"], width=2.5),
        name=f"Logistic Reg. (AUC={auc_lr:.4f})",
    ))
    fig.add_trace(go.Scatter(
        x=fpr_mlp, y=tpr_mlp, mode="lines",
        line=dict(color=PALETTE["mlp"], width=3.0),
        name=f"MLP Champion (AUC={auc_mlp:.4f})",
    ))
    fig.update_xaxes(title="False Positive Rate (1 - Specificity)", range=[-0.01, 1.01])
    fig.update_yaxes(title="True Positive Rate (Recall)",  range=[-0.01, 1.03])
    fig.update_layout(legend=dict(orientation="h", y=-0.22, x=0))
    return _apply_defaults(fig, "ROC Curve – Test Set Evaluation")


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
        line=dict(dash="dash", color="#9CA3AF", width=1.5),
        name=f"No Skill (Prevalence={baseline_rate:.2f})", showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=rec_lr, y=prec_lr, mode="lines",
        line=dict(color=PALETTE["lr"], width=2.5),
        name=f"Logistic Reg. (PR-AUC={prauc_lr:.4f})",
    ))
    fig.add_trace(go.Scatter(
        x=rec_mlp, y=prec_mlp, mode="lines",
        line=dict(color=PALETTE["mlp"], width=3.0),
        name=f"MLP Champion (PR-AUC={prauc_mlp:.4f})",
    ))
    fig.update_xaxes(title="Recall (Sensitivity)", range=[-0.01, 1.01])
    fig.update_yaxes(title="Precision (PPV)", range=[-0.01, 1.03])
    fig.update_layout(legend=dict(orientation="h", y=-0.22, x=0))
    return _apply_defaults(fig, "Precision-Recall Curve – Test Set Evaluation")


# ── 3. Confusion Matrix ───────────────────────────────────────────────────────
def confusion_matrix_chart(
    tn: int, fp: int, fn: int, tp: int,
    model_name: str,
    threshold: float,
) -> go.Figure:
    """Annotated confusion-matrix heatmap with clean SaaS styling."""
    z    = [[tn, fp], [fn, tp]]
    text = [
        [f"<b>True Negative (TN)</b><br><span style='font-size:22px;'>{tn:,}</span><br>Correct Non-Churn",
         f"<b>False Positive (FP)</b><br><span style='font-size:22px;'>{fp:,}</span><br>False Alarm"],
        [f"<b>False Negative (FN)</b><br><span style='font-size:22px;'>{fn:,}</span><br>Missed Churn",
         f"<b>True Positive (TP)</b><br><span style='font-size:22px;'>{tp:,}</span><br>Caught Churn"]
    ]

    fig = go.Figure(go.Heatmap(
        z=z,
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=13, color=PALETTE["text"]),
        colorscale=[
            [0.0, "#F7F9F7"],
            [0.3, "#E9FFE9"],
            [0.7, "#86EFAC"],
            [1.0, "#00DD00"],
        ],
        showscale=False,
        xgap=4, ygap=4,
    ))
    fig.update_xaxes(
        tickvals=[0, 1], ticktext=["<b>Predicted: Retained</b>", "<b>Predicted: Churn</b>"],
        title="Model Prediction", side="bottom",
    )
    fig.update_yaxes(
        tickvals=[0, 1], ticktext=["<b>Actual: Retained</b>", "<b>Actual: Churn</b>"],
        title="Actual Outcome", autorange="reversed",
    )
    return _apply_defaults(
        fig, f"{model_name} – Confusion Matrix (τ = {threshold:.2f})"
    )


# ── 4. Threshold F1 Search ────────────────────────────────────────────────────
def threshold_f1_chart(
    df_search: pd.DataFrame,
    optimal_lr: float,
    optimal_mlp: float,
) -> go.Figure:
    """F1 vs threshold curves for both models with optimal markers."""
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
        ("f1_lr",  PALETTE["lr"],  "Logistic Regression", optimal_lr),
        ("f1_mlp", PALETTE["mlp"], "MLP Champion",       optimal_mlp),
    ]:
        if col not in df_wide.columns:
            continue
        fig.add_trace(go.Scatter(
            x=df_wide["threshold"], y=df_wide[col],
            mode="lines", line=dict(color=color, width=2.5),
            name=f"{model_label} F1",
        ))
        best_rows = df_wide.loc[df_wide["threshold"].round(2) == round(opt, 2), col]
        if not best_rows.empty:
            fig.add_trace(go.Scatter(
                x=[opt], y=[best_rows.values[0]],
                mode="markers+text",
                marker=dict(symbol="star", size=15, color=color,
                            line=dict(color="#171717", width=1.5)),
                name=f"{model_label} Optimal (τ={opt:.2f}, F1={best_rows.values[0]:.4f})",
                text=[f"  τ={opt:.2f}"],
                textposition="top right",
                textfont=dict(size=12, color=color),
            ))

    # Add default reference line at 0.50
    fig.add_vline(x=0.50, line_dash="dash", line_color="#9CA3AF", annotation_text="Default τ=0.50")

    fig.update_xaxes(title="Decision Threshold (τ)", range=[0, 1])
    fig.update_yaxes(title="Validation F1 Score")
    fig.update_layout(legend=dict(orientation="h", y=-0.22, x=0))
    return _apply_defaults(fig, "Threshold Optimization Curve – Validation F1 Profile")


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
        .iloc[::-1]  # flip for horizontal bar
    )
    bar_color = color or PALETTE["mlp"]
    fig = go.Figure(go.Bar(
        x=df["importance"],
        y=df["feature"],
        orientation="h",
        marker=dict(
            color=bar_color,
            line=dict(color="rgba(0,0,0,0)", width=0),
        ),
        text=[f"{v:.4f}" for v in df["importance"]],
        textposition="outside",
        textfont=dict(size=11, color=PALETTE["muted"]),
    ))
    fig.update_xaxes(title="Mean |SHAP Value| (Impact on Probability Output)")
    fig.update_yaxes(tickfont=dict(size=11, color=PALETTE["text"]))
    fig.update_layout(margin=dict(l=150, r=40, t=50, b=40))
    return _apply_defaults(fig, f"Global Feature Importance – {model_label} (Top {top_n})")


# ── 6. Risk Distribution Donut ───────────────────────────────────────────────
def risk_distribution_donut(risk_summary: dict, model_label: str) -> go.Figure:
    """Donut chart of Low / Medium / High risk proportions."""
    labels  = ["Low", "Medium", "High"]
    values  = [risk_summary.get(k, {}).get("count", 0) for k in labels]
    colors  = [PALETTE["Low"], PALETTE["Medium"], PALETTE["High"]]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.60,
        marker=dict(colors=colors, line=dict(color="#FFFFFF", width=3)),
        textinfo="label+percent",
        textfont=dict(size=13, color="#FFFFFF", family="Inter, sans-serif"),
        insidetextorientation="radial",
        hovertemplate="<b>%{label} Risk Tier</b><br>Accounts: %{value:,}<br>Validation Cohort Share: %{percent}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(
            text=f"<b>Validation Cohort Risk Stratification (N=1,056) – {model_label}</b>",
            font=dict(size=15, color=PALETTE["text"]),
            x=0.01,
            xanchor="left",
        ),
        showlegend=True,
        legend=dict(orientation="h", y=-0.12, x=0.1),
        **_LAYOUT_DEFAULTS,
    )
    return fig


# ── 7. Training History ───────────────────────────────────────────────────────
def training_history_chart(history: dict) -> go.Figure:
    """Loss and accuracy training curves side by side."""
    epochs = list(range(1, len(history.get("loss", [])) + 1))
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("<b>Binary Cross-Entropy Loss</b>", "<b>Classification Accuracy</b>"),
        horizontal_spacing=0.12,
    )
    # Loss
    fig.add_trace(go.Scatter(
        x=epochs, y=history.get("loss", []),
        mode="lines", name="Train Loss",
        line=dict(color=PALETTE["mlp"], width=2.5),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=epochs, y=history.get("val_loss", []),
        mode="lines", name="Val Loss",
        line=dict(color=PALETTE["mlp"], width=2, dash="dash"),
    ), row=1, col=1)
    # Accuracy
    fig.add_trace(go.Scatter(
        x=epochs, y=history.get("accuracy", []),
        mode="lines", name="Train Accuracy",
        line=dict(color=PALETTE["lr"], width=2.5),
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=epochs, y=history.get("val_accuracy", []),
        mode="lines", name="Val Accuracy",
        line=dict(color=PALETTE["lr"], width=2, dash="dash"),
    ), row=1, col=2)

    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_layout(
        legend=dict(orientation="h", y=-0.18, x=0.2),
        **_LAYOUT_DEFAULTS,
    )
    fig.update_layout(
        title=dict(
            text="<b>MLP Neural Network Convergence History</b>",
            font=dict(size=15, color=PALETTE["text"]),
            x=0.01,
            xanchor="left",
        )
    )
    return fig


# ── 8. Class distribution bar ────────────────────────────────────────────────
def class_distribution_chart(n_no: int, n_yes: int) -> go.Figure:
    """Simple bar chart showing churn class imbalance."""
    total = n_no + n_yes
    pct_no = n_no / total * 100
    pct_yes = n_yes / total * 100

    fig = go.Figure(go.Bar(
        x=["Retained (No Churn)", "Churned (Yes)"],
        y=[n_no, n_yes],
        marker=dict(
            color=[PALETTE["Low"], PALETTE["High"]],
            line=dict(color="rgba(0,0,0,0)", width=0),
        ),
        text=[f"<b>{n_no:,}</b><br>({pct_no:.1f}%)", f"<b>{n_yes:,}</b><br>({pct_yes:.1f}%)"],
        textposition="outside",
        width=0.45,
    ))
    fig.update_yaxes(title="Customer Count", range=[0, max(n_no, n_yes) * 1.15])
    return _apply_defaults(fig, "Benchmark Class Distribution (Full Dataset N=7,043)")


# ── 9. Metrics comparison bar chart ─────────────────────────────────────────
def metrics_comparison_chart(
    metrics_lr: dict, metrics_mlp: dict, metric_keys: list[str]
) -> go.Figure:
    """Side-by-side metric comparison for LR vs MLP at optimal threshold."""
    fig = go.Figure()
    vals_lr  = [metrics_lr.get(k, 0)  for k in metric_keys]
    vals_mlp = [metrics_mlp.get(k, 0) for k in metric_keys]

    formatted_labels = [k.replace("_", "-").upper() if len(k) <= 4 else k.replace("_", " ").title() for k in metric_keys]

    fig.add_trace(go.Bar(
        name="Logistic Regression (τ=0.54)", x=formatted_labels, y=vals_lr,
        marker_color=PALETTE["lr"], text=[f"{v:.3f}" for v in vals_lr],
        textposition="outside", textfont=dict(size=11),
    ))
    fig.add_trace(go.Bar(
        name="MLP Champion (τ=0.46)", x=formatted_labels, y=vals_mlp,
        marker_color=PALETTE["mlp"], text=[f"{v:.3f}" for v in vals_mlp],
        textposition="outside", textfont=dict(size=11),
    ))
    fig.update_layout(
        barmode="group",
        yaxis=dict(range=[0, 1.12], title="Score"),
        legend=dict(orientation="h", y=-0.2, x=0.2),
    )
    return _apply_defaults(fig, "Test Set Model Evaluation – LR vs MLP at Optimal Thresholds")
