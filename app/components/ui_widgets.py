"""
ChurnIQ – app/components/ui_widgets.py
=======================================
Reusable Streamlit UI widget helpers for the dashboard.
Lime-inspired clean SaaS design system.

All functions return or render Streamlit components.
No model loading or computation occurs here.
"""

import streamlit as st


# ── Colour tokens (in sync with custom.css) ──────────────────────────────────
_RISK_EMOJI  = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
_RISK_ACTION = {
    "High":   "Immediate retention intervention — priority outreach, dedicated account review, or customized contract renegotiation.",
    "Medium": "Proactive automated engagement — targeted loyalty incentives, feature adoption nudge, and satisfaction survey.",
    "Low":    "Routine relationship maintenance — standard service marketing, zero invasive intervention required.",
}
_RISK_COLOR = {"High": "#DC2626", "Medium": "#D97706", "Low": "#087A08"}
_RISK_BG    = {"High": "#FEE2E2", "Medium": "#FEF3C7", "Low": "#E9FFE9"}
_RISK_BORDER= {"High": "#FCA5A5", "Medium": "#FCD34D", "Low": "#86EFAC"}


# ── Page Header Component ────────────────────────────────────────────────────
def page_header(
    title: str,
    description: str,
    kicker: str = "Production Telemetry",
    status_badge: str | None = "Live System",
) -> None:
    """Render a unified, sleek SaaS page header banner."""
    badge_html = ""
    if status_badge:
        badge_html = f'<span class="page-header-kicker"><span style="color:#00DD00;">●</span> {status_badge}</span>'
    elif kicker:
        badge_html = f'<span class="page-header-kicker">{kicker}</span>'

    st.markdown(
        f"""
        <div class="page-header-container">
            {badge_html}
            <h1 class="page-header-title">{title}</h1>
            <p class="page-header-desc">{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Metric card (HTML) ────────────────────────────────────────────────────────
def metric_card(
    label: str,
    value: str | float,
    delta: str | None = None,
    delta_positive: bool = True,
    fmt: str = "{:.4f}",
) -> None:
    """Render a clean SaaS styled metric card."""
    if isinstance(value, float):
        display_value = fmt.format(value)
    else:
        display_value = str(value)

    delta_html = ""
    if delta is not None:
        cls = "positive" if delta_positive else "negative"
        arrow = "↑" if delta_positive else "↓"
        delta_html = (
            f'<div class="metric-delta {cls}">{arrow} {delta}</div>'
        )

    st.markdown(
        f"""
        <div class="metric-card">
            <div>
                <div class="metric-label">{label}</div>
                <div class="metric-value">{display_value}</div>
            </div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Risk category badge ───────────────────────────────────────────────────────
def risk_badge(category: str) -> None:
    """Render a colour-coded risk badge with recommended business action."""
    emoji  = _RISK_EMOJI.get(category, "⚪")
    action = _RISK_ACTION.get(category, "")
    st.markdown(
        f"""
        <div style="margin: 0.5rem 0;">
            <span class="risk-badge {category}">{emoji} {category} Risk</span>
        </div>
        <div class="risk-action-box {category}">
            <strong>Action:</strong> {action}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Model selector ────────────────────────────────────────────────────────────
def model_selector(key: str = "model_select") -> str:
    """
    Radio widget to select between LR and MLP.
    Returns 'mlp' or 'logistic_regression'.
    """
    choice = st.radio(
        "Active Model Selection",
        options=["MLP (Neural Network — Champion)", "Logistic Regression (Baseline)"],
        key=key,
        horizontal=True,
    )
    return "mlp" if choice.startswith("MLP") else "logistic_regression"


# ── Threshold display ─────────────────────────────────────────────────────────
def threshold_display(default: float, optimal: float, model_name: str) -> None:
    """Show default vs optimised threshold side-by-side."""
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label=f"{model_name} — Standard Default",
            value=f"{default:.2f}",
        )
    with col2:
        delta_pct = round((optimal - default) * 100, 1)
        st.metric(
            label=f"{model_name} — F1-Optimized",
            value=f"{optimal:.2f}",
            delta=f"{delta_pct:+.1f} pp shift",
        )


# ── SHAP disclaimer banner ───────────────────────────────────────────────────
def shap_disclaimer() -> None:
    """Display the SHAP model-behaviour disclaimer."""
    st.markdown(
        """
        <div class="shap-disclaimer">
            <strong>⚠️ SHAP Interpretability Disclaimer:</strong> SHAP values explain
            <em>internal model behaviour</em> — revealing which features most strongly influenced
            the prediction. They do <strong>not</strong> prove causal relationships or identify
            the root cause of customer decisions. Use these attributions for model governance
            and auditability, not as causal marketing guarantees.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Churn probability gauge ───────────────────────────────────────────────────
def probability_gauge(prob: float) -> None:
    """Display a clean visual probability gauge as a progress bar + badge."""
    pct = int(prob * 100)
    if prob >= 0.70:
        color, bg, border, label = "#DC2626", "#FEE2E2", "#FCA5A5", "HIGH RISK"
    elif prob >= 0.30:
        color, bg, border, label = "#D97706", "#FEF3C7", "#FCD34D", "MEDIUM RISK"
    else:
        color, bg, border, label = "#087A08", "#E9FFE9", "#86EFAC", "LOW RISK"

    st.markdown(
        f"""
        <div style="background: #FFFFFF; border: 1px solid #E5E7E5; border-radius: 12px;
                    padding: 1.25rem 1.4rem; box-shadow: 0 1px 3px rgba(0,0,0,0.03); margin-bottom: 1rem;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.6rem;">
                <span style="font-size:0.78rem; font-weight:700; text-transform:uppercase; letter-spacing:0.06em; color:#737373;">
                    Predicted Churn Probability
                </span>
                <span style="font-size:0.85rem; font-weight:700; color:{color}; background:{bg};
                             border:1px solid {border}; padding:0.25rem 0.65rem; border-radius:9999px;">
                    {pct}% · {label}
                </span>
            </div>
            <div style="background:#F3F4F6; border-radius:9999px; height:12px; overflow:hidden; border: 1px solid #E5E7E5;">
                <div style="width:{pct}%; background:{color}; height:100%; border-radius:9999px;
                            transition: width 0.5s ease;"></div>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:0.74rem; color:#9CA3AF; margin-top:0.4rem;">
                <span>0% (Secure)</span>
                <span>30% (Medium Cutoff)</span>
                <span>70% (High Cutoff)</span>
                <span>100% (Imminent)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Top features table ────────────────────────────────────────────────────────
def top_features_table(features: list[dict]) -> None:
    """
    Render inference top-feature contributions as a clean styled dataframe.
    """
    import pandas as pd

    if not features:
        st.info("No feature contributions available.")
        return

    df = pd.DataFrame(features)[
        ["original_attribute", "feature", "encoded_value", "contribution_proxy"]
    ].rename(columns={
        "original_attribute":  "Feature Domain",
        "feature":             "Encoded Variable",
        "encoded_value":       "Input Value",
        "contribution_proxy":  "Relative Impact Proxy",
    })
    st.dataframe(
        df.style.format({"Input Value": "{:.4f}", "Relative Impact Proxy": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Relative Impact Proxy: coefficient × value (LR) or |encoded value| (MLP). "
        "These are fast local heuristics for immediate inference display, not KernelExplainer SHAP values."
    )


# ── Section divider ───────────────────────────────────────────────────────────
def section_header(title: str, subtitle: str = "") -> None:
    """Render a styled section header with SaaS visual hierarchy."""
    st.markdown(
        f"""
        <div class="section-header-wrap">
            <div class="section-header-title">{title}</div>
            {f'<div class="section-header-subtitle">{subtitle}</div>' if subtitle else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Artifact status row ───────────────────────────────────────────────────────
def artifact_status_row(label: str, path_exists: bool) -> None:
    """Single row in an artifact status telemetry table."""
    icon = "✓" if path_exists else "✗"
    status_text = "Verified" if path_exists else "Missing"
    color = "#087A08" if path_exists else "#DC2626"
    bg = "#E9FFE9" if path_exists else "#FEE2E2"
    border = "#86EFAC" if path_exists else "#FCA5A5"

    st.markdown(
        f"""
        <div style="display:flex; align-items:center; justify-content:space-between; gap:0.6rem;
                    padding:0.45rem 0.6rem; border-bottom:1px solid #F3F4F6;">
            <code style="font-size:0.82rem; color:#171717; background:#F7F9F7;
                         padding:0.25rem 0.55rem; border-radius:6px; border:1px solid #E5E7E5;
                         overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:70%;">{label}</code>
            <span style="font-size:0.75rem; font-weight:700; padding:0.25rem 0.65rem;
                         border-radius:9999px; background:{bg}; color:{color}; border:1px solid {border};
                         white-space:nowrap; flex-shrink:0;">
                {icon} {status_text}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
