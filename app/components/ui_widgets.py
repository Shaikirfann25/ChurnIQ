"""
ChurnIQ – app/components/ui_widgets.py
=======================================
Reusable Streamlit UI widget helpers for the dashboard.

All functions return or render Streamlit components.
No model loading or computation occurs here.
"""

import streamlit as st


# ── Colour tokens (must stay in sync with custom.css) ────────────────────────
_RISK_EMOJI  = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
_RISK_ACTION = {
    "High":   "Immediate retention offer — high-touch personalised discount or direct outreach.",
    "Medium": "Proactive outreach — automated low-cost nudge or loyalty programme engagement.",
    "Low":    "Routine engagement — standard marketing, no urgent intervention needed.",
}
_RISK_COLOR = {"High": "#f87171", "Medium": "#fbbf24", "Low": "#34d399"}


# ── Metric card (HTML) ────────────────────────────────────────────────────────
def metric_card(
    label: str,
    value: str | float,
    delta: str | None = None,
    delta_positive: bool = True,
    fmt: str = "{:.4f}",
) -> None:
    """Render a styled metric card via markdown."""
    if isinstance(value, float):
        display_value = fmt.format(value)
    else:
        display_value = str(value)

    delta_html = ""
    if delta is not None:
        cls = "positive" if delta_positive else "negative"
        arrow = "▲" if delta_positive else "▼"
        delta_html = (
            f'<div class="metric-delta {cls}">{arrow} {delta}</div>'
        )

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{display_value}</div>
            <div class="metric-label">{label}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Risk category badge ───────────────────────────────────────────────────────
def risk_badge(category: str) -> None:
    """Render a colour-coded risk badge."""
    emoji  = _RISK_EMOJI.get(category, "⚪")
    action = _RISK_ACTION.get(category, "")
    color  = _RISK_COLOR.get(category, "#9ca3af")
    st.markdown(
        f"""
        <div style="margin: 0.6rem 0;">
            <span class="risk-badge {category}">{emoji} {category} Risk</span>
        </div>
        <div style="color:{color}; font-size:0.88rem; margin-top:0.4rem;">
            {action}
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
        "Select Model",
        options=["MLP (Neural Network)", "Logistic Regression (Baseline)"],
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
            label=f"{model_name} — Default Threshold",
            value=f"{default:.2f}",
        )
    with col2:
        delta_pct = round((optimal - default) * 100, 1)
        st.metric(
            label=f"{model_name} — Optimal Threshold",
            value=f"{optimal:.2f}",
            delta=f"{delta_pct:+.1f} pp vs default",
        )


# ── SHAP disclaimer banner ───────────────────────────────────────────────────
def shap_disclaimer() -> None:
    """Display the SHAP model-behaviour disclaimer."""
    st.markdown(
        """
        <div class="shap-disclaimer">
            ⚠️ <strong>SHAP Disclaimer:</strong> SHAP values explain
            <em>model behaviour</em> — which features most influenced this specific
            prediction. They do <strong>not</strong> imply causal relationships or
            prove that a feature causes churn. Use these explanations to understand
            and audit model decisions, not to infer causal business factors.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Churn probability gauge (text-based) ─────────────────────────────────────
def probability_gauge(prob: float) -> None:
    """Display a visual probability gauge as a progress bar + badge."""
    pct = int(prob * 100)
    if prob >= 0.70:
        color, label = "#f87171", "HIGH RISK"
    elif prob >= 0.30:
        color, label = "#fbbf24", "MEDIUM RISK"
    else:
        color, label = "#34d399", "LOW RISK"

    st.markdown(
        f"""
        <div style="margin: 0.5rem 0 1rem;">
            <div style="display:flex; justify-content:space-between; margin-bottom:0.3rem;">
                <span style="font-size:0.85rem; color:#9ca3af;">Churn Probability</span>
                <span style="font-weight:700; color:{color};">{pct}% — {label}</span>
            </div>
            <div style="background:#2d3148; border-radius:6px; height:14px; overflow:hidden;">
                <div style="width:{pct}%; background:{color}; height:100%; border-radius:6px;
                            transition: width 0.5s ease;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Top features table ────────────────────────────────────────────────────────
def top_features_table(features: list[dict]) -> None:
    """
    Render inference top-feature contributions as a styled table.

    Parameters
    ----------
    features : list of dicts with keys
               feature, original_attribute, encoded_value, contribution_proxy
    """
    import pandas as pd

    if not features:
        st.info("No feature contributions available.")
        return

    df = pd.DataFrame(features)[
        ["original_attribute", "feature", "encoded_value", "contribution_proxy"]
    ].rename(columns={
        "original_attribute":  "Original Feature",
        "feature":             "Encoded Column",
        "encoded_value":       "Encoded Value",
        "contribution_proxy":  "Contribution Proxy",
    })
    st.dataframe(
        df.style.format({"Encoded Value": "{:.4f}", "Contribution Proxy": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Contribution Proxy: coefficient × value (LR) or |feature value| (MLP). "
        "These are lightweight approximations, not SHAP values."
    )


# ── Section divider ───────────────────────────────────────────────────────────
def section_header(title: str, subtitle: str = "") -> None:
    """Render a styled section header."""
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)
    st.divider()


# ── Artifact status row ───────────────────────────────────────────────────────
def artifact_status_row(label: str, path_exists: bool) -> None:
    """Single row in an artifact status table."""
    icon = "✅" if path_exists else "❌"
    color = "#34d399" if path_exists else "#f87171"
    st.markdown(
        f'<div style="color:{color}; padding:0.15rem 0;">{icon} <code>{label}</code></div>',
        unsafe_allow_html=True,
    )

