"""Unit and integration tests for ChurnIQ dashboard (Module 9)."""

import ast
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import plotly.graph_objects as go

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
)
from app.streamlit_app import (
    load_threshold_metadata,
    load_final_test_metrics,
    load_training_history,
    load_threshold_search,
    load_shap_importance,
    load_inference_predictions,
    load_raw_dataset_stats,
    compute_roc_pr_data,
)


@pytest.mark.unit
class TestAppUnit:
    """Unit tests for chart factories, UI components, and Streamlit API compatibility."""

    def test_streamlit_app_has_no_unsupported_image_arguments(self, project_root):
        """Verify that st.image() does not use 'use_container_width' (unsupported in Streamlit 1.39.1)."""
        app_file = project_root / "app" / "streamlit_app.py"
        assert app_file.exists(), "streamlit_app.py missing"
        source = app_file.read_text(encoding="utf-8")

        # Check line by line
        for idx, line in enumerate(source.splitlines(), start=1):
            if "st.image" in line:
                assert "use_container_width" not in line, (
                    f"Line {idx} in app/streamlit_app.py uses unsupported argument 'use_container_width' "
                    f"in st.image(): {line}"
                )
                assert "use_column_width=True" in line, (
                    f"Line {idx} in app/streamlit_app.py should use 'use_column_width=True': {line}"
                )

    def test_custom_css_exists_and_styled(self, project_root):
        """Custom CSS file must exist and contain dark-theme and metric-card classes."""
        css_file = project_root / "app" / "styles" / "custom.css"
        assert css_file.exists(), "custom.css missing"
        css_text = css_file.read_text(encoding="utf-8")
        assert ".metric-card" in css_text
        assert ".risk-badge" in css_text
        assert ".shap-disclaimer" in css_text

    def test_roc_and_pr_curve_charts_return_valid_figures(self):
        """roc_curve_chart and pr_curve_chart must return Plotly figures with expected traces."""
        fpr = np.array([0.0, 0.2, 0.5, 1.0])
        tpr = np.array([0.0, 0.6, 0.85, 1.0])
        fig_roc = roc_curve_chart(fpr, tpr, 0.84, fpr, tpr, 0.85)
        assert isinstance(fig_roc, go.Figure)
        assert len(fig_roc.data) == 3  # Diagonal, LR, MLP

        prec = np.array([1.0, 0.7, 0.5, 0.26])
        rec = np.array([0.0, 0.4, 0.8, 1.0])
        fig_pr = pr_curve_chart(prec, rec, 0.63, prec, rec, 0.64, baseline_rate=0.26)
        assert isinstance(fig_pr, go.Figure)
        assert len(fig_pr.data) == 3

    def test_confusion_matrix_chart_return_valid_figure(self):
        """confusion_matrix_chart must return annotated heatmap."""
        fig_cm = confusion_matrix_chart(tn=596, fp=180, fn=71, tp=210, model_name="LR", threshold=0.54)
        assert isinstance(fig_cm, go.Figure)
        assert len(fig_cm.data) == 1
        assert isinstance(fig_cm.data[0], go.Heatmap)

    def test_threshold_f1_chart_return_valid_figure(self):
        """threshold_f1_chart must plot curves for both models with optimal star markers."""
        df_dummy = pd.DataFrame({
            "model": ["LogisticRegression"] * 2 + ["MLP"] * 2,
            "threshold": [0.46, 0.54, 0.46, 0.54],
            "f1": [0.62, 0.63, 0.63, 0.62],
        })
        fig_th = threshold_f1_chart(df_dummy, optimal_lr=0.54, optimal_mlp=0.46)
        assert isinstance(fig_th, go.Figure)
        assert len(fig_th.data) >= 2

    def test_shap_importance_and_risk_donut_charts(self):
        """SHAP importance and risk distribution charts must produce valid Plotly figures."""
        df_imp = pd.DataFrame({
            "feature": ["tenure", "Contract", "MonthlyCharges"],
            "importance": [0.15, 0.12, 0.08],
        })
        fig_imp = shap_importance_bar(df_imp, "importance", "MLP", top_n=3)
        assert isinstance(fig_imp, go.Figure)
        assert isinstance(fig_imp.data[0], go.Bar)

        summary = {"Low": {"count": 400}, "Medium": {"count": 300}, "High": {"count": 200}}
        fig_donut = risk_distribution_donut(summary, "MLP")
        assert isinstance(fig_donut, go.Figure)
        assert isinstance(fig_donut.data[0], go.Pie)

    def test_training_history_and_metrics_comparison_charts(self):
        """Training history and metric comparison charts must return valid figures."""
        hist = {
            "loss": [0.65, 0.55],
            "val_loss": [0.60, 0.52],
            "accuracy": [0.70, 0.75],
            "val_accuracy": [0.72, 0.74],
        }
        fig_hist = training_history_chart(hist)
        assert isinstance(fig_hist, go.Figure)

        fig_cmp = metrics_comparison_chart({"f1": 0.62}, {"f1": 0.63}, ["f1"])
        assert isinstance(fig_cmp, go.Figure)


@pytest.mark.integration
class TestAppIntegration:
    """Integration tests verifying dashboard cached data loading functions and static assets."""

    def test_app_cached_data_loaders(self):
        """All dashboard data loading functions must read saved artifacts successfully."""
        meta = load_threshold_metadata()
        assert "models" in meta
        assert meta["models"]["logistic_regression"]["optimal_threshold"] == 0.54
        assert meta["models"]["mlp"]["optimal_threshold"] == 0.46

        metrics = load_final_test_metrics()
        assert len(metrics) == 4

        hist = load_training_history()
        assert "loss" in hist

        df_search = load_threshold_search()
        assert len(df_search) == 182

        df_shap = load_shap_importance()
        assert len(df_shap) == 38

        df_inf = load_inference_predictions("mlp")
        assert len(df_inf) == 1056

        stats = load_raw_dataset_stats()
        assert stats["n_total"] == 7043
        assert stats["churn_rate"] == 26.5

        roc_pr = compute_roc_pr_data()
        assert "logistic_regression" in roc_pr
        assert "mlp" in roc_pr

    def test_all_referenced_images_exist(self, figures_dir):
        """All static PNG plots referenced by the dashboard must exist and be non-empty."""
        required_images = [
            "roc_pr_curves_test.png",
            "confusion_matrices_test.png",
            "shap_importance_mlp.png",
            "shap_summary_mlp.png",
            "shap_importance_logistic.png",
            "shap_summary_logistic.png",
            "shap_local_mlp_tp.png",
            "shap_local_mlp_tn.png",
            "shap_local_mlp_fp.png",
            "shap_local_mlp_fn.png",
            "shap_local_lr_tp.png",
            "shap_local_lr_tn.png",
            "shap_local_lr_fp.png",
            "shap_local_lr_fn.png",
        ]
        for img_name in required_images:
            img_path = figures_dir / img_name
            assert img_path.exists(), f"Missing dashboard image: {img_name}"
            assert img_path.stat().st_size > 0, f"Empty dashboard image: {img_name}"

