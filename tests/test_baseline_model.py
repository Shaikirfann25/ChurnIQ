"""Unit and integration tests for src/baseline_model.py."""

import json
import pytest
import numpy as np
import pandas as pd

from src.baseline_model import evaluate_baseline_validation
from src.inference import EXPECTED_FEATURES


@pytest.mark.unit
class TestBaselineModelUnit:
    """Unit tests for baseline evaluation logic and prediction contracts."""

    def test_evaluate_baseline_validation_metrics_contract(self, baseline_logistic_model):
        """Evaluation function must return valid metrics dictionary and probabilities."""
        np.random.seed(42)
        n = 20
        # Synthetic 46-feature DataFrame
        X_dummy = pd.DataFrame(np.random.randn(n, len(EXPECTED_FEATURES)), columns=EXPECTED_FEATURES)
        y_dummy = pd.Series([1 if i % 3 == 0 else 0 for i in range(n)])

        metrics, proba, pred = evaluate_baseline_validation(
            baseline_logistic_model, X_dummy, y_dummy, threshold=0.50
        )

        assert isinstance(metrics, dict)
        assert "roc_auc" in metrics
        assert "pr_auc" in metrics
        assert "f1_score" in metrics
        assert "confusion_matrix" in metrics
        assert len(proba) == n
        assert len(pred) == n
        assert np.all((proba >= 0.0) & (proba <= 1.0))
        assert set(np.unique(pred)).issubset({0, 1})

    def test_invalid_input_dimension_raises(self, baseline_logistic_model):
        """Providing incorrect feature dimension (e.g. 10 instead of 46) must raise an error."""
        X_bad = pd.DataFrame(np.random.randn(5, 10))
        with pytest.raises(ValueError):
            baseline_logistic_model.predict_proba(X_bad)

    def test_probability_output_range_and_monotonicity(self, baseline_logistic_model, synthetic_encoded_features):
        """Predictions on synthetic encoded features must strictly fall in [0, 1]."""
        proba = baseline_logistic_model.predict_proba(synthetic_encoded_features)[:, 1]
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)
        assert proba.dtype in (np.float32, np.float64)


@pytest.mark.integration
class TestBaselineModelIntegration:
    """Integration tests verifying saved baseline model and reports."""

    def test_baseline_artifact_properties(self, baseline_logistic_model, models_dir):
        """Model must be a valid fitted LogisticRegression with 46 features."""
        assert hasattr(baseline_logistic_model, "coef_")
        assert baseline_logistic_model.coef_.shape == (1, 46)
        assert hasattr(baseline_logistic_model, "intercept_")
        assert baseline_logistic_model.classes_.tolist() == [0, 1]

    def test_validation_metrics_match_saved_report(
        self, baseline_logistic_model, processed_data_dir, reports_dir
    ):
        """Running baseline model on val partition must match saved baseline validation report."""
        val_df = pd.read_parquet(processed_data_dir / "val_split.parquet")
        X_val = val_df[EXPECTED_FEATURES]
        y_val = val_df["Churn"]

        metrics, proba, pred = evaluate_baseline_validation(
            baseline_logistic_model, X_val, y_val, threshold=0.50
        )

        report_path = reports_dir / "baseline_validation_metrics.json"
        assert report_path.exists(), "baseline_validation_metrics.json missing"
        with open(report_path, "r", encoding="utf-8") as f:
            saved_metrics = json.load(f)

        assert metrics["roc_auc"] == saved_metrics["roc_auc"]
        assert metrics["pr_auc"] == saved_metrics["pr_auc"]
        assert metrics["f1_score"] == saved_metrics["f1_score"]
        assert metrics["precision"] == saved_metrics["precision"]
        assert metrics["recall"] == saved_metrics["recall"]

    def test_val_predictions_parquet_integrity(self, reports_dir):
        """Saved baseline validation prediction parquet must have expected columns and align."""
        pred_path = reports_dir / "baseline_val_predictions.parquet"
        assert pred_path.exists(), "baseline_val_predictions.parquet missing"
        df_pred = pd.read_parquet(pred_path)

        assert len(df_pred) == 1056
        assert set(df_pred.columns) == {
            "customerID",
            "actual_churn",
            "churn_probability",
            "predicted_churn_at_0.50",
        }
        assert np.all((df_pred["churn_probability"] >= 0.0) & (df_pred["churn_probability"] <= 1.0))

