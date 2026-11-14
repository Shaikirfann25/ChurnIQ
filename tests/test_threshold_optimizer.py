"""Unit and integration tests for src/threshold_optimizer.py."""

import json
import pytest
import numpy as np
import pandas as pd

from src.threshold_optimizer import (
    search_thresholds,
    select_optimal_threshold,
    metrics_at_threshold,
    assign_risk_categories,
    verify_alignment,
)


@pytest.mark.unit
class TestThresholdOptimizerUnit:
    """Unit tests for threshold search, tie breaking, metrics, and risk tier assignment."""

    def test_search_thresholds_grid_and_metrics(self):
        """search_thresholds must evaluate thresholds from 0.05 to 0.95 in 0.01 increments."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_proba = np.array([0.1, 0.2, 0.4, 0.6, 0.7, 0.8, 0.3, 0.9])

        df_search = search_thresholds(y_true, y_proba, model_name="TestModel")

        assert len(df_search) == 91  # 0.05 to 0.95 step 0.01 is 91 steps
        assert df_search["threshold"].min() == 0.05
        assert df_search["threshold"].max() == 0.95
        assert set(df_search.columns) == {
            "model", "threshold", "precision", "recall", "f1", "tn", "fp", "fn", "tp"
        }
        # Invariants: tn + fp + fn + tp == len(y_true)
        total_cm = df_search["tn"] + df_search["fp"] + df_search["fn"] + df_search["tp"]
        assert (total_cm == len(y_true)).all()

    def test_deterministic_tie_breaking_selects_lowest_threshold(self):
        """When multiple thresholds achieve identical maximum F1, lowest threshold must be chosen."""
        # Artificial DataFrame with ties at threshold 0.40 and 0.50 with F1=0.75
        data = {
            "model": ["ModelA"] * 3,
            "threshold": [0.40, 0.45, 0.50],
            "precision": [0.70, 0.72, 0.75],
            "recall": [0.80, 0.78, 0.75],
            "f1": [0.75, 0.74, 0.75],  # 0.40 and 0.50 tie for max F1
            "tn": [10, 11, 12],
            "fp": [4, 3, 2],
            "fn": [2, 3, 4],
            "tp": [12, 11, 10],
        }
        df_search = pd.DataFrame(data)

        opt = select_optimal_threshold(df_search, model_name="ModelA")
        assert opt["optimal_threshold"] == 0.40
        assert opt["f1"] == 0.75
        assert opt["n_ties"] == 2

    def test_metrics_at_threshold_calculation(self):
        """metrics_at_threshold must correctly compute accuracy, precision, recall, f1, and cm."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.2, 0.4, 0.6, 0.8])

        # At threshold 0.50: predictions are [0, 0, 1, 1] => perfect
        m50 = metrics_at_threshold(y_true, y_proba, threshold=0.50)
        assert m50["f1"] == 1.0
        assert m50["precision"] == 1.0
        assert m50["recall"] == 1.0
        assert m50["roc_auc"] == 1.0
        assert m50["pr_auc"] == 1.0
        assert m50["tp"] == 2
        assert m50["tn"] == 2
        assert m50["fp"] == 0
        assert m50["fn"] == 0

        # At threshold 0.70: predictions are [0, 0, 0, 1] => recall 0.5, precision 1.0
        m70 = metrics_at_threshold(y_true, y_proba, threshold=0.70)
        assert m70["recall"] == 0.5
        assert m70["precision"] == 1.0
        assert m70["fn"] == 1

    def test_assign_risk_categories_boundaries(self):
        """Probabilities must map to Low (<0.30), Medium (0.30 to <0.70), and High (>=0.70)."""
        probs = np.array([0.0, 0.15, 0.2999, 0.30, 0.50, 0.6999, 0.70, 0.95, 1.0])
        categories = assign_risk_categories(probs)

        expected = ["Low", "Low", "Low", "Medium", "Medium", "Medium", "High", "High", "High"]
        assert list(categories) == expected

    def test_verify_alignment_detects_mismatch(self):
        """verify_alignment must raise AssertionError if customer IDs or labels don't match."""
        df1 = pd.DataFrame({"customerID": ["C1", "C2"], "actual_churn": [0, 1], "churn_probability": [0.1, 0.9]})
        df2_bad_id = pd.DataFrame({"customerID": ["C1", "C3"], "actual_churn": [0, 1], "churn_probability": [0.1, 0.9]})
        df2_bad_label = pd.DataFrame({"customerID": ["C1", "C2"], "actual_churn": [1, 1], "churn_probability": [0.1, 0.9]})

        with pytest.raises(AssertionError, match="CustomerID mismatch"):
            verify_alignment(df1, df2_bad_id)

        with pytest.raises(AssertionError, match="actual_churn labels differ"):
            verify_alignment(df1, df2_bad_label)


@pytest.mark.integration
class TestThresholdOptimizerIntegration:
    """Integration tests verifying saved threshold artifacts."""

    def test_saved_threshold_metadata_independent_optimization(self, threshold_metadata):
        """LR and MLP must have independently selected optimal thresholds recorded in metadata."""
        models = threshold_metadata["models"]
        assert "logistic_regression" in models
        assert "mlp" in models

        lr_opt = models["logistic_regression"]["optimal_threshold"]
        mlp_opt = models["mlp"]["optimal_threshold"]

        # Verified values from Module 4
        assert lr_opt == 0.54
        assert mlp_opt == 0.46
        assert lr_opt != mlp_opt, "LR and MLP should have independently optimized thresholds"

        # Risk boundaries recorded and independent
        rb = threshold_metadata["risk_boundaries"]
        assert rb["low_threshold"] == 0.30
        assert rb["high_threshold"] == 0.70

    def test_saved_threshold_search_results_csv(self, reports_dir):
        """Saved threshold search results CSV must contain both models and 182 rows (91 each)."""
        csv_path = reports_dir / "threshold_search_results.csv"
        assert csv_path.exists(), "threshold_search_results.csv missing"

        df = pd.read_csv(csv_path)
        assert len(df) == 182  # 91 thresholds * 2 models
        assert set(df["model"].unique()) == {"LogisticRegression", "MLP"}
        assert df["threshold"].min() == 0.05
        assert df["threshold"].max() == 0.95
