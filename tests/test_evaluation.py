"""Unit and integration tests for src/evaluation.py."""

import json
import pytest
import numpy as np
import pandas as pd

from src.evaluation import (
    full_metrics,
    generalization_analysis,
    compare_models,
    VALIDATION_METRICS,
)


@pytest.mark.unit
class TestEvaluationUnit:
    """Unit tests for metric calculation suites, generalization logic, and model comparison rules."""

    def test_full_metrics_suite_correctness(self):
        """full_metrics must compute all required threshold-dependent and independent metrics."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.4, 0.6, 0.8, 0.3, 0.9])

        m = full_metrics(
            y_true=y_true,
            y_proba=y_prob,
            threshold=0.50,
            model_name="UnitModel",
            threshold_label="test_50",
        )

        assert m["model"] == "UnitModel"
        assert m["threshold"] == 0.50
        assert m["n_test"] == len(y_true)
        assert 0.0 <= m["roc_auc"] <= 1.0
        assert 0.0 <= m["pr_auc"] <= 1.0
        assert 0.0 <= m["accuracy"] <= 1.0
        assert 0.0 <= m["precision"] <= 1.0
        assert 0.0 <= m["recall"] <= 1.0
        assert 0.0 <= m["f1"] <= 1.0
        assert 0.0 <= m["specificity"] <= 1.0
        assert 0.0 <= m["fpr"] <= 1.0
        assert 0.0 <= m["fnr"] <= 1.0

        cm = m["confusion_matrix"]
        assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == len(y_true)
        assert round(m["fpr"] + m["specificity"], 4) == 1.0
        assert round(m["recall"] + m["fnr"], 4) == 1.0

    def test_generalization_analysis_no_degradation(self):
        """Minimal delta between val and test should report no meaningful degradation."""
        val_m = {"roc_auc": 0.85, "pr_auc": 0.65, "f1_optimal": 0.64, "precision": 0.60, "recall": 0.70}
        test_m = {"roc_auc": 0.849, "pr_auc": 0.652, "f1": 0.639, "precision": 0.61, "recall": 0.69}

        gen = generalization_analysis("TestModel", val_m, test_m)
        assert gen["model"] == "TestModel"
        assert "No meaningful generalization degradation detected" in gen["finding"]
        assert abs(gen["deltas"]["roc_auc"]["delta"]) < 0.01

    def test_generalization_analysis_detects_gap(self):
        """Large drops in test metrics must trigger degradation warning."""
        val_m = {"roc_auc": 0.90, "pr_auc": 0.80, "f1_optimal": 0.85, "precision": 0.85, "recall": 0.85}
        test_m = {"roc_auc": 0.80, "pr_auc": 0.65, "f1": 0.70, "precision": 0.70, "recall": 0.70}

        gen = generalization_analysis("OverfitModel", val_m, test_m)
        assert "Moderate generalization gap detected" in gen["finding"]

    def test_compare_models_decisiveness_threshold(self):
        """Winner should be declared only if ROC-AUC or F1 difference exceeds 0.005."""
        # Case 1: MLP leads on F1 by > 0.005
        lr_m = {"roc_auc": 0.8448, "f1": 0.6259}
        mlp_m = {"roc_auc": 0.8465, "f1": 0.6326}  # F1 diff = 0.0067 > 0.005
        verdict = compare_models(lr_m, mlp_m)
        assert "MLP leads on F1" in verdict

        # Case 2: Below 0.005 difference on both metrics
        lr_close = {"roc_auc": 0.8450, "f1": 0.6300}
        mlp_close = {"roc_auc": 0.8470, "f1": 0.6320}  # both diffs <= 0.002
        verdict_close = compare_models(lr_close, mlp_close)
        assert "No meaningful winner" in verdict_close


@pytest.mark.integration
class TestEvaluationIntegration:
    """Integration tests verifying saved final evaluation artifacts without modifying them."""

    def test_final_test_metrics_json_integrity(self, reports_dir, threshold_metadata):
        """final_test_metrics.json must reflect frozen thresholds from threshold_metadata."""
        metrics_file = reports_dir / "final_test_metrics.json"
        assert metrics_file.exists(), "final_test_metrics.json missing"

        with open(metrics_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "results" in data
        assert "generalization_analysis" in data
        assert "model_comparison_verdict" in data

        results = data["results"]
        assert len(results) == 4  # LR (default, opt), MLP (default, opt)

        # Check optimal thresholds used
        lr_opt_meta = threshold_metadata["models"]["logistic_regression"]["optimal_threshold"]
        mlp_opt_meta = threshold_metadata["models"]["mlp"]["optimal_threshold"]

        lr_opt_res = next(r for r in results if r["model"] == "LogisticRegression" and r["threshold_label"] == "optimal")
        mlp_opt_res = next(r for r in results if r["model"] == "MLP" and r["threshold_label"] == "optimal")

        assert lr_opt_res["threshold"] == lr_opt_meta
        assert mlp_opt_res["threshold"] == mlp_opt_meta
        assert lr_opt_res["n_test"] == 1057
        assert mlp_opt_res["n_test"] == 1057

    def test_final_test_predictions_parquet(self, reports_dir):
        """final_test_predictions.parquet must have 1057 test records with valid probabilities."""
        pred_file = reports_dir / "final_test_predictions.parquet"
        assert pred_file.exists(), "final_test_predictions.parquet missing"

        df = pd.read_parquet(pred_file)
        assert len(df) == 1057
        for col in ("customerID", "actual_churn", "lr_churn_probability", "mlp_churn_probability"):
            assert col in df.columns

        assert np.all((df["lr_churn_probability"] >= 0.0) & (df["lr_churn_probability"] <= 1.0))
        assert np.all((df["mlp_churn_probability"] >= 0.0) & (df["mlp_churn_probability"] <= 1.0))

    def test_model_comparison_csv_matches_metrics(self, reports_dir):
        """model_comparison.csv must exist and have 4 metric rows."""
        csv_file = reports_dir / "model_comparison.csv"
        assert csv_file.exists(), "model_comparison.csv missing"

        df = pd.read_csv(csv_file)
        assert len(df) == 4
        assert "model" in df.columns
        assert "f1" in df.columns
        assert "roc_auc" in df.columns
