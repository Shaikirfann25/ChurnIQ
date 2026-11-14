"""Unit and integration tests for src/explainability.py."""

import json
import pytest
import numpy as np
import pandas as pd
import joblib

from src.explainability import (
    build_feature_groups,
    aggregate_shap_to_original,
    make_lr_predictor,
    make_mlp_predictor,
    build_background_dataset,
)
from src.inference import EXPECTED_FEATURES


@pytest.mark.unit
class TestExplainabilityUnit:
    """Unit tests for feature grouping, aggregation, and prediction wrappers."""

    def test_build_feature_groups_coverage(self, feature_mapping):
        """All 46 encoded features must be mapped into groups corresponding to original attributes."""
        groups = build_feature_groups(feature_mapping)
        assert isinstance(groups, dict)

        # Check total number of column indices mapped equals 46
        all_indices = []
        for indices in groups.values():
            all_indices.extend(indices)

        assert len(all_indices) == 46
        assert sorted(all_indices) == list(range(46))
        assert "tenure" in groups
        assert "Contract" in groups
        assert "PaymentMethod" in groups

    def test_aggregate_shap_to_original(self, feature_mapping):
        """aggregate_shap_to_original must correctly average SHAP magnitudes for categorical groups."""
        groups = build_feature_groups(feature_mapping)
        np.random.seed(42)
        # Synthetic SHAP values (10 records, 46 features)
        shap_dummy = np.random.randn(10, 46)

        df_agg = aggregate_shap_to_original(shap_dummy, groups)

        assert isinstance(df_agg, pd.DataFrame)
        assert set(df_agg.columns) == {"original_feature", "mean_abs_shap", "n_encoded_cols"}
        assert len(df_agg) == len(groups)
        assert df_agg["mean_abs_shap"].is_monotonic_decreasing
        assert (df_agg["mean_abs_shap"] >= 0.0).all()

    def test_make_lr_predictor_contract(self, baseline_logistic_model, synthetic_encoded_features):
        """LR predictor wrapper must output 1D array of probabilities in [0, 1]."""
        predict_fn = make_lr_predictor(baseline_logistic_model)
        probs = predict_fn(synthetic_encoded_features)

        assert isinstance(probs, np.ndarray)
        assert probs.ndim == 1
        assert len(probs) == len(synthetic_encoded_features)
        assert np.all((probs >= 0.0) & (probs <= 1.0))

    def test_make_mlp_predictor_contract(self, mlp_model, synthetic_encoded_features):
        """MLP predictor wrapper must output 1D array of probabilities in [0, 1]."""
        predict_fn = make_mlp_predictor(mlp_model)
        probs = predict_fn(synthetic_encoded_features)

        assert isinstance(probs, np.ndarray)
        assert probs.ndim == 1
        assert len(probs) == len(synthetic_encoded_features)
        assert np.all((probs >= 0.0) & (probs <= 1.0))


@pytest.mark.integration
class TestExplainabilityIntegration:
    """Integration tests verifying saved SHAP artifacts and background dataset."""

    def test_shap_background_dataset_properties(self, models_dir):
        """Saved background dataset must be (75, 46) float32 array with no NaN or Inf."""
        bg_path = models_dir / "shap_background.joblib"
        assert bg_path.exists(), "shap_background.joblib missing"

        bg = joblib.load(bg_path)
        assert isinstance(bg, np.ndarray)
        assert bg.shape == (75, 46)
        assert bg.dtype in (np.float32, np.float64)
        assert np.all(np.isfinite(bg))

    def test_shap_global_importance_csv_structure(self, reports_dir):
        """shap_global_importance.csv must contain importance scores for both LR and MLP."""
        csv_path = reports_dir / "shap_global_importance.csv"
        assert csv_path.exists(), "shap_global_importance.csv missing"

        df = pd.read_csv(csv_path)
        assert set(df.columns) == {"original_feature", "mean_abs_shap", "n_encoded_cols", "model"}
        assert set(df["model"].unique()) == {"LogisticRegression", "MLP"}
        # Both models should have all 19 original attributes evaluated
        assert len(df[df["model"] == "LogisticRegression"]) == 19
        assert len(df[df["model"] == "MLP"]) == 19

    def test_shap_local_explanations_json_cases(self, reports_dir):
        """shap_local_explanations.json must contain TP, TN, FP, FN cases for both models."""
        json_path = reports_dir / "shap_local_explanations.json"
        assert json_path.exists(), "shap_local_explanations.json missing"

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "logistic_regression" in data
        assert "mlp" in data

        expected_cases = {"tp", "tn", "fp", "fn"}
        assert set(data["logistic_regression"].keys()) == expected_cases
        assert set(data["mlp"].keys()) == expected_cases

    def test_shap_figure_artifacts_exist(self, figures_dir):
        """All required SHAP summary, importance, and waterfall figures must exist on disk."""
        expected_figs = [
            "shap_summary_mlp.png",
            "shap_summary_logistic.png",
            "shap_importance_mlp.png",
            "shap_importance_logistic.png",
            "shap_local_mlp_tp.png",
            "shap_local_mlp_tn.png",
            "shap_local_mlp_fp.png",
            "shap_local_mlp_fn.png",
            "shap_local_lr_tp.png",
            "shap_local_lr_tn.png",
            "shap_local_lr_fp.png",
            "shap_local_lr_fn.png",
        ]
        for fname in expected_figs:
            fig_path = figures_dir / fname
            assert fig_path.exists(), f"Missing SHAP figure: {fname}"
            assert fig_path.stat().st_size > 0, f"Empty SHAP figure: {fname}"
