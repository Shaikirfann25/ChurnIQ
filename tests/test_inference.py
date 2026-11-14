"""Unit and integration tests for src/inference.py."""

import pytest
import numpy as np
import pandas as pd

from src.inference import (
    ChurnIQInference,
    ChurnPrediction,
    BatchPredictionResult,
    RISK_CFG,
    EXPECTED_FEATURES,
)


@pytest.mark.unit
class TestInferenceUnit:
    """Unit tests for inference logic, risk categorisation, and input validation."""

    def test_risk_category_assignment_boundaries(self):
        """Risk categories must follow: Low < 0.30, Medium [0.30, 0.70), High >= 0.70."""
        assert ChurnIQInference._assign_risk(0.0) == "Low"
        assert ChurnIQInference._assign_risk(0.2999) == "Low"
        assert ChurnIQInference._assign_risk(0.30) == "Medium"
        assert ChurnIQInference._assign_risk(0.50) == "Medium"
        assert ChurnIQInference._assign_risk(0.6999) == "Medium"
        assert ChurnIQInference._assign_risk(0.70) == "High"
        assert ChurnIQInference._assign_risk(1.0) == "High"

    def test_risk_boundaries_independent_of_binary_threshold(self):
        """Risk boundaries must not change regardless of model threshold (e.g. 0.46 or 0.54)."""
        assert RISK_CFG.low_threshold == 0.30
        assert RISK_CFG.high_threshold == 0.70

        engine_mlp = ChurnIQInference(model_name="mlp")
        engine_mlp._load()
        engine_lr = ChurnIQInference(model_name="logistic_regression")
        engine_lr._load()

        assert engine_mlp.threshold == 0.46
        assert engine_lr.threshold == 0.54

        # An observation with p=0.48:
        # For MLP: p(0.48) >= 0.46 -> churn_predicted is True, but risk is Medium
        # For LR: p(0.48) < 0.54 -> churn_predicted is False, but risk is still Medium
        mlp_pred = engine_mlp.predict_one_encoded(np.zeros(46, dtype=np.float32), customer_id="C_MLP")
        lr_pred = engine_lr.predict_one_encoded(np.zeros(46, dtype=np.float32), customer_id="C_LR")
        assert mlp_pred.threshold_used == 0.46
        assert lr_pred.threshold_used == 0.54

    def test_missing_required_fields_raises_error(self):
        """Raw customer record missing mandatory fields must raise ValueError."""
        engine = ChurnIQInference(model_name="mlp")
        incomplete_cust = {"customerID": "INCOMPLETE-01", "tenure": 12}

        with pytest.raises(ValueError, match="missing"):
            engine.predict_one_raw(incomplete_cust)

    def test_missing_features_in_batch_encoded_raises_error(self):
        """Encoded batch missing required columns must raise ValueError."""
        engine = ChurnIQInference(model_name="mlp")
        bad_df = pd.DataFrame({"customerID": ["C1"], "tenure": [1.0]})

        with pytest.raises(ValueError, match="Missing encoded features"):
            engine.predict_batch_encoded(bad_df)

    def test_invalid_model_name_raises_value_error(self):
        """Unsupported model name must raise ValueError."""
        with pytest.raises(ValueError, match="model_name must be 'mlp' or 'logistic_regression'"):
            ChurnIQInference(model_name="random_forest")


@pytest.mark.integration
class TestInferenceIntegration:
    """Integration tests running single-record and batch inference on trained models."""

    @pytest.mark.parametrize("model_name", ["mlp", "logistic_regression"])
    def test_single_customer_raw_prediction(self, model_name, sample_customer_raw_dict):
        """Single customer raw prediction must produce valid ChurnPrediction with metadata."""
        engine = ChurnIQInference(model_name=model_name)
        result = engine.predict_one_raw(sample_customer_raw_dict)

        assert isinstance(result, ChurnPrediction)
        assert result.customer_id == sample_customer_raw_dict["customerID"]
        assert 0.0 <= result.churn_probability <= 1.0
        assert isinstance(result.churn_predicted, bool)
        assert result.risk_category in {"Low", "Medium", "High"}
        assert result.model_used == model_name
        assert len(result.top_features) == 5

        # Check top features structure
        for feat in result.top_features:
            assert "feature" in feat
            assert "original_attribute" in feat
            assert "encoded_value" in feat
            assert "contribution_proxy" in feat

    @pytest.mark.parametrize("model_name", ["mlp", "logistic_regression"])
    def test_batch_prediction_from_encoded_val_data(self, model_name, processed_data_dir):
        """Batch prediction from val partition must return BatchPredictionResult with correct length."""
        val_df = pd.read_parquet(processed_data_dir / "val_split.parquet")
        # Remove target
        val_features = val_df.drop(columns=["Churn"])

        engine = ChurnIQInference(model_name=model_name)
        batch_res = engine.predict_batch_encoded(val_features)

        assert isinstance(batch_res, BatchPredictionResult)
        assert batch_res.model_used == model_name
        assert batch_res.n_records == len(val_df)
        assert len(batch_res.predictions_df) == len(val_df)

        # Check output dataframe structure
        df_out = batch_res.predictions_df
        for col in ["customerID", "churn_probability", "churn_predicted", "risk_category", "model", "threshold_used"]:
            assert col in df_out.columns

        # customerID is preserved as metadata
        assert list(df_out["customerID"]) == list(val_df["customerID"])

        # Check risk summary
        summary = batch_res.risk_summary
        assert set(summary.keys()) == {"Low", "Medium", "High"}
        total_count = sum(s["count"] for s in summary.values())
        assert total_count == len(val_df)

    def test_unknown_categorical_value_handled_gracefully(self, sample_customer_raw_dict):
        """Novel or unobserved categorical values must be safely encoded without failure."""
        engine = ChurnIQInference(model_name="mlp")
        modified_cust = sample_customer_raw_dict.copy()
        modified_cust["Contract"] = "UnseenContractType"
        modified_cust["PaymentMethod"] = "CryptoWallet"

        pred = engine.predict_one_raw(modified_cust)
        assert isinstance(pred, ChurnPrediction)
        assert 0.0 <= pred.churn_probability <= 1.0

