"""Unit and integration tests for src/neural_network.py."""

import json
import pytest
import numpy as np
import pandas as pd
import keras

from src.neural_network import (
    build_mlp_model,
    evaluate_mlp_validation,
    verify_backend,
    compute_class_weights,
)
from src.config import MLPHyperparameters
from src.inference import EXPECTED_FEATURES


@pytest.mark.unit
class TestNeuralNetworkUnit:
    """Unit tests for MLP architecture construction, backend verification, and evaluation."""

    def test_verify_backend_torch(self):
        """Keras backend must be torch."""
        verify_backend()
        assert keras.backend.backend() == "torch"

    def test_build_mlp_model_architecture(self):
        """Model architecture must match specified layer count, units, and activations."""
        cfg = MLPHyperparameters(
            hidden_units=[64, 32, 16],
            hidden_activation="relu",
            output_activation="sigmoid",
            dropout_rate=0.25,
            l2_regularization=0.001,
        )
        model = build_mlp_model(n_features=46, config=cfg)

        assert model.input_shape == (None, 46)
        assert model.output_shape == (None, 1)

        # Layers: Input (implicit/explicit), Dense(64), Dropout(0.25), Dense(32), Dropout(0.25), Dense(16), Dense(1)
        dense_layers = [l for l in model.layers if isinstance(l, keras.layers.Dense)]
        dropout_layers = [l for l in model.layers if isinstance(l, keras.layers.Dropout)]

        assert len(dense_layers) == 4  # 64, 32, 16, 1
        assert [l.units for l in dense_layers] == [64, 32, 16, 1]
        assert len(dropout_layers) == 2

    def test_compute_class_weights(self):
        """Class weights should balance minority (positive) class."""
        y_dummy = np.array([0, 0, 0, 1])  # 75% 0, 25% 1
        cw = compute_class_weights(y_dummy)
        assert 0 in cw and 1 in cw
        assert cw[1] > cw[0]  # Minority class has higher weight
        assert round(cw[0] * 3, 2) == round(cw[1], 2)

    def test_evaluate_mlp_validation_contract(self, mlp_model, synthetic_encoded_features):
        """evaluate_mlp_validation must return properly formatted metrics and valid probabilities."""
        n = synthetic_encoded_features.shape[0]
        y_dummy = np.array([[1], [0], [1], [0], [0]], dtype=np.float32)

        metrics, proba, pred = evaluate_mlp_validation(
            mlp_model, synthetic_encoded_features, y_dummy, threshold=0.50
        )

        assert isinstance(metrics, dict)
        assert "roc_auc" in metrics
        assert "pr_auc" in metrics
        assert "f1" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "confusion_matrix" in metrics
        assert len(proba) == n
        assert len(pred) == n
        assert np.all((proba >= 0.0) & (proba <= 1.0))
        assert set(np.unique(pred)).issubset({0, 1})

    def test_invalid_input_dimension_raises(self, mlp_model):
        """Feeding incorrect feature dimension into MLP must raise an error."""
        X_bad = np.random.randn(5, 20).astype(np.float32)
        with pytest.raises(Exception):
            mlp_model.predict(X_bad, verbose=0)


@pytest.mark.integration
@pytest.mark.slow
class TestNeuralNetworkIntegration:
    """Integration tests verifying saved MLP model and training history."""

    def test_saved_mlp_model_properties(self, mlp_model):
        """Saved MLP model must load with PyTorch backend and accept (None, 46)."""
        assert mlp_model is not None
        assert mlp_model.input_shape == (None, 46)
        assert mlp_model.output_shape == (None, 1)

    def test_saved_mlp_validation_metrics_match_report(
        self, mlp_model, processed_data_dir, reports_dir
    ):
        """MLP predictions on val partition must match saved validation report."""
        val_df = pd.read_parquet(processed_data_dir / "val_split.parquet")
        X_val = val_df[EXPECTED_FEATURES].values.astype(np.float32)
        y_val = val_df["Churn"].values.astype(np.float32).reshape(-1, 1)

        metrics, proba, pred = evaluate_mlp_validation(mlp_model, X_val, y_val, threshold=0.50)

        report_path = reports_dir / "mlp_validation_metrics.json"
        assert report_path.exists(), "mlp_validation_metrics.json missing"
        with open(report_path, "r", encoding="utf-8") as f:
            saved = json.load(f)

        assert abs(metrics["roc_auc"] - saved["roc_auc"]) <= 1e-4
        assert abs(metrics["pr_auc"] - saved["pr_auc"]) <= 1e-4
        assert abs(metrics["f1"] - saved["f1"]) <= 1e-4

    def test_training_history_json_integrity(self, reports_dir):
        """Training history must contain loss and accuracy histories for training and validation."""
        hist_path = reports_dir / "mlp_training_history.json"
        assert hist_path.exists(), "mlp_training_history.json missing"
        with open(hist_path, "r", encoding="utf-8") as f:
            hist = json.load(f)

        for key in ("loss", "val_loss", "accuracy", "val_accuracy"):
            assert key in hist
            assert len(hist[key]) > 0
            assert all(isinstance(v, (int, float)) for v in hist[key])
        # Ensure training stopped within max_epochs
        assert len(hist["loss"]) <= 100
