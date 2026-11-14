"""Shared pytest configuration and fixtures for ChurnIQ testing suite."""

import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import json
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import joblib

from src.config import (
    ROOT_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    DEFAULT_RAW_FILENAME,
)
from src.inference import EXPECTED_FEATURES, RISK_CFG


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT_DIR


@pytest.fixture(scope="session")
def models_dir() -> Path:
    return MODELS_DIR


@pytest.fixture(scope="session")
def reports_dir() -> Path:
    return REPORTS_DIR


@pytest.fixture(scope="session")
def figures_dir() -> Path:
    return FIGURES_DIR


@pytest.fixture(scope="session")
def processed_data_dir() -> Path:
    return PROCESSED_DATA_DIR


@pytest.fixture(scope="session")
def expected_feature_names() -> list[str]:
    return list(EXPECTED_FEATURES)


@pytest.fixture(scope="session")
def preprocessor():
    path = MODELS_DIR / "preprocessor.joblib"
    assert path.exists(), f"Preprocessor artifact not found: {path}"
    return joblib.load(path)


@pytest.fixture(scope="session")
def baseline_logistic_model():
    path = MODELS_DIR / "baseline_logistic.joblib"
    assert path.exists(), f"Baseline model artifact not found: {path}"
    return joblib.load(path)


@pytest.fixture(scope="session")
def mlp_model():
    import keras
    path = MODELS_DIR / "mlp_churn_model.keras"
    assert path.exists(), f"MLP model artifact not found: {path}"
    return keras.models.load_model(str(path))


@pytest.fixture(scope="session")
def threshold_metadata() -> dict:
    path = MODELS_DIR / "threshold_metadata.json"
    assert path.exists(), f"Threshold metadata not found: {path}"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def feature_mapping() -> dict:
    path = MODELS_DIR / "feature_mapping.json"
    assert path.exists(), f"Feature mapping artifact not found: {path}"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def synthetic_raw_df() -> pd.DataFrame:
    """Create a minimal synthetic raw Telco DataFrame representing valid and edge cases."""
    data = {
        "customerID": ["TEST-001", "TEST-002", "TEST-003", "TEST-004", "TEST-005"],
        "gender": ["Female", "Male", "Female", "Male", "Female"],
        "SeniorCitizen": [0, 1, 0, 0, 1],
        "Partner": ["Yes", "No", "No", "Yes", "No"],
        "Dependents": ["No", "No", "Yes", "No", "No"],
        "tenure": [1, 24, 0, 72, 12],
        "PhoneService": ["No", "Yes", "Yes", "Yes", "Yes"],
        "MultipleLines": ["No phone service", "Yes", "No", "Yes", "No"],
        "InternetService": ["DSL", "Fiber optic", "DSL", "No", "Fiber optic"],
        "OnlineSecurity": ["No", "Yes", "Yes", "No internet service", "No"],
        "OnlineBackup": ["Yes", "No", "No", "No internet service", "Yes"],
        "DeviceProtection": ["No", "Yes", "No", "No internet service", "Yes"],
        "TechSupport": ["No", "No", "Yes", "No internet service", "No"],
        "StreamingTV": ["No", "Yes", "No", "No internet service", "Yes"],
        "StreamingMovies": ["No", "Yes", "No", "No internet service", "Yes"],
        "Contract": ["Month-to-month", "One year", "Two year", "Two year", "Month-to-month"],
        "PaperlessBilling": ["Yes", "No", "Yes", "No", "Yes"],
        "PaymentMethod": [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
            "Electronic check",
        ],
        "MonthlyCharges": [29.85, 89.10, 55.20, 25.00, 99.90],
        "TotalCharges": ["29.85", "2138.4", " ", "1800.0", "1198.8"],  # TEST-003 has whitespace with tenure=0
        "Churn": ["No", "Yes", "No", "No", "Yes"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_customer_raw_dict() -> dict:
    """A realistic single customer dictionary for inference testing."""
    return {
        "customerID": "CUST-9999",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 79.5,
        "TotalCharges": 397.5,
    }


@pytest.fixture
def synthetic_encoded_features(expected_feature_names) -> np.ndarray:
    """Synthetic encoded float32 feature array of shape (5, 46)."""
    np.random.seed(42)
    return np.random.randn(5, len(expected_feature_names)).astype(np.float32)
