"""Configuration module for ChurnIQ.

Defines all file paths, reproducibility seeds, hyperparameters,
risk thresholds, and operational settings across the project.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Dict, List, Optional


# ==============================================================================
# PROJECT METADATA
# ==============================================================================
PROJECT_NAME: str = "ChurnIQ"
PROJECT_DESCRIPTION: str = (
    "AI-Powered Telecom Customer Churn Prediction System featuring Multilayer "
    "Perceptrons (MLP), Baseline Logistic Regression, Explainable AI (SHAP), "
    "and an Interactive Streamlit Dashboard."
)
PROJECT_VERSION: str = "1.0.0"


# Set Keras 3 backend default to PyTorch
os.environ.setdefault("KERAS_BACKEND", "torch")

# ==============================================================================
# REPRODUCIBILITY CONFIGURATION
# ==============================================================================
RANDOM_SEED: int = 42


def set_all_seeds(seed: int = RANDOM_SEED) -> None:
    """Set global random seeds for deterministic execution where possible.

    Note:
        Bitwise exact reproducibility across differing hardware architectures,
        operating systems, or CUDA/cuDNN environments is mathematically not
        guaranteed due to floating-point reduction non-determinism.
    """
    import os
    import random

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    try:
        import keras

        keras.utils.set_random_seed(seed)
    except Exception:
        pass


# ==============================================================================
# DIRECTORY PATHS (Using pathlib.Path)
# ==============================================================================
ROOT_DIR: Path = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR: Path = ROOT_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

# Artifact and report directories
MODELS_DIR: Path = ROOT_DIR / "models"
NOTEBOOKS_DIR: Path = ROOT_DIR / "notebooks"
REPORTS_DIR: Path = ROOT_DIR / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
APP_DIR: Path = ROOT_DIR / "app"

# Ensure runtime directories exist
for directory in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    NOTEBOOKS_DIR,
    FIGURES_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# DATASET & PARTITION PATHS
# ==============================================================================
# The official IBM Telco dataset filename
DEFAULT_RAW_FILENAME: str = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
RAW_DATA_FILE: Path = RAW_DATA_DIR / DEFAULT_RAW_FILENAME
if not RAW_DATA_FILE.exists():
    _fallback_file = RAW_DATA_DIR / "Telco-Customer-Churn.csv"
    if _fallback_file.exists():
        RAW_DATA_FILE = _fallback_file

TRAIN_DATA_FILE: Path = PROCESSED_DATA_DIR / "train_split.parquet"
VAL_DATA_FILE: Path = PROCESSED_DATA_DIR / "val_split.parquet"
TEST_DATA_FILE: Path = PROCESSED_DATA_DIR / "test_split.parquet"

# Data schema definitions
TARGET_COLUMN: str = "Churn"
ID_COLUMN: str = "customerID"
POSITIVE_CLASS_LABEL: str = "Yes"
NEGATIVE_CLASS_LABEL: str = "No"


# ==============================================================================
# MODEL ARTIFACT PATHS
# ==============================================================================
PREPROCESSOR_PATH: Path = MODELS_DIR / "preprocessor.joblib"
FEATURE_MAPPING_PATH: Path = MODELS_DIR / "feature_mapping.json"
BASELINE_MODEL_PATH: Path = MODELS_DIR / "baseline_logistic.joblib"
MLP_MODEL_PATH: Path = MODELS_DIR / "mlp_churn_model.keras"
SHAP_BACKGROUND_PATH: Path = MODELS_DIR / "shap_background.joblib"
THRESHOLD_METADATA_PATH: Path = MODELS_DIR / "threshold_metadata.json"


# ==============================================================================
# DATA SPLIT CONFIGURATION
# ==============================================================================
@dataclass(frozen=True)
class DataSplitConfig:
    """Stratified dataset partition configuration."""

    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    stratify: bool = True
    random_state: int = RANDOM_SEED

    def __post_init__(self) -> None:
        total = round(self.train_ratio + self.val_ratio + self.test_ratio, 4)
        if total != 1.0:
            raise ValueError(
                f"Split ratios must sum to 1.0 (got {total})"
            )


# ==============================================================================
# MLP NEURAL NETWORK HYPERPARAMETERS
# ==============================================================================
@dataclass
class MLPHyperparameters:
    """Multilayer Perceptron architecture and optimization parameters."""

    hidden_units: List[int] = field(default_factory=lambda: [64, 32, 16])
    hidden_activation: str = "relu"
    output_activation: str = "sigmoid"
    dropout_rate: float = 0.25
    l2_regularization: float = 0.001
    learning_rate: float = 0.001
    optimizer: str = "adam"
    loss_function: str = "binary_crossentropy"
    batch_size: int = 32
    max_epochs: int = 100
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5
    min_learning_rate: float = 1e-6


# ==============================================================================
# BASELINE MODEL HYPERPARAMETERS
# ==============================================================================
@dataclass
class BaselineHyperparameters:
    """Logistic Regression baseline parameters."""

    max_iter: int = 1000
    c_regularization: float = 1.0
    solver: str = "lbfgs"
    class_weight: str = "balanced"
    random_state: int = RANDOM_SEED


# ==============================================================================
# DECOUPLED RISK CLASSIFICATION THRESHOLDS
# ==============================================================================
@dataclass
class RiskThresholdsConfig:
    """Three-level customer churn risk stratification boundaries.

    Business Rationale:
    - Low Risk (P < low_threshold):
      Churn probability is below normal baseline churn prevalence.
      Retention intervention has negative expected ROI.
    - Medium Risk (low_threshold <= P < high_threshold):
      Borderline customers demonstrating churn propensity signals.
      Cost-effective for automated, low-cost retention nudges.
    - High Risk (P >= high_threshold):
      High-confidence churn candidates. Justifies expensive,
      high-touch interventions (e.g., promotional discounts, direct outreach).
    """

    low_threshold: float = 0.30
    high_threshold: float = 0.70

    def __post_init__(self) -> None:
        if not (0.0 < self.low_threshold < self.high_threshold < 1.0):
            raise ValueError(
                "Risk thresholds must satisfy: 0.0 < low_threshold < high_threshold < 1.0"
            )


# ==============================================================================
# SHAP EXPLAINABILITY CONFIGURATION
# ==============================================================================
@dataclass
class SHAPConfig:
    """Configuration for Explainable AI (SHAP) reference sampling."""

    # Using representative actual training records rather than synthetic K-means centroids
    sample_size: int = 75
    stratified: bool = True
    random_state: int = RANDOM_SEED


# Default instances for direct import across modules
DEFAULT_SPLIT_CONFIG = DataSplitConfig()
DEFAULT_MLP_CONFIG = MLPHyperparameters()
DEFAULT_BASELINE_CONFIG = BaselineHyperparameters()
DEFAULT_RISK_CONFIG = RiskThresholdsConfig()
DEFAULT_SHAP_CONFIG = SHAPConfig()

