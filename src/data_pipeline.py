"""Data pipeline module for ChurnIQ.

Handles dataset loading, validation, leakage-free stratified splitting,
transformer fitting on training data, feature mapping generation,
and artifact serialization.
"""

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    DEFAULT_RAW_FILENAME,
    DEFAULT_SPLIT_CONFIG,
    FEATURE_MAPPING_PATH,
    ID_COLUMN,
    NEGATIVE_CLASS_LABEL,
    POSITIVE_CLASS_LABEL,
    PREPROCESSOR_PATH,
    RANDOM_SEED,
    RAW_DATA_FILE,
    TARGET_COLUMN,
    TEST_DATA_FILE,
    TRAIN_DATA_FILE,
    VAL_DATA_FILE,
    DataSplitConfig,
    set_all_seeds,
)

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ChurnIQ.DataPipeline")


# ==============================================================================
# EXPECTED SCHEMA CONSTANTS
# ==============================================================================
EXPECTED_NUMERICAL_COLS: List[str] = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

EXPECTED_CATEGORICAL_COLS: List[str] = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

ALL_REQUIRED_COLS: List[str] = (
    [ID_COLUMN]
    + EXPECTED_NUMERICAL_COLS
    + EXPECTED_CATEGORICAL_COLS
    + [TARGET_COLUMN]
)


@dataclass
class ProcessedSplits:
    """Holds partitioned and transformed DataFrames with metadata."""

    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    feature_names: List[str]
    feature_mapping: Dict[str, str]
    preprocessor: ColumnTransformer


# ==============================================================================
# 1. DATA LOADING
# ==============================================================================
def load_raw_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """Load the raw telecom churn CSV dataset and validate required columns.

    Args:
        filepath: Optional path to the CSV file. If None, uses RAW_DATA_FILE.

    Returns:
        pd.DataFrame: Loaded raw dataset.

    Raises:
        FileNotFoundError: If the specified raw data file does not exist.
        ValueError: If required columns are missing from the dataset.
    """
    path = filepath or RAW_DATA_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at '{path}'. Please ensure the IBM Telco "
            f"dataset is placed in the designated raw data folder."
        )

    logger.info(f"Loading raw dataset from: {path}")
    df = pd.read_csv(path)
    logger.info(f"Dataset successfully loaded with shape: {df.shape}")

    missing_cols = [c for c in ALL_REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Raw dataset is missing required columns: {missing_cols}"
        )

    return df


# ==============================================================================
# 2. DATA CLEANING & TYPE SANITIZATION
# ==============================================================================
def clean_dataset(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Clean and sanitize the raw dataset without modifying the source.

    Specific operations:
    1. Cast TotalCharges to numeric.
    2. Replace whitespace strings in TotalCharges with 0.0 for tenure == 0 records.
    3. Verify that no invalid or unparseable values remain in TotalCharges.
    4. Map binary Churn labels: 'No' -> 0, 'Yes' -> 1.
    5. Separate customerID as a non-predictive metadata Series.

    Args:
        df: Raw DataFrame.

    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.Series]:
            - X: Predictive features DataFrame (customerID and Churn excluded).
            - y: Binary integer target Series (0 or 1).
            - customer_ids: Series of customerID metadata.

    Raises:
        ValueError: If unparseable TotalCharges or unexpected Churn values remain.
    """
    logger.info("Initiating dataset cleaning and type sanitization...")
    data = df.copy()

    # 1. Handle TotalCharges whitespace anomaly
    tc_str = data["TotalCharges"].astype(str).str.strip()
    blank_mask = tc_str == ""
    blank_count = blank_mask.sum()

    if blank_count > 0:
        logger.info(
            f"Detected {blank_count} blank/whitespace entries in 'TotalCharges'."
        )
        # Verify tenure is 0 for these records
        tenure_for_blanks = data.loc[blank_mask, "tenure"]
        non_zero_tenure_blanks = (tenure_for_blanks != 0).sum()
        if non_zero_tenure_blanks > 0:
            raise ValueError(
                f"Found {non_zero_tenure_blanks} blank TotalCharges entries "
                f"where tenure != 0. Requires manual inspection."
            )
        # Impute 0.0 for brand new customers (tenure == 0)
        data.loc[blank_mask, "TotalCharges"] = "0.0"

    # Convert TotalCharges to float64
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")

    # Final check on TotalCharges nulls
    remaining_nulls = data["TotalCharges"].isna().sum()
    if remaining_nulls > 0:
        raise ValueError(
            f"Failed to parse {remaining_nulls} TotalCharges values to numeric."
        )

    # 2. Map Churn target: 'No' -> 0, 'Yes' -> 1
    unique_targets = set(data[TARGET_COLUMN].dropna().unique())
    expected_targets = {NEGATIVE_CLASS_LABEL, POSITIVE_CLASS_LABEL}
    if not unique_targets.issubset(expected_targets):
        raise ValueError(
            f"Unexpected values in '{TARGET_COLUMN}': {unique_targets - expected_targets}"
        )

    target_mapping = {NEGATIVE_CLASS_LABEL: 0, POSITIVE_CLASS_LABEL: 1}
    y = data[TARGET_COLUMN].map(target_mapping).astype(int)

    # 3. Extract customerID metadata and drop non-features
    customer_ids = data[ID_COLUMN].copy()
    X = data.drop(columns=[ID_COLUMN, TARGET_COLUMN])

    logger.info(
        f"Cleaning complete. Features shape: {X.shape}, Target shape: {y.shape}"
    )
    return X, y, customer_ids


# ==============================================================================
# 3. DATA SPLITTING (3-Way Stratified Partition)
# ==============================================================================
def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    customer_ids: pd.Series,
    split_config: DataSplitConfig = DEFAULT_SPLIT_CONFIG,
) -> Tuple[
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """Perform a leakage-free, stratified 3-way partition (Train / Val / Test).

    Ratios:
    - Train: 70%
    - Validation: 15%
    - Test: 15% (quarantined)

    Preserves target proportions and aligns customerID metadata.

    Args:
        X: Feature matrix.
        y: Target Series.
        customer_ids: Customer identifier Series.
        split_config: Partition configuration dataclass.

    Returns:
        Tuple containing 9 elements:
        (X_train, y_train, ids_train,
         X_val, y_val, ids_val,
         X_test, y_test, ids_test)
    """
    logger.info(
        f"Partitioning data: Train={split_config.train_ratio:.0%}, "
        f"Val={split_config.val_ratio:.0%}, Test={split_config.test_ratio:.0%} "
        f"(Stratified={split_config.stratify}, Seed={split_config.random_state})"
    )

    stratify_first = y if split_config.stratify else None
    temp_ratio = split_config.val_ratio + split_config.test_ratio

    # Step 1: Split Train (70%) vs Temporary (30%)
    X_train, X_temp, y_train, y_temp, ids_train, ids_temp = train_test_split(
        X,
        y,
        customer_ids,
        test_size=temp_ratio,
        random_state=split_config.random_state,
        stratify=stratify_first,
    )

    # Step 2: Split Temporary (30%) into Val (15%) and Test (15%) -> 50/50 of temp
    stratify_temp = y_temp if split_config.stratify else None
    val_share_of_temp = split_config.val_ratio / temp_ratio

    X_val, X_test, y_val, y_test, ids_val, ids_test = train_test_split(
        X_temp,
        y_temp,
        ids_temp,
        test_size=(1.0 - val_share_of_temp),
        random_state=split_config.random_state,
        stratify=stratify_temp,
    )

    logger.info(
        f"Splits successfully created: Train={X_train.shape[0]}, "
        f"Val={X_val.shape[0]}, Test={X_test.shape[0]}"
    )

    return (
        X_train,
        y_train,
        ids_train,
        X_val,
        y_val,
        ids_val,
        X_test,
        y_test,
        ids_test,
    )


# ==============================================================================
# 4. PREPROCESSING PIPELINE & FEATURE MAPPING
# ==============================================================================
def build_preprocessor(
    numerical_cols: List[str] = EXPECTED_NUMERICAL_COLS,
    categorical_cols: List[str] = EXPECTED_CATEGORICAL_COLS,
) -> ColumnTransformer:
    """Build a Scikit-Learn ColumnTransformer for numerical and categorical data.

    - Numerical: StandardScaler
    - Categorical: OneHotEncoder (handle_unknown='ignore', sparse_output=False)

    Args:
        numerical_cols: List of numerical column names.
        categorical_cols: List of categorical column names.

    Returns:
        ColumnTransformer: Unfitted preprocessor pipeline.
    """
    num_transformer = StandardScaler()
    cat_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, numerical_cols),
            ("cat", cat_transformer, categorical_cols),
        ],
        verbose_feature_names_out=False,
    )
    return preprocessor


def extract_feature_mapping(
    preprocessor: ColumnTransformer,
    numerical_cols: List[str] = EXPECTED_NUMERICAL_COLS,
    categorical_cols: List[str] = EXPECTED_CATEGORICAL_COLS,
) -> Dict[str, str]:
    """Map each transformed feature name back to its parent original attribute.

    Used by SHAP explainability and dashboard visualization to provide both
    fine-grained one-hot attributions and aggregated parent-level attributions.

    Args:
        preprocessor: Fitted ColumnTransformer.
        numerical_cols: List of numerical columns.
        categorical_cols: List of categorical columns.

    Returns:
        Dict[str, str]: Map of {transformed_feature_name: original_feature_name}.
    """
    mapping: Dict[str, str] = {}

    # 1. Numerical features map directly to themselves
    for col in numerical_cols:
        mapping[col] = col

    # 2. Categorical features map from one-hot dummy name to original column
    ohe: OneHotEncoder = preprocessor.named_transformers_["cat"]
    for orig_col, categories in zip(categorical_cols, ohe.categories_):
        for cat in categories:
            dummy_name = f"{orig_col}_{cat}"
            mapping[dummy_name] = orig_col

    return mapping


def fit_and_transform_splits(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    ids_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    ids_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    ids_test: pd.Series,
    numerical_cols: List[str] = EXPECTED_NUMERICAL_COLS,
    categorical_cols: List[str] = EXPECTED_CATEGORICAL_COLS,
) -> ProcessedSplits:
    """Fit ColumnTransformer ONLY on X_train and transform all partitions.

    Ensures zero data leakage.

    Args:
        X_train, y_train, ids_train: Training partition.
        X_val, y_val, ids_val: Validation partition.
        X_test, y_test, ids_test: Quarantined test partition.
        numerical_cols: Numerical column list.
        categorical_cols: Categorical column list.

    Returns:
        ProcessedSplits: Dataclass containing transformed DataFrames,
        feature names, mapping, and fitted preprocessor.
    """
    logger.info("Fitting ColumnTransformer STRICTLY on training data...")
    preprocessor = build_preprocessor(numerical_cols, categorical_cols)

    # FIT ONLY ON TRAINING SET
    preprocessor.fit(X_train)

    feature_names = list(preprocessor.get_feature_names_out())
    logger.info(
        f"Preprocessor fitted. Total transformed features: {len(feature_names)}"
    )

    feature_mapping = extract_feature_mapping(
        preprocessor, numerical_cols, categorical_cols
    )

    def _assemble_split_df(
        X_split: pd.DataFrame, y_split: pd.Series, ids_split: pd.Series
    ) -> pd.DataFrame:
        """Transform features and pack with customerID and Churn."""
        arr = preprocessor.transform(X_split)
        split_df = pd.DataFrame(
            arr, columns=feature_names, index=X_split.index
        )
        # Prepend customerID and append Churn target
        split_df.insert(0, ID_COLUMN, ids_split.values)
        split_df[TARGET_COLUMN] = y_split.values
        return split_df

    logger.info("Transforming train, validation, and test partitions...")
    train_df = _assemble_split_df(X_train, y_train, ids_train)
    val_df = _assemble_split_df(X_val, y_val, ids_val)
    test_df = _assemble_split_df(X_test, y_test, ids_test)

    return ProcessedSplits(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_names=feature_names,
        feature_mapping=feature_mapping,
        preprocessor=preprocessor,
    )


# ==============================================================================
# 5. ARTIFACT PERSISTENCE
# ==============================================================================
def save_artifacts(
    processed_splits: ProcessedSplits,
    preprocessor_path: Path = PREPROCESSOR_PATH,
    feature_mapping_path: Path = FEATURE_MAPPING_PATH,
    train_path: Path = TRAIN_DATA_FILE,
    val_path: Path = VAL_DATA_FILE,
    test_path: Path = TEST_DATA_FILE,
) -> None:
    """Save the preprocessor, feature mapping, and partitioned parquets to disk.

    Args:
        processed_splits: ProcessedSplits container.
        preprocessor_path: Destination for preprocessor joblib.
        feature_mapping_path: Destination for feature_mapping.json.
        train_path: Destination for train_split.parquet.
        val_path: Destination for val_split.parquet.
        test_path: Destination for test_split.parquet.
    """
    logger.info(f"Saving fitted preprocessor to: {preprocessor_path}")
    preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(processed_splits.preprocessor, preprocessor_path)

    logger.info(f"Saving feature mapping to: {feature_mapping_path}")
    feature_mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with open(feature_mapping_path, "w", encoding="utf-8") as f:
        json.dump(processed_splits.feature_mapping, f, indent=2)

    logger.info("Saving partitioned datasets to Parquet...")
    train_path.parent.mkdir(parents=True, exist_ok=True)
    processed_splits.train_df.to_parquet(
        train_path, index=False, engine="pyarrow"
    )
    processed_splits.val_df.to_parquet(val_path, index=False, engine="pyarrow")
    processed_splits.test_df.to_parquet(
        test_path, index=False, engine="pyarrow"
    )

    logger.info("All data pipeline artifacts saved successfully.")


# ==============================================================================
# 6. PIPELINE ORCHESTRATOR & VERIFICATION
# ==============================================================================
def run_data_pipeline(
    raw_csv_path: Optional[Path] = None,
    split_config: DataSplitConfig = DEFAULT_SPLIT_CONFIG,
) -> ProcessedSplits:
    """Execute the full end-to-end data pipeline with validation assertions.

    Args:
        raw_csv_path: Path to raw dataset CSV (optional).
        split_config: Split configuration (optional).

    Returns:
        ProcessedSplits: Resulting processed split objects.
    """
    set_all_seeds(split_config.random_state)

    # 1. Load data
    raw_df = load_raw_data(raw_csv_path)

    # 2. Clean data
    X, y, customer_ids = clean_dataset(raw_df)

    # 3. Stratified split
    (
        X_train,
        y_train,
        ids_train,
        X_val,
        y_val,
        ids_val,
        X_test,
        y_test,
        ids_test,
    ) = split_data(X, y, customer_ids, split_config)

    # 4. Fit on train ONLY and transform all partitions
    splits = fit_and_transform_splits(
        X_train,
        y_train,
        ids_train,
        X_val,
        y_val,
        ids_val,
        X_test,
        y_test,
        ids_test,
    )

    # 5. Persist artifacts
    save_artifacts(splits)

    # 6. Execute pipeline validation checks
    validate_pipeline_outputs(splits, raw_df)

    return splits


def validate_pipeline_outputs(
    splits: ProcessedSplits, raw_df: pd.DataFrame
) -> None:
    """Run comprehensive validation assertions on data pipeline outputs.

    Validates:
    - Partition file existence.
    - Total record preservation across partitions.
    - Zero missing values in transformed features.
    - Target class proportion preservation across splits.
    - Alignment of customerID metadata.
    - Strict fitting boundary (transformer contains statistics from train only).

    Args:
        splits: ProcessedSplits dataclass instance.
        raw_df: Original raw DataFrame.

    Raises:
        AssertionError: If any integrity condition is violated.
    """
    logger.info("Executing comprehensive pipeline validation assertions...")

    # A. File existence check
    for path in [
        PREPROCESSOR_PATH,
        FEATURE_MAPPING_PATH,
        TRAIN_DATA_FILE,
        VAL_DATA_FILE,
        TEST_DATA_FILE,
    ]:
        assert path.exists(), f"Expected artifact missing: {path}"

    # B. Record count preservation
    total_split_records = (
        len(splits.train_df) + len(splits.val_df) + len(splits.test_df)
    )
    assert total_split_records == len(raw_df), (
        f"Record count mismatch! Raw: {len(raw_df)}, Splits: {total_split_records}"
    )

    # C. Missing value checks in transformed features
    for name, df in [
        ("Train", splits.train_df),
        ("Val", splits.val_df),
        ("Test", splits.test_df),
    ]:
        feat_cols = [c for c in df.columns if c not in [ID_COLUMN, TARGET_COLUMN]]
        null_count = df[feat_cols].isna().sum().sum()
        assert null_count == 0, f"Found {null_count} nulls in {name} transformed features!"

    # D. Target class proportion preservation
    churn_rate_raw = (raw_df[TARGET_COLUMN] == POSITIVE_CLASS_LABEL).mean()
    churn_rate_train = (splits.train_df[TARGET_COLUMN] == 1).mean()
    churn_rate_val = (splits.val_df[TARGET_COLUMN] == 1).mean()
    churn_rate_test = (splits.test_df[TARGET_COLUMN] == 1).mean()

    for name, rate in [
        ("Train", churn_rate_train),
        ("Val", churn_rate_val),
        ("Test", churn_rate_test),
    ]:
        delta = abs(rate - churn_rate_raw)
        assert delta < 0.005, (
            f"Class proportion drift in {name} split! Raw: {churn_rate_raw:.4f}, "
            f"{name}: {rate:.4f} (Delta={delta:.4f})"
        )

    # E. customerID uniqueness and alignment
    all_split_ids = pd.concat(
        [
            splits.train_df[ID_COLUMN],
            splits.val_df[ID_COLUMN],
            splits.test_df[ID_COLUMN],
        ]
    )
    assert all_split_ids.nunique() == len(raw_df), (
        f"customerID duplicates or leakage detected across splits!"
    )
    assert set(all_split_ids) == set(raw_df[ID_COLUMN]), (
        f"Split customerIDs do not match raw customerIDs!"
    )

    # F. Preprocessor fit validation
    num_scaler = splits.preprocessor.named_transformers_["num"]
    assert hasattr(num_scaler, "mean_"), "StandardScaler was not fitted!"
    assert len(num_scaler.mean_) == len(EXPECTED_NUMERICAL_COLS), (
        f"Expected {len(EXPECTED_NUMERICAL_COLS)} scalar means, found {len(num_scaler.mean_)}"
    )

    logger.info("ALL VALIDATION ASSERTIONS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("CHURNIQ - EXECUTING DATA PIPELINE MODULE")
    print("=" * 70 + "\n")
    processed_results = run_data_pipeline()
    print("\n" + "=" * 70)
    print("DATA PIPELINE EXECUTION COMPLETED")
    print("=" * 70 + "\n")

