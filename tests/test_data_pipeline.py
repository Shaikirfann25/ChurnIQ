"""Unit and integration tests for src/data_pipeline.py."""

import pytest
import numpy as np
import pandas as pd

from src.data_pipeline import (
    ALL_REQUIRED_COLS,
    clean_dataset,
    split_data,
    build_preprocessor,
    fit_and_transform_splits,
    extract_feature_mapping,
    load_raw_data,
)
from src.config import DataSplitConfig
from src.inference import EXPECTED_FEATURES


@pytest.mark.unit
class TestDataPipelineUnit:
    """Fast, isolated unit tests using synthetic DataFrames."""

    def test_missing_required_column_raises_error(self, synthetic_raw_df, tmp_path):
        """Missing required column should be detected and raise ValueError."""
        df_missing = synthetic_raw_df.drop(columns=["PaymentMethod"])
        bad_csv = tmp_path / "bad.csv"
        df_missing.to_csv(bad_csv, index=False)

        with pytest.raises(ValueError, match="missing required columns"):
            load_raw_data(bad_csv)

    def test_clean_dataset_whitespace_totalcharges_tenure_zero(self, synthetic_raw_df):
        """TotalCharges with whitespace for tenure=0 should be cleanly imputed to 0.0."""
        # synthetic_raw_df has TEST-003 with tenure=0 and TotalCharges=" "
        X, y, ids = clean_dataset(synthetic_raw_df)

        # Check that TEST-003 has TotalCharges == 0.0
        idx_003 = ids[ids == "TEST-003"].index[0]
        assert X.loc[idx_003, "TotalCharges"] == 0.0
        assert pd.api.types.is_numeric_dtype(X["TotalCharges"])

    def test_clean_dataset_whitespace_totalcharges_tenure_nonzero_raises(self, synthetic_raw_df):
        """Whitespace TotalCharges for tenure != 0 must raise ValueError."""
        df_bad = synthetic_raw_df.copy()
        df_bad.loc[df_bad["tenure"] == 24, "TotalCharges"] = "   "

        with pytest.raises(ValueError, match="blank TotalCharges entries where tenure != 0"):
            clean_dataset(df_bad)

    def test_clean_dataset_unparseable_totalcharges_raises(self, synthetic_raw_df):
        """Unparseable string values in TotalCharges should raise ValueError."""
        df_bad = synthetic_raw_df.copy()
        df_bad.loc[0, "TotalCharges"] = "INVALID_CHARGES"

        with pytest.raises(ValueError, match="Failed to parse"):
            clean_dataset(df_bad)

    def test_target_encoding_and_separation(self, synthetic_raw_df):
        """Churn target must be binary encoded ('No'->0, 'Yes'->1) and removed from features."""
        X, y, ids = clean_dataset(synthetic_raw_df)

        assert set(y.unique()).issubset({0, 1})
        assert "Churn" not in X.columns
        assert "customerID" not in X.columns
        assert len(y) == len(synthetic_raw_df)
        assert len(ids) == len(synthetic_raw_df)

        # Test invalid churn label
        df_bad_churn = synthetic_raw_df.copy()
        df_bad_churn.loc[0, "Churn"] = "Maybe"
        with pytest.raises(ValueError, match="Unexpected values in 'Churn'"):
            clean_dataset(df_bad_churn)

    def test_customer_id_exclusion_from_features(self, synthetic_raw_df):
        """CustomerID must be preserved as metadata series and strictly excluded from X."""
        X, y, ids = clean_dataset(synthetic_raw_df)
        assert "customerID" not in X.columns
        assert list(ids) == list(synthetic_raw_df["customerID"])

    def test_stratified_split_integrity_and_disjoint_ids(self):
        """Synthetic larger dataset must split into 70/15/15 partitions with no overlapping IDs."""
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            "customerID": [f"ID_{i:04d}" for i in range(n)],
            "feature1": np.random.randn(n),
            "Churn": ["Yes" if i % 4 == 0 else "No" for i in range(n)],  # 25% churn
        })
        X = df[["feature1"]]
        y = pd.Series([1 if c == "Yes" else 0 for c in df["Churn"]])
        ids = df["customerID"]

        X_tr, y_tr, id_tr, X_va, y_val, id_va, X_te, y_te, id_te = split_data(
            X, y, ids, split_config=DataSplitConfig(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
        )

        # Sizes
        assert len(id_tr) == 70
        assert len(id_va) == 15
        assert len(id_te) == 15

        # Disjoint IDs
        set_tr = set(id_tr)
        set_va = set(id_va)
        set_te = set(id_te)
        assert len(set_tr.intersection(set_va)) == 0
        assert len(set_tr.intersection(set_te)) == 0
        assert len(set_va.intersection(set_te)) == 0

        # Stratification: churn rate close to 25%
        assert abs(y_tr.mean() - 0.25) < 0.05
        assert abs(y_val.mean() - 0.25) < 0.10
        assert abs(y_te.mean() - 0.25) < 0.10

    def test_preprocessor_feature_count_and_mapping(self, synthetic_raw_df):
        """Preprocessor should produce expected feature names and a feature mapping."""
        X, y, ids = clean_dataset(synthetic_raw_df)
        prep = build_preprocessor()
        prep.fit(X)

        feature_mapping = extract_feature_mapping(prep)
        assert isinstance(feature_mapping, dict)
        assert len(feature_mapping) > 0

        transformed = prep.transform(X)
        assert transformed.shape[0] == len(synthetic_raw_df)


@pytest.mark.integration
class TestDataPipelineIntegration:
    """Integration tests verifying processed splits on disk."""

    def test_processed_partitions_exist_and_match_specs(self, processed_data_dir):
        """Processed parquet splits must exist, have 46 features, and maintain customer ID isolation."""
        train_path = processed_data_dir / "train_split.parquet"
        val_path = processed_data_dir / "val_split.parquet"
        test_path = processed_data_dir / "test_split.parquet"

        assert train_path.exists(), "train_split.parquet missing"
        assert val_path.exists(), "val_split.parquet missing"
        assert test_path.exists(), "test_split.parquet missing"

        df_tr = pd.read_parquet(train_path)
        df_va = pd.read_parquet(val_path)
        df_te = pd.read_parquet(test_path)

        # 46 features + customerID + Churn = 48 columns
        for name, df in [("Train", df_tr), ("Val", df_va), ("Test", df_te)]:
            assert df.shape[1] == 48, f"{name} split has {df.shape[1]} columns, expected 48"
            assert "customerID" in df.columns
            assert "Churn" in df.columns
            feature_cols = [c for c in df.columns if c not in ("customerID", "Churn")]
            assert feature_cols == EXPECTED_FEATURES, f"{name} feature columns mismatch"

        # Check total record count and split ratios (4930, 1056, 1057 => 7043)
        assert len(df_tr) == 4930
        assert len(df_va) == 1056
        assert len(df_te) == 1057

        # Strictly no ID overlap across partitions
        ids_tr = set(df_tr["customerID"])
        ids_va = set(df_va["customerID"])
        ids_te = set(df_te["customerID"])
        assert len(ids_tr.intersection(ids_va)) == 0
        assert len(ids_tr.intersection(ids_te)) == 0
        assert len(ids_va.intersection(ids_te)) == 0

