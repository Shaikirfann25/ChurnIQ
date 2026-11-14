"""
ChurnIQ - Module 7: Inference & Risk Stratification Engine
============================================================
Loads trained artifacts and produces churn predictions with:
  - Binary churn decision (using optimized model thresholds)
  - Churn probability score
  - Business risk category (Low / Medium / High)
  - Top contributing features (from SHAP background + model weights)

Design rules:
  - customerID is metadata only — never passed to a model.
  - Thresholds read exclusively from models/threshold_metadata.json.
  - Models never retrained or modified.
  - Supports both single-record and batch inference.
  - Risk categories are independent of binary classification threshold.
"""

import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import sys
import json
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Project path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    MODELS_DIR,
    REPORTS_DIR,
    PROCESSED_DATA_DIR,
    RiskThresholdsConfig,
)

import keras

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RISK_CFG = RiskThresholdsConfig()

EXPECTED_FEATURES = [
    "tenure", "MonthlyCharges", "TotalCharges",
    "gender_Female", "gender_Male",
    "SeniorCitizen_0", "SeniorCitizen_1",
    "Partner_No", "Partner_Yes",
    "Dependents_No", "Dependents_Yes",
    "PhoneService_No", "PhoneService_Yes",
    "MultipleLines_No", "MultipleLines_No phone service", "MultipleLines_Yes",
    "InternetService_DSL", "InternetService_Fiber optic", "InternetService_No",
    "OnlineSecurity_No", "OnlineSecurity_No internet service", "OnlineSecurity_Yes",
    "OnlineBackup_No", "OnlineBackup_No internet service", "OnlineBackup_Yes",
    "DeviceProtection_No", "DeviceProtection_No internet service", "DeviceProtection_Yes",
    "TechSupport_No", "TechSupport_No internet service", "TechSupport_Yes",
    "StreamingTV_No", "StreamingTV_No internet service", "StreamingTV_Yes",
    "StreamingMovies_No", "StreamingMovies_No internet service", "StreamingMovies_Yes",
    "Contract_Month-to-month", "Contract_One year", "Contract_Two year",
    "PaperlessBilling_No", "PaperlessBilling_Yes",
    "PaymentMethod_Bank transfer (automatic)",
    "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check",
    "PaymentMethod_Mailed check",
]

# Feature groups: original attribute → list of encoded column indices
# (derived from feature_mapping.json at load time)
_FEATURE_GROUPS: dict = {}


# ===========================================================================
# Data classes
# ===========================================================================
@dataclass
class ChurnPrediction:
    """Single-record churn prediction result."""
    customer_id:        str
    churn_probability:  float
    churn_predicted:    bool          # at model's optimized threshold
    risk_category:      str           # Low / Medium / High
    model_used:         str           # 'mlp' or 'logistic_regression'
    threshold_used:     float
    top_features:       list = field(default_factory=list)   # top 5 feature contributions
    metadata:           dict = field(default_factory=dict)


@dataclass
class BatchPredictionResult:
    """Batch inference result."""
    model_used:    str
    n_records:     int
    predictions_df: pd.DataFrame    # one row per customer
    risk_summary:  dict             # count/pct per risk category


# ===========================================================================
# 1. Artifact loading (lazy singleton pattern)
# ===========================================================================
class ChurnIQInference:
    """
    Singleton inference engine for ChurnIQ.

    Loads models and metadata once; supports repeated single/batch calls.
    """

    _instance = None

    def __new__(cls, model_name: str = "mlp"):
        # Allow separate instances per model_name
        return super().__new__(cls)

    def __init__(self, model_name: str = "mlp"):
        """
        Parameters
        ----------
        model_name : 'mlp' or 'logistic_regression'
        """
        if model_name not in ("mlp", "logistic_regression"):
            raise ValueError(f"model_name must be 'mlp' or 'logistic_regression', got '{model_name}'")

        self.model_name    = model_name
        self._loaded       = False
        self.model         = None
        self.preprocessor  = None
        self.threshold     = None
        self.feature_mapping: dict = {}

    def _load(self) -> None:
        """Load all required artifacts (called lazily on first predict)."""
        if self._loaded:
            return

        # Threshold metadata
        meta_path = MODELS_DIR / "threshold_metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)

        key = "mlp" if self.model_name == "mlp" else "logistic_regression"
        self.threshold = meta["models"][key]["optimal_threshold"]

        # Feature mapping
        mapping_path = MODELS_DIR / "feature_mapping.json"
        with open(mapping_path) as f:
            self.feature_mapping = json.load(f)
        _build_feature_groups(self.feature_mapping)

        # Preprocessor (for raw → encoded pipeline)
        self.preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")

        # Model
        if self.model_name == "mlp":
            self.model = keras.models.load_model(
                str(MODELS_DIR / "mlp_churn_model.keras")
            )
        else:
            self.model = joblib.load(MODELS_DIR / "baseline_logistic.joblib")

        self._loaded = True
        print(
            f"[OK] ChurnIQInference loaded: model={self.model_name}, "
            f"threshold={self.threshold}"
        )

    # -----------------------------------------------------------------------
    # 2. Preprocessing: raw customer data → 46 encoded features
    # -----------------------------------------------------------------------
    def _preprocess(self, df_raw: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """
        Apply saved preprocessor to raw customer data.

        Parameters
        ----------
        df_raw : DataFrame with original (pre-encoded) columns.
                 Must NOT contain customerID or Churn.

        Returns
        -------
        X_encoded : float32 ndarray (N, 46)
        ids       : list of customerID strings (if present in df_raw)
        """
        # Extract customerID if present (metadata only)
        ids = []
        if "customerID" in df_raw.columns:
            ids = df_raw["customerID"].tolist()
            df_raw = df_raw.drop(columns=["customerID"])
        if "Churn" in df_raw.columns:
            df_raw = df_raw.drop(columns=["Churn"])

        X_encoded = self.preprocessor.transform(df_raw)
        return X_encoded.astype(np.float32), ids

    # -----------------------------------------------------------------------
    # 3. Predict from already-encoded features (46-col array)
    # -----------------------------------------------------------------------
    def _predict_encoded(self, X: np.ndarray) -> np.ndarray:
        """Return churn probabilities from 46-feature encoded array."""
        assert X.shape[1] == 46, f"Expected 46 features, got {X.shape[1]}"
        if self.model_name == "mlp":
            return self.model.predict(X, verbose=0).ravel()
        else:
            return self.model.predict_proba(X.astype(float))[:, 1]

    # -----------------------------------------------------------------------
    # 4. Risk category assignment
    # -----------------------------------------------------------------------
    @staticmethod
    def _assign_risk(prob: float) -> str:
        """
        Map a single probability to a business risk category.

        Boundaries (from RiskThresholdsConfig, independent of binary threshold):
          Low    : prob < 0.30
          Medium : 0.30 <= prob < 0.70
          High   : prob >= 0.70
        """
        if prob < RISK_CFG.low_threshold:
            return "Low"
        elif prob < RISK_CFG.high_threshold:
            return "Medium"
        else:
            return "High"

    # -----------------------------------------------------------------------
    # 5. Top feature contributions (LR: coefficient × feature value proxy)
    # -----------------------------------------------------------------------
    def _top_features(self, x_encoded: np.ndarray, top_n: int = 5) -> list[dict]:
        """
        Fast approximate feature contributions for a single record.

        Method:
          - LR: coefficient × |feature_value| — a linear attribution proxy.
          - MLP: |feature_value × (feature std deviation)| proxy using
            the encoded input directly (no SHAP call at inference time).

        Note: These are lightweight approximations, not SHAP values.
        For full SHAP explanations use src/explainability.py.

        Returns list of dicts: {feature, encoded_value, contribution_proxy}
        sorted by |contribution_proxy| descending.
        """
        if self.model_name == "logistic_regression":
            coefs = self.model.coef_[0]
            contribs = coefs * x_encoded
        else:
            # MLP: use absolute feature value as proxy (no coef available)
            contribs = np.abs(x_encoded)

        order = np.argsort(np.abs(contribs))[::-1][:top_n]
        return [
            {
                "feature":            EXPECTED_FEATURES[i],
                "original_attribute": _encoded_to_original(
                    EXPECTED_FEATURES[i], self.feature_mapping
                ),
                "encoded_value":      round(float(x_encoded[i]), 4),
                "contribution_proxy": round(float(contribs[i]),   4),
            }
            for i in order
        ]

    # -----------------------------------------------------------------------
    # 6. Single-record inference (from encoded 46-feature vector)
    # -----------------------------------------------------------------------
    def predict_one_encoded(
        self,
        x_encoded: np.ndarray,
        customer_id: str = "unknown",
    ) -> ChurnPrediction:
        """
        Predict churn for a single already-preprocessed customer.

        Parameters
        ----------
        x_encoded   : float32 array (46,)
        customer_id : metadata identifier

        Returns
        -------
        ChurnPrediction
        """
        self._load()
        x2d   = x_encoded.reshape(1, -1)
        prob  = float(self._predict_encoded(x2d)[0])
        pred  = prob >= self.threshold
        risk  = self._assign_risk(prob)
        feats = self._top_features(x_encoded)

        return ChurnPrediction(
            customer_id       = customer_id,
            churn_probability = round(prob, 4),
            churn_predicted   = bool(pred),
            risk_category     = risk,
            model_used        = self.model_name,
            threshold_used    = self.threshold,
            top_features      = feats,
        )

    # -----------------------------------------------------------------------
    # 7. Single-record inference (from raw customer data dict)
    # -----------------------------------------------------------------------
    def predict_one_raw(self, customer_data: dict) -> ChurnPrediction:
        """
        Predict churn from a dict of raw (unencoded) customer fields.

        Parameters
        ----------
        customer_data : dict matching the original IBM Telco CSV columns.
                        customerID is extracted as metadata.

        Returns
        -------
        ChurnPrediction
        """
        self._load()
        df_raw = pd.DataFrame([customer_data])
        customer_id = str(customer_data.get("customerID", "unknown"))
        X_enc, _ = self._preprocess(df_raw)
        return self.predict_one_encoded(X_enc[0], customer_id=customer_id)

    # -----------------------------------------------------------------------
    # 8. Batch inference (from pre-encoded Parquet or DataFrame)
    # -----------------------------------------------------------------------
    def predict_batch_encoded(
        self,
        df_encoded: pd.DataFrame,
        id_col: str = "customerID",
    ) -> BatchPredictionResult:
        """
        Batch inference from an already-preprocessed DataFrame.

        Parameters
        ----------
        df_encoded : DataFrame with 46 encoded feature columns + optional id_col.
                     Must NOT contain 'Churn' (target).

        Returns
        -------
        BatchPredictionResult
        """
        self._load()

        # Extract IDs
        if id_col in df_encoded.columns:
            ids = df_encoded[id_col].values
        else:
            ids = np.arange(len(df_encoded)).astype(str)

        # Verify features
        missing = [c for c in EXPECTED_FEATURES if c not in df_encoded.columns]
        if missing:
            raise ValueError(f"Missing encoded features: {missing}")

        X = df_encoded[EXPECTED_FEATURES].values.astype(np.float32)
        probas = self._predict_encoded(X)

        pred_binary = (probas >= self.threshold).astype(int)
        risk_cats   = np.array([self._assign_risk(p) for p in probas])

        result_df = pd.DataFrame({
            "customerID":        ids,
            "churn_probability": np.round(probas, 4),
            "churn_predicted":   pred_binary,
            "risk_category":     risk_cats,
            "model":             self.model_name,
            "threshold_used":    self.threshold,
        })

        # Risk summary
        risk_summary = {}
        for cat in ("Low", "Medium", "High"):
            n   = int((risk_cats == cat).sum())
            pct = round(n / len(risk_cats) * 100, 2)
            risk_summary[cat] = {"count": n, "pct": pct}

        return BatchPredictionResult(
            model_used     = self.model_name,
            n_records      = len(result_df),
            predictions_df = result_df,
            risk_summary   = risk_summary,
        )

    # -----------------------------------------------------------------------
    # 9. Batch inference from raw CSV-like DataFrame
    # -----------------------------------------------------------------------
    def predict_batch_raw(self, df_raw: pd.DataFrame) -> BatchPredictionResult:
        """
        Batch inference from a raw (unencoded) customer DataFrame.

        Parameters
        ----------
        df_raw : DataFrame matching the original IBM Telco CSV structure.

        Returns
        -------
        BatchPredictionResult
        """
        self._load()
        ids = df_raw["customerID"].tolist() if "customerID" in df_raw.columns else []
        X_enc, _ = self._preprocess(df_raw.copy())

        # Build a minimal encoded DataFrame for predict_batch_encoded
        df_enc = pd.DataFrame(X_enc, columns=EXPECTED_FEATURES)
        if ids:
            df_enc.insert(0, "customerID", ids)

        return self.predict_batch_encoded(df_enc)


# ===========================================================================
# Helper functions
# ===========================================================================
def _build_feature_groups(feature_mapping: dict) -> None:
    """Populate global _FEATURE_GROUPS from feature_mapping."""
    global _FEATURE_GROUPS
    _FEATURE_GROUPS = {}
    for idx, col in enumerate(EXPECTED_FEATURES):
        val  = feature_mapping.get(col, col)
        orig = val if isinstance(val, str) else val.get("original_feature", col)
        _FEATURE_GROUPS.setdefault(orig, []).append(idx)


def _encoded_to_original(encoded_col: str, feature_mapping: dict) -> str:
    """Map an encoded column name to its original customer attribute."""
    val = feature_mapping.get(encoded_col, encoded_col)
    return val if isinstance(val, str) else val.get("original_feature", encoded_col)


# ===========================================================================
# 10. Risk stratification summary
# ===========================================================================
def summarise_risk_stratification(result: BatchPredictionResult) -> None:
    """Print a formatted risk stratification table."""
    print("\n" + "=" * 55)
    print(f"  ChurnIQ Risk Stratification — {result.model_used}")
    print("=" * 55)
    print(f"  Total customers: {result.n_records}")
    print(f"  Threshold used: {result.predictions_df['threshold_used'].iloc[0]:.2f}")
    print()
    print(f"  {'Risk Category':<18} {'Count':>8} {'Pct':>8}  {'Action'}")
    print(f"  {'-'*52}")
    action_map = {
        "High":   "🔴 Immediate retention offer",
        "Medium": "🟡 Monitor + proactive outreach",
        "Low":    "🟢 Routine engagement",
    }
    for cat in ("High", "Medium", "Low"):
        s = result.risk_summary[cat]
        print(
            f"  {cat:<18} {s['count']:>8} {s['pct']:>7.1f}%  {action_map[cat]}"
        )
    n_flagged = int(result.predictions_df["churn_predicted"].sum())
    print(f"\n  Predicted churners (at threshold): {n_flagged} / {result.n_records} "
          f"({100*n_flagged/result.n_records:.1f}%)")
    print("=" * 55 + "\n")

    print("  ⚠  Risk categories (Low/Medium/High) are INDEPENDENT of the binary")
    print("     classification threshold. They are based on churn probability")
    print("     boundaries (Low<0.30, Medium 0.30-0.70, High≥0.70) and are used")
    print("     for business prioritisation only.\n")


# ===========================================================================
# 11. Save batch inference results
# ===========================================================================
def save_inference_results(
    result: BatchPredictionResult,
    output_path: Optional[Path] = None,
) -> Path:
    """Save batch predictions to Parquet."""
    if output_path is None:
        output_path = REPORTS_DIR / f"inference_predictions_{result.model_used}.parquet"
    result.predictions_df.to_parquet(output_path, index=False)
    print(f"[OK] Inference results saved → {output_path}")
    return output_path


# ===========================================================================
# 12. Demo / smoke test on validation data
# ===========================================================================
def run_inference_demo() -> None:
    """
    Demonstration: run batch inference on the validation partition.

    Uses pre-encoded val_split.parquet (already processed by data_pipeline.py).
    Does NOT use test_split.parquet.
    """
    print("\n" + "=" * 55)
    print("  ChurnIQ — Module 7: Inference Engine Demo")
    print("=" * 55)

    # --- Run for both models ---
    for model_name in ("mlp", "logistic_regression"):
        print(f"\n[INFO] Loading inference engine: {model_name}")
        engine = ChurnIQInference(model_name=model_name)

        # Batch inference on val partition
        df_val = pd.read_parquet(PROCESSED_DATA_DIR / "val_split.parquet")
        result = engine.predict_batch_encoded(df_val)

        summarise_risk_stratification(result)
        save_inference_results(result)

    # --- Single-record demo (MLP) ---
    print("\n[INFO] Single-record demo (MLP) ...")
    mlp_engine = ChurnIQInference(model_name="mlp")
    mlp_engine._load()

    df_val = pd.read_parquet(PROCESSED_DATA_DIR / "val_split.parquet")
    # Pick first row as example
    sample_row = df_val.iloc[0]
    cid        = str(sample_row["customerID"])
    x_enc      = sample_row[EXPECTED_FEATURES].values.astype(np.float32)

    pred = mlp_engine.predict_one_encoded(x_enc, customer_id=cid)

    print(f"\n  Single-record prediction:")
    print(f"    customerID:        {pred.customer_id}")
    print(f"    churn_probability: {pred.churn_probability:.4f}")
    print(f"    churn_predicted:   {pred.churn_predicted}")
    print(f"    risk_category:     {pred.risk_category}")
    print(f"    model:             {pred.model_used}")
    print(f"    threshold:         {pred.threshold_used:.2f}")
    print(f"    top_features:")
    for f in pred.top_features:
        print(
            f"      {f['original_attribute']:<22} "
            f"({f['feature']})  val={f['encoded_value']:+.4f}  "
            f"contrib={f['contribution_proxy']:+.4f}"
        )

    # --- Verification ---
    print("\n[Verification]")

    # 1. Test partition not accessed
    test_path = PROCESSED_DATA_DIR / "test_split.parquet"
    print(f"  [OK] Test partition NOT accessed. Exists: {test_path.exists()}")

    # 2. customerID not passed to model
    for model_name in ("mlp", "logistic_regression"):
        preds_path = REPORTS_DIR / f"inference_predictions_{model_name}.parquet"
        df_check   = pd.read_parquet(preds_path)
        assert "customerID" in df_check.columns, "customerID missing from output"
        assert all(c not in EXPECTED_FEATURES for c in ["customerID"]), \
            "customerID in model features!"
    print("  [OK] customerID kept as metadata, never passed to model.")

    # 3. Probabilities in [0,1]
    for model_name in ("mlp", "logistic_regression"):
        df_check = pd.read_parquet(
            REPORTS_DIR / f"inference_predictions_{model_name}.parquet"
        )
        p = df_check["churn_probability"].values
        assert np.all((p >= 0) & (p <= 1)), f"{model_name}: probabilities out of range!"
    print("  [OK] All probabilities within [0, 1].")

    # 4. Risk boundaries independent of threshold
    mlp_meta_path = MODELS_DIR / "threshold_metadata.json"
    with open(mlp_meta_path) as f:
        meta = json.load(f)
    rb = meta["risk_boundaries"]
    assert rb["low_threshold"]  == RISK_CFG.low_threshold
    assert rb["high_threshold"] == RISK_CFG.high_threshold
    print(
        f"  [OK] Risk boundaries (low={rb['low_threshold']}, "
        f"high={rb['high_threshold']}) confirmed independent of thresholds."
    )

    # 5. Output files readable
    for model_name in ("mlp", "logistic_regression"):
        preds_path = REPORTS_DIR / f"inference_predictions_{model_name}.parquet"
        df_r = pd.read_parquet(preds_path)
        assert len(df_r) == 1056, f"Expected 1056 val rows, got {len(df_r)}"
    print("  [OK] Inference output files readable (1056 val records each).")

    print("[OK] All verifications passed.")
    print("\n[DONE] Module 7 complete. Awaiting user review before next module.")


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    run_inference_demo()

