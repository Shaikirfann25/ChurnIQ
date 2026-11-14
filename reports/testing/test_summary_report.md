# ChurnIQ – Module 10: Automated Testing Suite Report

**Generated:** 2026-11-14  
**Environment:** Python 3.12.10, pytest 9.2.0, Keras 3.15.1, PyTorch 2.15.0+cpu  
**Project Root:** `C:\Users\shaik\OneDrive\Desktop\ChurnIQ`  
**Test Suite Directory:** `tests/`  

---

## 1. Executive Summary

| Category | Tests Executed | Passed | Failed | Skipped | Pass Rate | Runtime |
|----------|---------------|--------|--------|---------|-----------|---------|
| **Unit Tests (`-m unit`)** | 41 | 41 | 0 | 0 | **100%** | 1.90s |
| **Integration Tests (`-m integration`)** | 23 | 23 | 0 | 0 | **100%** | 1.35s |
| **Combined Full Suite** | **64** | **64** | **0** | **0** | **100%** | **2.17s** |

All tests completed deterministically without modifying saved models, model weights, thresholds, raw datasets, or evaluation metrics.

---

## 2. Test Coverage by Module

### Module 1: Data Pipeline (`tests/test_data_pipeline.py`)
- **Unit Tests (8):**
  - `test_missing_required_column_raises_error`: Validates `ValueError` when required columns are absent.
  - `test_clean_dataset_whitespace_totalcharges_tenure_zero`: Validates imputation of whitespace `TotalCharges` to `0.0` when `tenure == 0`.
  - `test_clean_dataset_whitespace_totalcharges_tenure_nonzero_raises`: Validates that whitespace `TotalCharges` with `tenure != 0` raises `ValueError`.
  - `test_clean_dataset_unparseable_totalcharges_raises`: Validates error handling for unparseable strings.
  - `test_target_encoding_and_separation`: Ensures binary mapping ('No' -> 0, 'Yes' -> 1) and target exclusion from features.
  - `test_customer_id_exclusion_from_features`: Validates `customerID` preservation as metadata only.
  - `test_stratified_split_integrity_and_disjoint_ids`: Validates 70/15/15 stratified partition ratios and pairwise disjoint ID sets.
  - `test_preprocessor_feature_count_and_mapping`: Validates that `ColumnTransformer` fits properly and generates a feature mapping.
- **Integration Tests (1):**
  - `test_processed_partitions_exist_and_match_specs`: Validates that `train_split.parquet` (4,930 rows), `val_split.parquet` (1,056 rows), and `test_split.parquet` (1,057 rows) have exactly 46 predictive features and zero customer ID leakage.

### Module 2: Baseline Model (`tests/test_baseline_model.py`)
- **Unit Tests (3):**
  - `test_evaluate_baseline_validation_metrics_contract`: Validates dictionary format, metrics computation, and probability bounds.
  - `test_invalid_input_dimension_raises`: Ensures mismatched input dimensions raise `ValueError`.
  - `test_probability_output_range_and_monotonicity`: Confirms all output probabilities strictly lie in `[0.0, 1.0]`.
- **Integration Tests (3):**
  - `test_baseline_artifact_properties`: Validates that `models/baseline_logistic.joblib` has 46 input coefficients and binary classes `[0, 1]`.
  - `test_validation_metrics_match_saved_report`: Validates that validation predictions recomputed against `val_split.parquet` exactly match `reports/baseline_validation_metrics.json` (ROC-AUC=0.8449, PR-AUC=0.6314, F1=0.6247).
  - `test_val_predictions_parquet_integrity`: Confirms `reports/baseline_val_predictions.parquet` integrity (1,056 rows).

### Module 3: Neural Network MLP (`tests/test_neural_network.py`)
- **Unit Tests (5):**
  - `test_verify_backend_torch`: Verifies that Keras 3 backend is confirmed as `"torch"`.
  - `test_build_mlp_model_architecture`: Validates layer sequence (Dense 64 -> Dropout 0.25 -> Dense 32 -> Dropout 0.25 -> Dense 16 -> Dense 1) and activations.
  - `test_compute_class_weights`: Ensures class weights adjust for minority positive class.
  - `test_evaluate_mlp_validation_contract`: Verifies metric schema and valid probability outputs.
  - `test_invalid_input_dimension_raises`: Verifies model rejects invalid tensor shapes.
- **Integration Tests (3):**
  - `test_saved_mlp_model_properties`: Validates `models/mlp_churn_model.keras` loading and input shape `(None, 46)`.
  - `test_saved_mlp_validation_metrics_match_report`: Confirms that running the model on `val_split.parquet` matches `reports/mlp_validation_metrics.json` (ROC-AUC=0.8454, PR-AUC=0.6321, F1=0.6289).
  - `test_training_history_json_integrity`: Validates `reports/mlp_training_history.json` contains `loss`, `val_loss`, `accuracy`, and `val_accuracy` across epochs.

### Module 4: Threshold Optimizer (`tests/test_threshold_optimizer.py`)
- **Unit Tests (5):**
  - `test_search_thresholds_grid_and_metrics`: Validates search over [0.05, 0.95] in 0.01 increments (91 steps) and confusion matrix conservation.
  - `test_deterministic_tie_breaking_selects_lowest_threshold`: Confirms deterministic rule: lowest threshold among tied max-F1 candidates is chosen.
  - `test_metrics_at_threshold_calculation`: Tests threshold-dependent calculations on synthetic cases.
  - `test_assign_risk_categories_boundaries`: Verifies mapping into Low (<0.30), Medium [0.30, 0.70), and High (>=0.70).
  - `test_verify_alignment_detects_mismatch`: Confirms detection of customer ID or label misalignment.
- **Integration Tests (2):**
  - `test_saved_threshold_metadata_independent_optimization`: Verifies independent thresholds in `models/threshold_metadata.json` (LR=0.54, MLP=0.46) and risk boundaries.
  - `test_saved_threshold_search_results_csv`: Confirms 182 rows (91 per model) in `reports/threshold_search_results.csv`.

### Module 5: Evaluation & Final Benchmark (`tests/test_evaluation.py`)
- **Unit Tests (4):**
  - `test_full_metrics_suite_correctness`: Verifies accuracy, precision, recall, F1, specificity, FPR, FNR, ROC-AUC, PR-AUC formulas.
  - `test_generalization_analysis_no_degradation`: Confirms detection of consistent validation vs. test metrics.
  - `test_generalization_analysis_detects_gap`: Verifies alert when severe metric drop occurs.
  - `test_compare_models_decisiveness_threshold`: Enforces 0.005 difference threshold for declaring a model winner.
- **Integration Tests (3):**
  - `test_final_test_metrics_json_integrity`: Validates `reports/final_test_metrics.json` has frozen thresholds and complete evaluation rows (n=1,057).
  - `test_final_test_predictions_parquet`: Confirms 1,057 test predictions with `lr_churn_probability` and `mlp_churn_probability` in [0, 1].
  - `test_model_comparison_csv_matches_metrics`: Validates `reports/model_comparison.csv` contains 4 rows.

### Module 6: SHAP Explainability (`tests/test_explainability.py`)
- **Unit Tests (4):**
  - `test_build_feature_groups_coverage`: Verifies all 46 encoded features map to their 19 original attributes.
  - `test_aggregate_shap_to_original`: Tests aggregation calculation from encoded columns to original features.
  - `test_make_lr_predictor_contract`: Validates LR prediction wrapper output properties.
  - `test_make_mlp_predictor_contract`: Validates MLP prediction wrapper output properties.
- **Integration Tests (4):**
  - `test_shap_background_dataset_properties`: Confirms `models/shap_background.joblib` is a (75, 46) float array of real training records.
  - `test_shap_global_importance_csv_structure`: Validates 38 rows (19 attributes x 2 models) in `reports/shap_global_importance.csv`.
  - `test_shap_local_explanations_json_cases`: Validates presence of `tp`, `tn`, `fp`, `fn` cases for both LR and MLP.
  - `test_shap_figure_artifacts_exist`: Confirms all 12 SHAP figures exist and are non-empty.

### Module 7: Inference Engine (`tests/test_inference.py`)
- **Unit Tests (5):**
  - `test_risk_category_assignment_boundaries`: Validates risk category thresholds.
  - `test_risk_boundaries_independent_of_binary_threshold`: Confirms business risk boundaries remain separate from binary classification thresholds.
  - `test_missing_required_fields_raises_error`: Ensures missing fields in raw customer input raise `ValueError`.
  - `test_missing_features_in_batch_encoded_raises_error`: Ensures missing columns in encoded batch raise `ValueError`.
  - `test_invalid_model_name_raises_value_error`: Validates rejection of unsupported model names.
- **Integration Tests (5):**
  - `test_single_customer_raw_prediction[mlp]`: Tests raw dict inference with MLP.
  - `test_single_customer_raw_prediction[logistic_regression]`: Tests raw dict inference with LR.
  - `test_batch_prediction_from_encoded_val_data[mlp]`: Tests batch inference on validation set (1,056 rows).
  - `test_batch_prediction_from_encoded_val_data[logistic_regression]`: Tests batch inference on validation set with LR.
  - `test_unknown_categorical_value_handled_gracefully`: Confirms robust handling of unseen categorical categories via OneHotEncoder ignore.

### Module 9: Dashboard Components (`tests/test_app.py`)
- **Unit Tests (7):**
  - `test_streamlit_app_has_no_unsupported_image_arguments`: Verifies `st.image()` does not use unsupported `use_container_width` argument.
  - `test_custom_css_exists_and_styled`: Checks custom CSS styling classes.
  - `test_roc_and_pr_curve_charts_return_valid_figures`: Verifies Plotly ROC and PR curve figures.
  - `test_confusion_matrix_chart_return_valid_figure`: Verifies Plotly confusion matrix heatmap.
  - `test_threshold_f1_chart_return_valid_figure`: Verifies Plotly threshold vs. F1 curve.
  - `test_shap_importance_and_risk_donut_charts`: Verifies SHAP importance bar and risk donut charts.
  - `test_training_history_and_metrics_comparison_charts`: Verifies training curves and metric comparison charts.
- **Integration Tests (2):**
  - `test_app_cached_data_loaders`: Confirms all Streamlit cached data loading functions load their target artifacts without error.
  - `test_all_referenced_images_exist`: Confirms all 14 images referenced in the dashboard exist and are non-empty.

---

## 3. Artifact Verification

- JUnit XML Report: [`reports/testing/junit_report.xml`](file:///C:/Users/shaik/OneDrive/Desktop/ChurnIQ/reports/testing/junit_report.xml)
- Test Summary Report: [`reports/testing/test_summary_report.md`](file:///C:/Users/shaik/OneDrive/Desktop/ChurnIQ/reports/testing/test_summary_report.md)
- `pytest.ini` markers:
  - `-m unit`: 41 passed
  - `-m integration`: 23 passed
  - `-m "not slow"`: supported for fast local feedback

