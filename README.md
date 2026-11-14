# ChurnIQ – AI-Powered Telecom Customer Churn Prediction System

[![Python](https://img.shields.io/badge/Python-3.12-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.15.0+cpu-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Keras 3](https://img.shields.io/badge/Keras%203-3.15.1%20(torch%20backend)-D00000.svg?logo=keras&logoColor=white)](https://keras.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.5.2-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.39.1-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![pytest](https://img.shields.io/badge/pytest-9.2.0%20(85%20passed)-0A9EDC.svg?logo=pytest&logoColor=white)](https://pytest.org/)

---

## 1. Project Title and Overview

**ChurnIQ** is an end-to-end, production-grade machine learning and deep learning system built to predict customer attrition in the telecommunications industry.

### Problem Statement & Motivation
Customer attrition (churn) directly threatens top-line recurring revenue in subscription telecommunications. Because acquiring a new subscriber typically costs 5 to 7 times more than retaining an existing customer, proactive churn mitigation is a central business driver. Traditional solutions suffer from three fundamental pitfalls:
1. **Data Leakage:** Transformers fitted across entire datasets before splitting, leading to overly optimistic cross-validation and production failure.
2. **Arbitrary Default Thresholds:** Evaluating models strictly at threshold $0.50$ on imbalanced customer distributions severely depresses minority churn recall.
3. **Black-Box Opacity:** Deep neural networks provide predictive power without explaining why a customer was flagged, preventing retention specialists from crafting targeted interventions.

### Project Objectives
- Build a strictly leakage-free preprocessing and feature engineering pipeline on the IBM Telco Customer Churn dataset.
- Train and compare a regularized **Logistic Regression** baseline with a customized **Multilayer Perceptron (MLP)** using Keras 3 with the PyTorch CPU backend.
- Independently optimize binary decision thresholds on validation data to maximize $F_1$-score.
- Conduct an unbiased, isolated final benchmark on a quarantined test partition ($n=1,057$).
- Provide Explainable AI (XAI) using model-agnostic **SHAP** (KernelExplainer) with representative background sampling.
- Decouple statistical decision thresholds from business risk stratification tiers (Low, Medium, High).
- Deliver an interactive 7-page **Streamlit** dashboard and a comprehensive 85-test automated verification suite.

---

## 2. Dataset

- **Source:** IBM Telco Customer Churn Dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`).
- **Scale:** 7,043 customer records and 21 raw columns (1 customer ID, 19 predictive attributes, 1 target label).
- **Target Distribution:**
  - Non-Churners (`No`): 5,174 customers (73.46%)
  - Churners (`Yes`): 1,869 customers (26.54%)
  - Class Imbalance Ratio: approximately 2.77:1.

### Data Cleaning Decisions
1. **`TotalCharges` Whitespace Handling:** In 11 records, `TotalCharges` contained empty whitespace strings (`" "`). Inspection confirmed that all 11 records had `tenure == 0` (brand-new accounts who had not yet received their first billing cycle). These values were cleanly imputed to `0.0` rather than dropping rows or using arbitrary mean/median imputation.
2. **Metadata Isolation:** The unique customer identifier (`customerID`) is preserved strictly as a non-predictive tracking identifier and is dropped before feature transformation and model training.
3. **Target Encoding:** Binary churn values (`No`/`Yes`) are mapped to numeric labels (`0`/`1`).

---

## 3. Technology Stack

- **Core Runtime:** Python 3.12.10 (AMD64, Windows 11)
- **Data Manipulation & Serialization:** `pandas 2.3.3`, `numpy 1.26.4`, `pyarrow 17.0.0`, `joblib 1.4.2`
- **Machine Learning Baseline:** `scikit-learn 1.5.2` (`ColumnTransformer`, `StandardScaler`, `OneHotEncoder`, `LogisticRegression`)
- **Deep Learning Framework:** `keras 3.15.1` with `torch 2.15.0+cpu` backend (`KERAS_BACKEND=torch`)
  - *Note:* TensorFlow 2.17.1 native C-extensions (`.pyd`) were blocked by Windows Smart App Control (SAC) policies. Keras 3 with the PyTorch CPU backend provided complete, verified execution without altering system security settings.
- **Explainable AI (XAI):** `shap 0.46.0` (model-agnostic `KernelExplainer`)
- **Interactive UI & Visualization:** `streamlit 1.39.1`, `plotly 5.24.1`, `matplotlib 3.9.4`, `seaborn 0.13.2`
- **Automated Testing Suite:** `pytest 9.2.0`, `pytest-cov 7.1.0`

---

## 4. System Architecture

The ChurnIQ system follows a modular architecture:

```
[Raw CSV: data/raw/]
        │
        ▼
[Module 1: data_pipeline.py] ───> 3-Way Stratified Split (70/15/15)
        │                         Fit ColumnTransformer on Train only (46 features)
        ├──> train_split.parquet (4,930 rows)
        ├──> val_split.parquet   (1,056 rows)
        └──> test_split.parquet  (1,057 rows, quarantined)
        │
        ├───> [Module 2: baseline_model.py]  ──> Logistic Regression (class_weight='balanced')
        │
        ├───> [Module 3: neural_network.py]   ──> Keras 3 MLP (PyTorch CPU, early stopping)
        │
        ▼
[Module 4: threshold_optimizer.py] ───> Search [0.05, 0.95] on Validation Set only
        │                               Selected: LR=0.54, MLP=0.46
        ▼
[Module 5: evaluation.py] ────────────> Isolated single-pass benchmark on Test Set
        │
        ├───> [Module 6: explainability.py] ──> KernelExplainer (75 real training records bg)
        │
        ├───> [Module 7: inference.py]       ──> Single & batch inference engine (Risk Tiers)
        │
        ├───> [Module 8: train.py]           ──> Master pipeline orchestrator & CLI
        │
        ├───> [Module 9: streamlit_app.py]   ──> 7-page interactive dashboard
        │
        └───> [Module 10: pytest suite]      ──> 64 automated unit & integration tests
```

---

## 5. Methodology

1. **Stratified 3-Way Splitting (70% / 15% / 15%):**
   - Training Set: 4,930 records (1,308 churners, 26.53%)
   - Validation Set: 1,056 records (280 churners, 26.52%)
   - Test Set: 1,057 records (281 churners, 26.58%)
   - Random seed `42` guarantees partition reproducibility. Zero customer ID leakage across splits.
2. **Training-Only Preprocessing:**
   - Numerical columns (`tenure`, `MonthlyCharges`, `TotalCharges`) normalized via `StandardScaler` fitted exclusively on the training partition.
   - 16 categorical features encoded via `OneHotEncoder(handle_unknown='ignore')`, producing 43 dummy columns.
   - Total transformed feature dimension: **46 features**.
3. **Class Weighting:**
   - Logistic Regression trained with `class_weight='balanced'`.
   - MLP uses computed class weights ($W_0 \approx 0.68$, $W_1 \approx 1.88$) applied during batch training to counteract churn minority penalty.
4. **Early Stopping Regularization:**
   - MLP monitored `val_loss` with `patience=3` and `restore_best_weights=True`.
   - Training stopped automatically at epoch 14 (optimal weights restored from epoch 11).
5. **Validation Threshold Optimization:**
   - Decision threshold evaluated from $0.05$ to $0.95$ in $0.01$ increments exclusively on validation predictions.
   - Maximized $F_1$-score using a deterministic tie-breaking rule (selecting the lowest threshold among maximum-$F_1$ candidates to prioritize recall).
6. **Isolated Final Benchmark:**
   - The quarantined test partition ($n=1,057$) was loaded exactly once in read-only mode.
   - Zero model retraining, tuning, or threshold modification occurred based on test set outcomes.

---

## 6. Model Architecture

The deep learning model is an engineered Multilayer Perceptron (MLP) built with Keras 3 and PyTorch:

```
Input: 46 numerical and one-hot encoded features
  │
  ▼
Dense(64 units, activation='relu', kernel_regularizer=L2(0.001))
  │
  ▼
Dropout(rate=0.25)
  │
  ▼
Dense(32 units, activation='relu', kernel_regularizer=L2(0.001))
  │
  ▼
Dropout(rate=0.25)
  │
  ▼
Dense(16 units, activation='relu', kernel_regularizer=L2(0.001))
  │
  ▼
Dense(1 unit, activation='sigmoid')  ──> Output Churn Probability P in [0.0, 1.0]
```

- **Optimizer:** Adam (learning rate = $0.001$)
- **Loss Function:** Binary Cross-Entropy
- **Batch Size:** 32
- **Regularization:** Layer-wise $L_2$ weight regularization ($10^{-3}$) plus $25\%$ dropout after each hidden layer.

---

## 7. Results

Final unbiased evaluation on the quarantined test set ($n=1,057$, 281 actual churners):

| Metric | Logistic Regression (Default 0.50) | Logistic Regression (Optimal 0.54) | MLP Neural Net (Default 0.50) | MLP Neural Net (Optimal 0.46) | Delta (MLP Opt vs LR Opt) |
|---|:---:|:---:|:---:|:---:|:---:|
| **ROC-AUC** | 0.8448 | 0.8448 | **0.8465** | **0.8465** | **+0.0017** |
| **PR-AUC (AP)** | **0.6592** | **0.6592** | 0.6559 | 0.6559 | -0.0033 |
| **F1-Score** | 0.6222 | 0.6259 | 0.6293 | **0.6326** | **+0.0067** |
| **Precision** | 0.5177 | **0.5385** | 0.5474 | 0.5259 | -0.0126 |
| **Recall** | 0.7794 | 0.7473 | 0.7402 | **0.7936** | **+0.0463** |
| **Accuracy** | 0.7483 | 0.7625 | **0.7682** | 0.7550 | -0.0075 |
| **Specificity** | 0.7371 | **0.7680** | 0.7784 | 0.7410 | -0.0270 |
| **True Positives (TP)** | 219 | 210 | 208 | **223** | **+13** |
| **False Negatives (FN)** | 62 | 71 | 73 | **58** | **-13** |
| **True Negatives (TN)** | 572 | **596** | **604** | 575 | -21 |
| **False Positives (FP)** | 204 | **180** | **172** | 201 | +21 |

### Performance Analysis
- **Model Comparison & Trade-offs:** The MLP at threshold $0.46$ achieved an $F_1$-score of **0.6326** compared to **0.6259** for the Logistic Regression baseline at threshold $0.54$, an absolute difference of $+0.0067$. The MLP achieved higher recall ($79.36\%$ vs. $74.73\%$, identifying 13 additional churners), whereas Logistic Regression achieved higher PR-AUC ($0.6592$ vs. $0.6559$) and higher precision ($53.85\%$ vs. $52.59\%$, yielding 21 fewer false positives).
- **Practical Interpretation:** The $F_1$ difference is small ($+0.0067$), and statistical significance was not tested. In practice, the choice between models depends on business priorities: the MLP is preferred when minimizing missed churners (false negatives) is critical, while Logistic Regression is preferable when retention budget constraints require higher precision and lower false-positive outreach costs.
- **Generalization Integrity:** Comparison of validation vs. test metrics confirmed no meaningful overfitting for either model ($|\Delta \text{ROC-AUC}| \le 0.0011$, $|\Delta F_1| \le 0.0070$).

---

## 8. SHAP Explainability

Model-agnostic SHAP (**KernelExplainer**) was deployed to interpret model behaviors:

### Sampling & Background Dataset
- To satisfy application security boundaries without deep gradient hooks, KernelExplainer was initialized with **75 representative records** drawn directly from training data, stratified by churn class (55 non-churners, 20 churners). Synthetic $k$-means centroids were deliberately avoided.
- Evaluated over a validation sample ($n=200$, $nsamples=100$).

### Top Global Churn Drivers (Aggregated to Original Attributes)
Features are aggregated to parent attributes in `reports/shap_global_importance.csv`. Note that attribution profiles differ between models:

- **Top Features for Multilayer Perceptron (MLP):**
  1. `tenure` (mean $|SHAP| = 0.1091$): Long customer tenure strongly reduces churn probability.
  2. `MonthlyCharges` (mean $|SHAP| = 0.0350$): Higher recurring monthly cost increases churn propensity.
  3. `Contract` (mean $|SHAP| = 0.0288$): Month-to-month contracts elevate churn risk; 1- and 2-year contracts reduce it.
  4. `InternetService` (mean $|SHAP| = 0.0177$): Fiber optic service is associated with higher churn likelihood.
  5. `PaperlessBilling` (mean $|SHAP| = 0.0116$): Customers on paperless billing show higher churn probability.
  6. `OnlineSecurity` (mean $|SHAP| = 0.0113$): Lack of online security increases churn propensity.
  7. `TechSupport` (mean $|SHAP| = 0.0096$): Lack of technical support increases churn propensity.
  *(Note: `TotalCharges` ranks 12th for MLP with mean $|SHAP| = 0.0061$.)*

- **Top Features for Logistic Regression:**
  1. `tenure` (mean $|SHAP| = 0.1769$): Dominant protective factor against churn.
  2. `TotalCharges` (mean $|SHAP| = 0.0699$): Cumulative spend is a strong linear attribution feature.
  3. `MonthlyCharges` (mean $|SHAP| = 0.0576$): Higher monthly bill increases churn likelihood.
  4. `Contract` (mean $|SHAP| = 0.0302$): Commitment length strongly influences retention.
  5. `InternetService` (mean $|SHAP| = 0.0291$): Fiber optic option correlates with churn.

> **Important Analytical Disclaimer:** SHAP explanations reflect *model behavior and feature attribution*, not customer causality. A high positive SHAP score indicates that a feature increased the model's predicted probability of churn, not that the feature caused the customer to leave.

---

## 9. Interactive Streamlit Dashboard

The production dashboard (`app/streamlit_app.py`) is organized into 7 specialized views:

1. 📊 **Overview:** High-level project KPIs (7,043 customers, 26.5% baseline churn), pipeline module matrix, class balance distribution, and key findings.
2. 🏆 **Model Performance:** Side-by-side metric tables (default vs. optimal thresholds), Plotly grouped bar comparison, confusion matrix heatmaps, test ROC/PR curves, and MLP training history.
3. 🎯 **Threshold Analysis:** Interactive $F_1$ vs. threshold curves for both models with marked optimal operating points, and full threshold search tables.
4. ⚠️ **Risk Stratification:** Business risk donut chart, risk tier customer breakdown, validation churn rate per tier table, and sample prediction records.
5. 🔍 **SHAP Explainability:** Global feature importance bar plots, beeswarm summary plots, and 4 local waterfall case studies (True Positive, True Negative, False Positive, False Negative).
6. 🔮 **Individual Prediction:** Real-time interactive scoring form with 19 customer attributes, visual probability gauge, business risk badge, and top-5 contributing features.
7. 🛠️ **Pipeline Status:** Automated artifact verification panel checking 20+ pipeline assets on disk, accompanied by active runtime environment diagnostics.

---

## 10. Installation and Execution

### Prerequisites
- Windows 10/11 (AMD64)
- Python 3.12 (Python 3.12.10 recommended)
- Git and PowerShell

### 1. Clone & Set Up Virtual Environment
```powershell
# Clone repository
git clone https://github.com/your-username/ChurnIQ.git
cd ChurnIQ

# Create dedicated virtual environment
python -m venv .venv

# Activate environment
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```powershell
# Install pinned core dependencies
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. Verify Keras PyTorch Backend
Ensure the backend is set to PyTorch (already configured in `src/config.py`):
```powershell
.\.venv\Scripts\python.exe -c "import os; os.environ['KERAS_BACKEND']='torch'; import keras; print('Keras backend:', keras.backend.backend())"
# Output must be: Keras backend: torch
```

### 4. Run the Pipeline Orchestrator (CLI)
```powershell
# Check current artifact status
.\.venv\Scripts\python.exe src/train.py --status

# Run full pipeline if rebuilding artifacts
.\.venv\Scripts\python.exe src/train.py --all
```

### 5. Launch the Streamlit Dashboard
```powershell
# Use python module invocation to ensure SAC compliance:
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```
Open **`http://localhost:8501`** in any web browser.

---

## 11. Automated Testing Suite

The testing suite (`pytest 9.2.0`) includes **64 automated tests** with 100% pass rate:

```powershell
# Run entire test suite (2.2 seconds)
.\.venv\Scripts\python.exe -m pytest

# Run unit tests only (41 tests)
.\.venv\Scripts\python.exe -m pytest -m unit

# Run integration tests only (23 tests)
.\.venv\Scripts\python.exe -m pytest -m integration
```

### Test Suite Structure
- `tests/test_data_pipeline.py` (9 tests): Missing column checks, zero-tenure whitespace imputation, binary encoding, disjoint IDs across splits.
- `tests/test_baseline_model.py` (6 tests): Artifact properties, probability range $[0, 1]$, dimension mismatch checks, validation replication.
- `tests/test_neural_network.py` (8 tests): Architecture layer validation, class weight balancing, early stopping checks, probability outputs.
- `tests/test_threshold_optimizer.py` (7 tests): Threshold search over 91 steps, deterministic tie-breaking, risk tier boundaries.
- `tests/test_evaluation.py` (7 tests): Metric formulas, generalization verification, empirical comparison rules, frozen test set verification.
- `tests/test_explainability.py` (8 tests): Feature grouping, SHAP aggregation, prediction wrappers, artifact integrity.
- `tests/test_inference.py` (10 tests): Single customer raw scoring, batch dataframe inference, unknown category tolerance.
- `tests/test_app.py` (9 tests): Plotly chart generation, static image existence, Streamlit 1.39 image compatibility (`use_column_width=True`), data loaders.
- `tests/test_train.py` (21 tests): CLI argument parsing, dependency verification, artifact status reporting, stage input/output checks, dry-run safety, selective stage dependency enforcement.

*Reports:* See [`reports/testing/test_summary_report.md`](reports/testing/test_summary_report.md) and [`reports/testing/junit_report.xml`](reports/testing/junit_report.xml).

---

## 12. Limitations and Future Improvements

1. **Class Imbalance & Sample Size:** The dataset contains 7,043 rows with 26.5% churners. While class weighting mitigated imbalance, larger enterprise datasets with millions of subscriber months would support more complex architectures.
2. **Static Cross-Sectional Data:** The IBM Telco dataset provides a single snapshot. Real-world telecom churn benefits from longitudinal time-series data (e.g., month-over-month usage trends, call drops, payment delays).
3. **Threshold vs. Business Payoff:** The current threshold maximizes statistical $F_1$. In production, optimization should integrate explicit financial cost matrices ($\text{Cost of Retention Offer}$ vs. $\text{Customer Lifetime Value}$).
4. **Causality vs. Correlation:** SHAP identifies associative signals. Prospective A/B testing is required before altering business policies based on feature attributions.

---

## 13. Project Structure

```
ChurnIQ/
├── .gitignore
├── pytest.ini
├── README.md
├── requirements.txt
├── app/
│   ├── streamlit_app.py           # 7-page interactive dashboard
│   ├── components/
│   │   ├── __init__.py
│   │   ├── charts.py              # 9 Plotly chart factories
│   │   └── ui_widgets.py          # Metric cards, risk badges, tables
│   └── styles/
│       └── custom.css             # Dark-theme custom styling
├── data/
│   ├── raw/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── processed/
│       ├── train_split.parquet    # 4,930 rows, 48 cols
│       ├── val_split.parquet      # 1,056 rows, 48 cols
│       └── test_split.parquet     # 1,057 rows, 48 cols (quarantined)
├── docs/
│   ├── ARCHITECTURE.md            # Detailed system design & data flow
│   ├── METHODOLOGY.md             # In-depth training & evaluation protocol
│   └── PROJECT_REPORT.md          # Comprehensive academic project report
├── models/
│   ├── baseline_logistic.joblib   # Trained Logistic Regression baseline
│   ├── feature_mapping.json       # 46 encoded cols -> 19 original attributes
│   ├── mlp_churn_model.keras      # Trained Keras 3 MLP model
│   ├── preprocessor.joblib        # Fitted ColumnTransformer (train only)
│   ├── shap_background.joblib     # 75 representative real training records
│   └── threshold_metadata.json    # Optimal thresholds (LR=0.54, MLP=0.46)
├── notebooks/                     # Exploratory analysis & prototype notebooks
├── reports/
│   ├── baseline_validation_metrics.json
│   ├── baseline_val_predictions.parquet
│   ├── final_test_metrics.json
│   ├── final_test_predictions.parquet
│   ├── inference_predictions_logistic_regression.parquet
│   ├── inference_predictions_mlp.parquet
│   ├── mlp_training_history.json
│   ├── mlp_validation_metrics.json
│   ├── mlp_val_predictions.parquet
│   ├── model_comparison.csv
│   ├── pipeline_run.log
│   ├── shap_global_importance.csv
│   ├── shap_local_explanations.json
│   ├── threshold_optimization_comparison.json
│   ├── threshold_search_results.csv
│   ├── figures/                   # 14 high-resolution diagnostic plots
│   └── testing/
│       ├── junit_report.xml
│       └── test_summary_report.md
├── src/
│   ├── __init__.py
│   ├── config.py                  # Global configurations, seeds, paths
│   ├── data_pipeline.py           # Ingestion, cleaning, splitting, transformation
│   ├── baseline_model.py          # Logistic Regression baseline pipeline
│   ├── neural_network.py          # Keras 3 MLP architecture & training
│   ├── threshold_optimizer.py     # Independent threshold search on val
│   ├── evaluation.py              # Unbiased final benchmark on test set
│   ├── explainability.py          # SHAP global & local feature attributions
│   ├── inference.py               # Single & batch prediction engine
│   └── train.py                   # Master pipeline orchestrator & CLI
└── tests/
    ├── __init__.py
    ├── conftest.py                # Shared fixtures & synthetic generators
    ├── test_app.py
    ├── test_baseline_model.py
    ├── test_data_pipeline.py
    ├── test_evaluation.py
    ├── test_explainability.py
    ├── test_inference.py
    ├── test_neural_network.py
    ├── test_threshold_optimizer.py
    └── test_train.py
```
