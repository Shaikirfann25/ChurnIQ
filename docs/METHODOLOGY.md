# ChurnIQ: Machine Learning & Evaluation Methodology Specification

**Document Version:** 1.0.0  
**Status:** Approved / Production  
**Scope:** Data Preprocessing, Model Training, Threshold Optimization, Benchmark Evaluation, and Model Interpretability  

---

## 1. Data Preprocessing & Sanitization Protocol

### 1.1 Ingestion & Schema Enforcement
Raw data ingestion (`src/data_pipeline.py:load_raw_data`) validates that incoming CSV files strictly contain all 21 expected columns:
- 1 Identification column: `customerID`
- 3 Continuous numerical attributes: `tenure`, `MonthlyCharges`, `TotalCharges`
- 16 Categorical attributes: `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`
- 1 Target column: `Churn`

Any missing column immediately triggers a `ValueError`, halting ingestion before downstream execution.

### 1.2 Deterministic TotalCharges Imputation
In raw data, 11 subscriber records contain whitespace strings (`" "`) within the `TotalCharges` column. Rather than applying naive statistical imputation (e.g., mean or median) or discarding records:
1. The pipeline checks the `tenure` of all accounts exhibiting blank `TotalCharges`.
2. Cross-validation confirmed that $100\%$ of these records had `tenure == 0`, representing brand-new subscribers enrolled during the current billing cycle who had not yet received their first invoice.
3. The pipeline imputes `TotalCharges = 0.0` for these zero-tenure records.
4. If a blank `TotalCharges` is ever encountered for an account with `tenure > 0`, the pipeline raises a `ValueError` requiring manual inspection.

### 1.3 Target Encoding & Metadata Isolation
- `customerID` is isolated into a separate tracking Series and removed from feature matrices.
- `Churn` labels are mapped to integer binaries:
  $$\text{Churn} = \begin{cases} 1 & \text{if } \text{label} = \text{"Yes"} \\ 0 & \text{if } \text{label} = \text{"No"} \end{cases}$$

---

## 2. Partitioning Protocol & Leakage Prevention

### 2.1 Stratified 3-Way Splitting
To prevent data snooping and information leakage, the sanitized dataset ($N = 7,043$) is partitioned into three disjoint subsets:
- **Training Set (70%):** 4,930 samples (1,308 churners, $26.53\%$)
- **Validation Set (15%):** 1,056 samples (280 churners, $26.52\%$)
- **Test Set (15%):** 1,057 samples (281 churners, $26.58\%$)

Partitioning uses `sklearn.model_selection.train_test_split` with `stratify=y` and `random_state=42`. Automated assertions verify that:
$$\mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{val}} = \emptyset, \quad \mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{test}} = \emptyset, \quad \mathcal{S}_{\text{val}} \cap \mathcal{S}_{\text{test}} = \emptyset$$

### 2.2 Training-Only Preprocessor Fitting
All data transformers are encapsulated inside a `ColumnTransformer` fitted **exclusively** on the training partition:
1. **StandardScaler:** Continuous numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) are normalized using training sample means ($\mu_{\text{train}}$) and variances ($\sigma_{\text{train}}^2$):
   $$z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
2. **OneHotEncoder:** Categorical columns are converted to binary indicator variables with `handle_unknown='ignore'`. This expands 16 categorical features into 43 dummy columns while ensuring runtime inference remains resilient to unseen categories.
3. **Feature Space:** Concatenating 3 continuous features and 43 one-hot columns produces an immutable feature space of **46 predictive features**.

The fitted transformer is serialized to `models/preprocessor.joblib`. The validation and test partitions are transformed using this frozen preprocessor without re-fitting.

---

## 3. Baseline Modeling Protocol: Regularized Logistic Regression

### 3.1 Mathematical Formulation
The linear baseline (`src/baseline_model.py`) models the log-odds of customer churn as a linear combination of the 46 features:

$$\log \left( \frac{P(y=1|x)}{1 - P(y=1|x)} \right) = w^T x + b$$

Parameters $w \in \mathbb{R}^{46}$ and $b \in \mathbb{R}$ are estimated by minimizing the $L_2$-penalized cross-entropy objective:

$$\mathcal{J}(w, b) = \frac{1}{2} \|w\|_2^2 + C \sum_{i=1}^{N_{\text{train}}} w_{c_i} \left[ -y_i \log(\hat{p}_i) - (1 - y_i) \log(1 - \hat{p}_i) \right]$$

### 3.2 Hyperparameters & Configuration
- **Inverse Regularization Strength:** $C = 1.0$
- **Optimization Algorithm:** Limited-memory Broyden–Fletcher–Goldfarb–Shanno (L-BFGS)
- **Maximum Iterations:** 1,000 (converged in 43 iterations)
- **Class Balancing:** `class_weight='balanced'` assigns sample weights inversely proportional to class frequencies:
  $$w_0 = \frac{4930}{2 \times 3622} \approx 0.6806, \quad w_1 = \frac{4930}{2 \times 1308} \approx 1.8846$$
- **Reproducibility:** `random_state = 42`

---

## 4. Deep Learning Methodology: Multilayer Perceptron (MLP)

### 4.1 Framework & Runtime Configuration
To maintain compliance with Windows 11 Smart App Control (which blocks native TensorFlow C-extensions), the neural network is authored in **Keras 3.15.1** executing directly over the **PyTorch 2.15.0+cpu** backend (`KERAS_BACKEND=torch`).

### 4.2 Network Architecture
The network topology is a dense feedforward architecture with layer-wise regularizers:

```
Layer (type)                Output Shape              Param #       Activation / Details
========================================================================================
Input                       (None, 46)                0             Input features
Dense_1 (Dense)             (None, 64)                3,008         ReLU, L2 reg (0.001)
Dropout_1 (Dropout)         (None, 64)                0             Rate = 0.25
Dense_2 (Dense)             (None, 32)                2,080         ReLU, L2 reg (0.001)
Dropout_2 (Dropout)         (None, 32)                0             Rate = 0.25
Dense_3 (Dense)             (None, 16)                528           ReLU, L2 reg (0.001)
Dense_Out (Dense)           (None, 1)                 17            Sigmoid
========================================================================================
Total Parameters: 5,633 (all trainable)
```

### 4.3 Training Dynamics & Regularization
- **Loss Function:** Binary Cross-Entropy with computed training class weights ($W_0 \approx 0.68$, $W_1 \approx 1.88$):
  $$\mathcal{L}(\theta) = - \frac{1}{B} \sum_{i=1}^B \left[ W_1 y_i \log(\hat{y}_i) + W_0 (1 - y_i) \log(1 - \hat{y}_i) \right]$$
- **Optimizer:** Adam ($\eta = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-7}$)
- **Batch Size:** 32 samples per mini-batch
- **Epoch Budget:** Up to 100 epochs with Early Stopping
- **Early Stopping Configuration:** Monitored `val_loss` with `patience=3` and `restore_best_weights=True`.
  - The model reached its minimum validation loss at **Epoch 11** ($\text{val\_loss} = 0.5227$).
  - Training continued through Epochs 12, 13, and 14 without improvement, triggering early termination at Epoch 14 and restoring the Epoch 11 weights.

---

## 5. Decision Threshold Optimization Protocol

### 5.1 Objective & Search Strategy
Standard binary classification maps probabilities to decisions using $\tau = 0.50$. For imbalanced target classes, $\tau = 0.50$ heavily favors precision at the expense of recall.

Threshold optimization is conducted strictly on validation predictions:
$$\tau^* = \arg\max_{\tau \in \mathcal{T}} F_1(\tau; \mathcal{D}_{\text{val}})$$
where $\mathcal{T} = \{0.05, 0.06, 0.07, \dots, 0.95\}$ (91 candidate thresholds).

### 5.2 Deterministic Tie-Breaking Rule
If multiple thresholds produce the same maximum $F_1$-score:
$$\tau^* = \min \left\{ \tau \in \mathcal{T} \mid F_1(\tau) = \max_{\tau'} F_1(\tau') \right\}$$
*Operational Rationale:* A lower threshold produces higher sensitivity (recall), capturing a larger fraction of true churners. In telecom customer retention, false negatives are significantly more expensive than false positives.

### 5.3 Optimal Operating Points
- **Logistic Regression:** Optimal threshold $\tau^* = 0.54$ ($F_1 = 0.6329$, precision = $0.5220$, recall = $0.8036$).
- **Multilayer Perceptron:** Optimal threshold $\tau^* = 0.46$ ($F_1 = 0.6319$, precision = $0.5134$, recall = $0.8214$).

---

## 6. Final Test Evaluation & Benchmark Protocol

### 6.1 Strict Test Data Isolation
The quarantined test partition ($n = 1,057$) is evaluated in a single isolated step (`src/evaluation.py`):
1. Test data is loaded read-only.
2. The preprocessor and models are loaded from disk without re-fitting.
3. Thresholds are read directly from `models/threshold_metadata.json`.
4. No threshold adjustments, hyperparameter re-tuning, or weight updates are permitted based on test set results.

### 6.2 Evaluation Metrics
The evaluation computes the full suite of statistical metrics:
- **Threshold-Independent:**
  - Area Under the Receiver Operating Characteristic Curve (ROC-AUC)
  - Area Under the Precision-Recall Curve (PR-AUC / Average Precision)
- **Threshold-Dependent (@ default 0.50 and optimal thresholds):**
  - Accuracy: $\frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$
  - Precision (Positive Predictive Value): $\frac{\text{TP}}{\text{TP} + \text{FP}}$
  - Recall (Sensitivity / True Positive Rate): $\frac{\text{TP}}{\text{TP} + \text{FN}}$
  - $F_1$-Score: $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$
  - Specificity (True Negative Rate): $\frac{\text{TN}}{\text{TN} + \text{FP}}$
  - False Positive Rate (FPR): $\frac{\text{FP}}{\text{FP} + \text{TN}}$
  - False Negative Rate (FNR): $\frac{\text{FN}}{\text{FN} + \text{TP}}$
  - Full $2 \times 2$ Confusion Matrix

### 6.3 Empirical Difference Threshold & Overfitting Criteria
- **Empirical Difference Threshold:** An empirical margin of $\Delta \ge 0.005$ in ROC-AUC or F1 was defined as an operational heuristic to distinguish meaningful performance separation from trivial numerical noise. However, formal hypothesis testing for statistical significance was not performed; differences should be interpreted as small metric trade-offs rather than proven statistical superiority.
- **Overfitting Signal:** A generalization gap is flagged if validation versus test drop exceeds $|\Delta \text{ROC-AUC}| > 0.03$ or $|\Delta F_1| > 0.04$.

---

## 7. Model Interpretability Protocol (SHAP)

### 7.1 Method Selection: Model-Agnostic KernelExplainer
Because deep gradient hooks (DeepExplainer / GradientExplainer) require native TensorFlow symbols that are restricted by Windows SAC, ChurnIQ utilizes model-agnostic **KernelExplainer**. KernelExplainer estimates Shapley values via weighted local linear regression around a background reference dataset:

$$\phi_j(x) = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|! (|F| - |S| - 1)!}{|F|!} \left( f_x(S \cup \{j\}) - f_x(S) \right)$$

### 7.2 Background Dataset Construction
- A background reference of **75 real customer accounts** was drawn directly from the training partition.
- Sampling was stratified by churn class (55 non-churners, 20 churners), preserving real feature correlations and avoiding unrealistic synthetic centroids.
- Test partition records were strictly excluded from background reference sampling.

### 7.3 Feature Aggregation Hierarchy
Because one-hot encoding splits categorical features into dummy indicators, SHAP attributions across dummy columns are aggregated to report overall attribute sensitivity:

$$\bar{\phi}_{\text{attribute}} = \frac{1}{|\mathcal{G}|} \sum_{k \in \mathcal{G}} \frac{1}{N_{\text{sample}}} \sum_{i=1}^{N_{\text{sample}}} |\phi_{i, k}|$$

This provides executive stakeholders with intuitive importance metrics for `Contract`, `InternetService`, and `PaymentMethod` rather than fragmented indicator attributions.

---

## 8. Business Risk Stratification Protocol

To prevent frontline retention teams from misinterpreting a statistical binary decision threshold as an all-or-nothing marketing trigger, churn probabilities are decoupled into three business tiers:

1. **Low Risk ($P < 0.30$):**
   - Churn probability is well below the population baseline ($26.5\%$).
   - Retention expenditure has negative expected ROI.
   - Recommended Action: Standard service maintenance and non-intrusive satisfaction surveys.
2. **Medium Risk ($0.30 \le P < 0.70$):**
   - Customer exhibits moderate dissatisfaction or competitive risk signals.
   - Ideal cohort for automated, low-cost proactive engagement.
   - Recommended Action: Digital retention nudges, self-service feature onboarding, loyalty point incentives.
3. **High Risk ($P \ge 0.70$):**
   - High-confidence churn candidates.
   - Justifies expensive, high-touch interventions.
   - Recommended Action: Direct outreach by retention specialists, customized service contract renegotiation, targeted billing discounts.

This stratification guarantees that operational budgeting remains independent of the mathematical threshold used to evaluate model $F_1$.

