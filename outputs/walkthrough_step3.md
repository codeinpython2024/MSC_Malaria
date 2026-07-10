# Walkthrough: Step 3 Validation Partitioning & Step 4 SMOTE-NC Resampling

This document serves as the comprehensive walkthrough and academic audit trail for **Step 3: Establish the Validation Partition via Stratified Splitting** and **Step 4: Execute Data-Level Resampling via SMOTE-NC** as implemented in [third_step.py](file:///c:/MSc_malaria/third_step.py).

---

## 1. Theoretical & Mathematical Foundations

### A. Step 3: Stratified Train-Test Splitting
To evaluate machine learning models honestly and prevent data leakage, the stabilized cohort is divided into training and validation-testing sets. 

Because the target variable `hml32` has a baseline positive prevalence of only **18.52%**, we must enforce **stratification**. Stratified splitting guarantees that the ratio of positive to negative outcomes is identical across both training and testing datasets.

- **Split Ratio**: 80% Training ($N=194$) / 20% Testing ($N=49$)
- **Data Leakage Block**: Administrative columns (`hv001`, `hv002`, `hvidx`, `v005`, `analytical_weight`) are dropped completely from the feature space ($X$) so that the models train strictly on clinical and environmental exposures.

### B. Step 4: SMOTE-NC Oversampling
Standard classifiers optimize for global accuracy, causing them to completely ignore minority classes when faced with severe imbalance (e.g. classifying every record as negative for an 81% accuracy, yielding a **zero-sensitivity** clinical failure).

To fix this, we apply the **Synthetic Minority Over-sampling Technique for Nominal and Continuous (SMOTE-NC)**. Standard SMOTE only supports continuous variables, whereas SMOTE-NC handles mixed datasets:

1. **Continuous Attribute (`b19` - Age)**:
   Computes standard linear interpolation between a minority instance $x_i$ and one of its $k$-nearest neighbors $x_{nn}$ (with random factor $\lambda \in [0,1]$):
   
   $$x_{new} = x_i + \lambda \times (x_{nn} - x_i)$$

2. **Nominal Attributes (All other 11 variables)**:
   For nominal attributes (e.g., toilet type, wall material), SMOTE-NC computes a value distance metric and assigns the **local structural mode** (the most frequent category among the $k$-nearest neighbors) to the newly generated synthetic instance. This prevents generating invalid fractional codes (e.g. no values like `11.5` for floor materials).

> [!IMPORTANT]
> **Resampling Constraint**:
> SMOTE-NC is applied **strictly to the training partition only**. The validation-testing partition ($X_{\text{test}}, Y_{\text{test}}$) remains completely untouched and unbalanced. This ensures the testing set mimics the real-world clinical prevalence, preventing test-set data leakage.

---

## 2. Deliverable 1: Consolidated Partition & Imbalance Diagnostic Matrix

Upon running the pipeline, the class distributions across each step were tracked and validated:

| Dataset Partition | Total N | Negative N (0) | Negative % | Positive N (1) | Positive % | Imbalance Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original Full Dataset** | 243 | 198 | 81.48% | 45 | 18.52% | 4.40:1 |
| **Isolated Test Split (Step 3)** | 49 | 40 | 81.63% | 9 | 18.37% | 4.44:1 |
| **Original Train Split (Step 3)** | 194 | 158 | 81.44% | 36 | 18.56% | 4.39:1 |
| **Balanced Train Split (Step 4)** | **316** | **158** | **50.00%** | **158** | **50.00%** | **1.00:1** |

---

## 3. Algorithmic Verification & Output Persistence

### A. Discrete Categorical Verification
To guarantee the mathematical correctness of SMOTE-NC, the synthetic training rows were scanned for any fractional values across all nominal columns (`b4`, `v106`, `v190`, `v119`, `v158`, `v127`, `v128`, `v129`, `v113`, `v116`, `hml20`). 

* **Status**: **100% Verified Success**. All nominal fields contain discrete, structurally valid integer codes, with zero floating-point corruptions.

### B. Reproducibility
* **Seed Config**: `random_state=42` guarantees deterministic splitting and oversampling.
* **Neighborhood Scale**: $k=5$ nearest neighbors.

### C. Persistent Artifacts
The finalized partitions were written to separate CSV files to serve as clean inputs for downstream rule induction and baseline models:
1. `X_train_resampled.csv`: The balanced training feature matrix ($316 \times 12$).
2. `Y_train_resampled.csv`: The balanced training target vector ($316 \times 1$).
3. `X_test.csv`: The isolated stratified validation feature matrix ($49 \times 12$).
4. `Y_test.csv`: The isolated stratified validation target vector ($49 \times 1$).

---

## 4. Verification & Audit Trail

### How to Run
Execute the script using the local virtual environment interpreter:
```powershell
.venv\Scripts\python.exe third_step.py
```

### Verified Criteria
- **Dependency Loading**: `sklearn` and `imblearn` imported successfully without errors.
- **Stratified Partitioning**: The testing set preserves the severe baseline prevalence of ~18.37% positive cases.
- **Perfect Balance**: The balanced training set contains exactly 158 positive and 158 negative cases (perfect 50-50 class parity).
- **String Stabilization**: All nominal dimensions are cleanly cast to string formats to prevent tree algorithms from treating them as continuous digits.
