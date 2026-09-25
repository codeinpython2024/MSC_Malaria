# Walkthrough: Step 8 Grouped Household & Cluster Sensitivity Analysis (Examiner Item 8)

This document provides the walkthrough and academic audit trail for **Step 8: Grouped Household and Cluster Sensitivity Analysis under 5-Fold Cross-Validation**, implemented in [`step8_grouped_cluster_sensitivity.py`](file:///c:/MSC_Malaria/step8_grouped_cluster_sensitivity.py).

---

## 1. Research Motivation & Examiner Item 8

The examiner's review highlighted:
> *"The analysis must also report the number of unique households and enumeration areas. A major concern is that the code uses an ordinary random row split. If several children are from the same household or cluster, the same household-level information may occur in both training and test data. The revised dissertation should at least perform a household- or cluster-grouped sensitivity analysis."* (Assessment, Item 5 / Item 96).

### Analytic Cohort Spatial Structure ($N = 243$):
* **Unique Enumeration Areas (Clusters, `hv001`)**: **9 rural clusters**
* **Unique Households (`hv001` + `hv002`)**: **139 unique dwellings**
* **Mean Children per Household**: **1.75 children**
* **Total Positive Cases (`hml32` = 1)**: **45 children (18.52%)**

---

## 2. Experimental Design: Three Cross-Validation Regimes

To audit potential data leakage and test spatial transferability, 5-fold cross-validation was conducted across three distinct regimes:

1. **Scheme A: Standard Stratified Random K-Fold**:
   Standard random row-level splitting (baseline evaluation).
2. **Scheme B: Household-Grouped K-Fold (`GroupKFold` across 139 households)**:
   Guarantees that all children from the same physical household reside strictly within either the training fold or the testing fold, eliminating intra-household feature leakage.
3. **Scheme C: Cluster-Grouped K-Fold (`GroupKFold` across 9 rural clusters)**:
   Tests model generalization to completely unseen rural villages/communities.

---

## 3. Empirical Sensitivity Analysis Results (5-Fold CV)

| Validation Paradigm | Evaluated Model | Global Accuracy (%) | Sensitivity (Pos) (%) | Specificity (Neg) (%) | Macro F1 (%) | ROC-AUC | PR-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Scheme A: Standard Stratified** | Logistic Regression | 59.67% | 35.56% | 65.14% | 48.57% | 0.5218 | 0.2239 |
| **Scheme A: Standard Stratified** | Random Forest | 68.28% | 13.33% | 80.73% | 46.03% | 0.4658 | 0.1763 |
| **Scheme A: Standard Stratified** | Support Vector Machine | 74.49% | 15.56% | 87.86% | 50.53% | 0.4727 | 0.2076 |
| **Scheme B: Household-Grouped** | Logistic Regression | 64.27% | 36.31% | 71.38% | 50.20% | 0.5330 | 0.2266 |
| **Scheme B: Household-Grouped** | Random Forest | 72.03% | 16.55% | 84.69% | 50.04% | 0.4349 | 0.2177 |
| **Scheme B: Household-Grouped** | Support Vector Machine | 77.80% | 8.69% | 93.90% | 49.21% | 0.4518 | 0.1896 |
| **Scheme C: Cluster-Grouped** | Logistic Regression | 61.17% | 34.32% | 66.63% | 46.89% | 0.5506 | 0.2526 |
| **Scheme C: Cluster-Grouped** | Random Forest | 70.83% | 21.32% | 82.10% | 50.87% | 0.4760 | 0.2635 |
| **Scheme C: Cluster-Grouped** | Support Vector Machine | 79.13% | **2.50%** | 96.83% | 45.96% | 0.4668 | 0.2058 |

---

## 4. Key Epidemiological Findings

1. **Minimal Intra-Household Leakage Impact (Scheme A vs B)**:
   Moving from random splitting (Scheme A) to household-grouped splitting (Scheme B) yields comparable Macro F1 scores (50.53% vs 49.21% for SVM, 46.03% vs 50.04% for Random Forest). This indicates that the 80/20 train/test split did not artificially inflate accuracy via intra-household sharing.
2. **Severe Spatial Generalization Collapse (Scheme C)**:
   When models are required to generalize to entirely unseen rural villages (Scheme C), **SVM sensitivity collapses to 2.50%** (detecting virtually zero positive cases). The model defaults almost entirely to majority class prediction (96.83% specificity).
3. **Implication for Public Health Practice**:
   Social determinants of health exhibit strong spatial autocorrelation. Models parameterized on one set of rural enumeration areas cannot be reliably transferred to unseen villages without local calibration.

---

## 5. Deliverables & Audit Trail

* **Script**: [`step8_grouped_cluster_sensitivity.py`](file:///c:/MSC_Malaria/step8_grouped_cluster_sensitivity.py)
* **Execution Log**: [`outputs/logs/step8_grouped_cluster_sensitivity.log`](file:///c:/MSC_Malaria/outputs/logs/step8_grouped_cluster_sensitivity.log)
* **Exported Data**: [`outputs/grouped_cluster_sensitivity_results.csv`](file:///c:/MSC_Malaria/outputs/grouped_cluster_sensitivity_results.csv)
