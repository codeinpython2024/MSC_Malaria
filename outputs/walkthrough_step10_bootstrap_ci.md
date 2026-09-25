# Walkthrough: Step 10 Non-Parametric Bootstrap Evaluation Engine (Examiner Items 5 & 9)

This document provides the walkthrough and academic audit trail for **Step 10: Non-Parametric Empirical Bootstrapping ($B = 1{,}000$ Replications) and Small-Sample Uncertainty Estimation on the Test Cohort** ($N = 49$), implemented in [`run_bootstrap_ci_final.py`](file:///c:/MSC_Malaria/run_bootstrap_ci_final.py).

---

## 1. Research Motivation & Examiner Items 5 & 9

The held-out test partition contains $N = 49$ children with exactly $N_{\text{pos}} = 9$ positive microscopy smear tests and $N_{\text{neg}} = 40$ negative tests. Under this sample size, point estimates are subject to substantial sampling variability.

To address Examiner Items 5 and 9, a non-parametric empirical bootstrapping engine was executed:
* **Replications**: $B = 1{,}000$ bootstrap resamples with replacement drawn from the test partition ($N = 49$).
* **Confidence Level**: Empirical 95% Percentile Confidence Intervals derived from the **2.5th and 97.5th percentiles** of the bootstrap metric distributions.

---

## 2. Table 4.8 Panel A: Positive Target Class Discrimination & 95% Bootstrap CIs

| Model Architecture | Global Accuracy (%) [95% CI] | Sensitivity (%) [95% CI] | Precision (Pos) (%) [95% CI] | F1-Score (Pos) (%) [95% CI] | ROC-AUC [95% CI] | PR-AUC [95% CI] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Class (Zero-R)** | 81.63% [71.4, 91.8] | 0.00% [0.0, 0.0] | 0.00% [0.0, 0.0] | 0.00% [0.0, 0.0] | 0.5000 [—] | 0.1837 [0.0816, 0.2857] |
| **Logistic Regression** | 71.43% [59.2, 83.7] | 22.22% [0.0, 54.5] | 22.22% [0.0, 57.1] | 22.22% [0.0, 46.2] | 0.5472 [0.31, 0.76] | 0.2001 [0.0714, 0.4581] |
| **C4.5 Decision Tree** | 57.14% [42.9, 71.4] | **33.33% [0.0, 71.4]** | 16.67% [0.0, 36.9] | 22.22% [0.0, 43.5] | 0.4792 [0.30, 0.67] | 0.3112 [0.0714, 0.5371] |
| **RIPPER Ruleset** | 75.51% [61.2, 87.8] | 0.00% [0.0, 0.0] | 0.00% [0.0, 0.0] | 0.00% [0.0, 0.0] | 0.4625 [0.42, 0.50] | 0.0918 [0.0408, 0.5918] |
| **Random Forest** | 73.47% [61.2, 85.7] | 22.22% [0.0, 54.6] | 25.00% [0.0, 60.0] | 23.53% [0.0, 50.0] | 0.5194 [0.27, 0.77] | 0.2025 [0.0793, 0.4234] |
| **Support Vector Machine** | **75.51% [63.3, 87.8]** | **33.33% [0.0, 66.7]** | **33.33% [0.0, 66.7]** | **33.33% [0.0, 59.3]** | 0.5083 [0.25, 0.77] | 0.2216 [0.0789, 0.5066] |

---

## 3. Table 4.8 Panel B: Negative Non-Target Class & Macro-Averaged Balance

| Model Architecture | Specificity (Neg) (%) [95% CI] | Precision (Neg) (%) [95% CI] | F1-Score (Neg) (%) [95% CI] | Macro Avg F1 (%) [95% CI] | Diagnostic Profile |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Majority Class (Zero-R)** | **100.00% [100.0, 100.0]** | 81.63% [71.4, 91.8] | **89.89% [83.3, 95.7]** | 44.94% [41.7, 47.9] | Trivial majority assignment |
| **Logistic Regression** | 82.50% [69.7, 93.3] | 82.50% [69.2, 93.6] | 82.50% [73.0, 90.5] | 52.36% [39.5, 66.7] | Symmetric error distribution |
| **C4.5 Decision Tree** | 62.50% [47.5, 77.5] | 80.65% [64.5, 93.8] | 70.42% [56.2, 81.6] | 46.32% [32.9, 60.6] | Minority recall prioritized; lower specificity |
| **RIPPER Ruleset** | 92.50% [83.3, 100.0] | 80.43% [68.1, 91.3] | 86.05% [75.9, 93.5] | 43.02% [38.0, 46.7] | Severe rule conservatism; 0% recall |
| **Random Forest** | 85.00% [72.2, 95.1] | 82.93% [70.3, 93.0] | 83.95% [74.4, 92.0] | 53.74% [40.2, 68.6] | Moderate minority detection |
| **Support Vector Machine** | 85.00% [73.2, 95.2] | **85.00% [73.2, 95.0]** | 85.00% [76.3, 92.7] | **59.17% [43.0, 74.4]** | Optimal harmonic trade-off |

---

## 4. Key Statistical Insights

1. **Wide Confidence Intervals on Minority Class**:
   Positive class precision and sensitivity exhibit 95% bootstrap intervals spanning up to 60 percentage points (e.g. $[0.0\%, 66.7\%]$ for SVM). This is the mathematical consequence of having $N_{\text{pos}} = 9$ instances in the held-out partition.
2. **Stable Macro F1 Intervals**:
   Macro F1 provides a more stable holistic assessment, with confidence intervals ranging from $[32.9\%, 60.6\%]$ for C4.5 to $[43.0\%, 74.4\%]$ for SVM.
3. **PR-AUC Bounds**:
   Precision-Recall AUC 95% intervals for all models overlap substantially with the random prevalence baseline ($0.1837$, 95% CI $[0.0816, 0.2857]$), reinforcing the conclusion that socio-demographic features do not provide strong independent diagnostic separation.

---

## 5. Deliverables & Audit Trail

* **Script**: [`run_bootstrap_ci_final.py`](file:///c:/MSC_Malaria/run_bootstrap_ci_final.py)
* **Execution Log**: [`outputs/logs/step10_bootstrap_ci.log`](file:///c:/MSC_Malaria/outputs/logs/step10_bootstrap_ci.log)
* **Underlying Array Artifact**: [`outputs/models/test_predictions.npz`](file:///c:/MSC_Malaria/outputs/models/test_predictions.npz)
