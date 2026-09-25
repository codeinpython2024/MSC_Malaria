# Walkthrough: Step 7 Multi-Seed Stability Analysis (Examiner Item 6)

This document provides the walkthrough and academic audit trail for **Step 7: Empirical Multi-Seed Stability Analysis across 10 Distinct Random Train-Test Splits**, implemented in [`step7_multiseed_stability.py`](file:///c:/MSC_Malaria/step7_multiseed_stability.py).

---

## 1. Research Motivation & Examiner Item 6

The examiner's review identified that with $N_{\text{test}} = 49$ and only $N_{\text{pos}} = 9$ positive microscopy cases:
> *"Sensitivity estimates are therefore extremely unstable. One additional correctly detected positive changes sensitivity by approximately 11 percentage points. The dissertation reports point estimates but no confidence intervals, bootstrap intervals, repeated-split distributions, or uncertainty analysis."* (Assessment, Item 6 / Item 114).

To resolve this issue, the complete pipeline (stratified 80/20 train/test splitting, training-fold SMOTE-NC oversampling, and model fitting) was replicated across **10 distinct random seeds**:
$$\text{Seeds} \in [10, 20, 30, 42, 50, 60, 70, 80, 90, 100]$$

---

## 2. Multi-Seed Stability Results Matrix (Mean ± SD)

| Model Architecture | Global Accuracy (%) | Sensitivity (Pos) (%) | Specificity (Neg) (%) | Precision (Pos) (%) | Macro F1-Score (%) | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Class (Zero-R)** | 81.63% ± 0.00% | 0.00% ± 0.00% | 100.00% ± 0.00% | 0.00% ± 0.00% | 44.94% ± 0.00% | 0.5000 ± 0.0000 | 0.1837 ± 0.0000 |
| **Logistic Regression** | 64.08% ± 7.15% | **35.56% ± 15.54%** | 70.50% ± 8.88% | 21.96% ± 8.57% | **51.20% ± 6.56%** | **0.5533 ± 0.0849** | **0.2117 ± 0.0398** |
| **Decision Tree (C4.5)** | 63.88% ± 3.73% | 20.00% ± 10.21% | 73.75% ± 6.04% | 13.87% ± 5.91% | 46.51% ± 3.33% | 0.4389 ± 0.1093 | 0.1938 ± 0.0835 |
| **Random Forest** | 68.57% ± 6.02% | 14.44% ± 11.77% | 80.75% ± 7.17% | 13.97% ± 10.65% | 47.27% ± 6.41% | 0.4536 ± 0.1136 | 0.2087 ± 0.0816 |
| **Support Vector Machine** | **74.49% ± 4.54%** | 13.33% ± 12.61% | **88.25% ± 4.26%** | **19.29% ± 19.67%** | 50.25% ± 8.42% | 0.4606 ± 0.0933 | 0.1956 ± 0.0664 |

---

## 3. Academic & Methodological Findings

1. **Empirical Verification of Small-Sample Variance**:
   The standard deviation in test sensitivity reaches **±15.54%** for Logistic Regression and **±12.61%** for SVM. This confirms the examiner's theoretical point: single-split point estimates in small imbalanced test sets carry substantial sample-dependent variance.
2. **Comparison against Zero-R**:
   None of the evaluated machine learning models reliably beats the Zero-R majority class accuracy (**81.63%**). This mathematically demonstrates that high accuracy in imbalanced health survey data can be misleading.
3. **ROC-AUC Distributions**:
   Across all 10 seeds, the mean ROC-AUC of C4.5 (**0.4389**), Random Forest (**0.4536**), and SVM (**0.4606**) remain bounded around the 0.5000 random-chance boundary, verifying that weak discriminative performance is persistent across partitions.

---

## 4. Deliverables & Audit Trail

* **Script**: [`step7_multiseed_stability.py`](file:///c:/MSC_Malaria/step7_multiseed_stability.py)
* **Execution Log**: [`outputs/logs/step7_multiseed_stability.log`](file:///c:/MSC_Malaria/outputs/logs/step7_multiseed_stability.log)
* **Exported Data**: [`outputs/multiseed_stability_results.csv`](file:///c:/MSC_Malaria/outputs/multiseed_stability_results.csv)
