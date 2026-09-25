# Walkthrough: Step 9 Rule Coverage, Support Profiling, and Calibrated ROC (Examiner Items 10 & 11)

This document provides the walkthrough and academic audit trail for **Step 9: Propositional Rule Support, Coverage Auditing, and Calibrated Continuous ROC Implementation**, implemented in [`step9_rule_coverage_support.py`](file:///c:/MSC_Malaria/step9_rule_coverage_support.py).

---

## 1. Research Motivation & Examiner Items 10 & 11

The examiner's review required:
> *"The extracted rules should be interpreted as candidate patterns. Their current presentation omits support, precision, recall, coverage, confidence intervals, and stability. A rule that appears in one tree trained after synthetic oversampling should not be presented as a validated epidemiological pathway."* (Assessment, Item 7 / Item 128).
> *"The ROC analysis also requires correction... That does not constitute a valid continuous probability score across the full threshold range."* (Assessment, Item 6 / Item 115).

---

## 2. RIPPER Propositional Rule Support & Coverage Audit Matrix

Evaluated across resampled training data ($N = 316$) and untouched test data ($N = 49$):

| Rule Statement | Evaluation Dataset | Support ($n$) | Population Coverage (%) | True Positives (TP) | False Positives (FP) | Rule Precision (%) | Class Recall (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rule R1**: $[v113=31 \wedge v127=34 \wedge v116=31 \wedge b4=2 \wedge v129=31 \wedge hml20=0]$ | Resampled Train ($N=316$) | 56 | 17.72% | 52 | 4 | 92.86% | 32.91% |
| **Rule R2**: $[v128=21 \wedge v129=31 \wedge 14 < b19 \le 17]$ | Resampled Train ($N=316$) | 17 | 5.38% | 15 | 2 | 88.24% | 9.49% |
| **Default Rule**: $[\text{ELSE} \Rightarrow \text{Class 0}]$ | Resampled Train ($N=316$) | 243 | 76.90% | 152 | 91 | 62.55% | 96.20% |
| **Rule R1**: $[v113=31 \wedge v127=34 \wedge v116=31 \wedge b4=2 \wedge v129=31 \wedge hml20=0]$ | Clinical Test ($N=49$) | 2 | 4.08% | 0 | 2 | 0.00% | 0.00% |
| **Rule R2**: $[v128=21 \wedge v129=31 \wedge 14 < b19 \le 17]$ | Clinical Test ($N=49$) | 1 | 2.04% | 0 | 1 | 0.00% | 0.00% |
| **Default Rule**: $[\text{ELSE} \Rightarrow \text{Class 0}]$ | Clinical Test ($N=49$) | 46 | 93.88% | 37 (TN) | 9 (FN) | 80.43% | 92.50% |

> [!WARNING]
> **Audit Finding**:
> In the held-out test partition ($N = 49$), Rules R1 and R2 fire on only **3 instances total** (all 3 are false positives, capturing 0 true positives). The default rule captures **93.88%** of test children, missing all 9 positive cases. This confirms that RIPPER's rules are candidate exploratory patterns rather than robust clinical decision rules.

---

## 3. C4.5 Decision Pathway Support & Coverage Audit Matrix

Evaluated across resampled training ($N = 316$) and clinical test ($N = 49$) cohorts:

| Decision Pathway Description | Evaluation Dataset | Support ($n$) | Population Coverage (%) | True Positives (TP) | False Positives (FP) | Path Precision (%) | Target Recall (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pathway C1**: Infancy Protection $[b19 \le 6 \Rightarrow \text{Neg}]$ | Resampled Train ($N=316$) | 5 | 1.58% | 5 | 0 | 100.00% | 3.16% |
| **Pathway C2**: Protected Well $\times$ Metal Roof $[b19>6 \wedge v158=0 \wedge hml20=0 \wedge v113=31 \wedge v129=31 \Rightarrow \text{Pos}]$ | Resampled Train ($N=316$) | 124 | 39.24% | 100 | 24 | 80.65% | 63.29% |
| **Pathway C3**: Rainwater $\times$ Maternal Illiteracy $[b19>6 \wedge v158=0 \wedge hml20=0 \wedge v113=43 \wedge v106=0 \Rightarrow \text{Pos}]$ | Resampled Train ($N=316$) | 49 | 15.51% | 26 | 23 | 53.06% | 16.46% |
| **Pathway C4**: Borehole $\times$ Mud Walls $[b19>6 \wedge v158=0 \wedge hml20=0 \wedge v113 \in \{21, 32\} \wedge v128=21 \Rightarrow \text{Pos}]$ | Resampled Train ($N=316$) | 28 | 8.86% | 17 | 11 | 60.71% | 10.76% |
| **Pathway C5**: Intervention Paradox $[b19>6 \wedge v158=0 \wedge hml20=1 \wedge v119=1 \wedge v190 \le 2 \Rightarrow \text{Pos}]$ | Resampled Train ($N=316$) | 1 | 0.32% | 1 | 0 | 100.00% | 0.63% |
| **Pathway C1**: Infancy Protection $[b19 \le 6 \Rightarrow \text{Neg}]$ | Clinical Test ($N=49$) | 1 | 2.04% | 1 | 0 | 100.00% | 2.50% |
| **Pathway C2**: Protected Well $\times$ Metal Roof | Clinical Test ($N=49$) | 9 | 18.37% | 2 | 7 | 22.22% | 22.22% |
| **Pathway C3**: Rainwater $\times$ Maternal Illiteracy | Clinical Test ($N=49$) | 5 | 10.20% | 0 | 5 | 0.00% | 0.00% |
| **Pathway C4**: Borehole $\times$ Mud Walls | Clinical Test ($N=49$) | 4 | 8.16% | 1 | 3 | 25.00% | 11.11% |
| **Pathway C5**: Intervention Paradox | Clinical Test ($N=49$) | 0 | 0.00% | 0 | 0 | 0.00% | 0.00% |

> [!NOTE]
> **Intervention Paradox Audit**:
> Pathway C5 (electricity + bednet adherence + poverty predicting positive) was supported by exactly $n = 1$ synthetic observation in the SMOTE-NC training set and had $n = 0$ test observations. It is an artifactual outlier rather than a reproducible epidemiological rule.

---

## 4. Continuous Calibrated ROC vs Hard Threshold Projections (Item 11)

| Evaluation Metric | Hard Binary Step Curve (Old) | Continuous Calibrated Leaf Curve (New) |
| :--- | :---: | :---: |
| **ROC-AUC** | 0.4792 | 0.4653 |
| **PR-AUC** | 0.3112 | 0.1570 |
| **Operating Thresholds** | 3 points (trivial step) | 10 distinct threshold cutpoints |

---

## 5. Deliverables & Audit Trail

* **Script**: [`step9_rule_coverage_support.py`](file:///c:/MSC_Malaria/step9_rule_coverage_support.py)
* **Execution Log**: [`outputs/logs/step9_rule_coverage_support.log`](file:///c:/MSC_Malaria/outputs/logs/step9_rule_coverage_support.log)
* **Exported Data**:
  - [`outputs/ripper_rule_coverage_support.csv`](file:///c:/MSC_Malaria/outputs/ripper_rule_coverage_support.csv)
  - [`outputs/c45_pathway_coverage_support.csv`](file:///c:/MSC_Malaria/outputs/c45_pathway_coverage_support.csv)
