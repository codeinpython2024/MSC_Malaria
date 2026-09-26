# Master Reproducibility Guide & Computational Audit Trail

**Project Title**: Machine Learning and Rule Induction for Pediatric Malaria Screening Using Social Determinants of Health  
**Target Study Cohort**: Rural Nasarawa State, Nigeria (2021 Nigeria Malaria Indicator Survey - NMIS)  
**Primary Research Goal**: Comparative evaluation of white-box rule-based models (C4.5 Decision Tree, RIPPER) versus black-box benchmarks (Random Forest, Support Vector Machine) and linear/majority-class baselines using survey-weighted demographic and environmental exposures.  
**Dissertation Manuscripts Synchronized**: [`Full_Work.docx`](file:///c:/MSC_Malaria/Full_Work.docx) & [`Full_Work_Revised.docx`](file:///c:/MSC_Malaria/Full_Work_Revised.docx)

---

## 1. Executive Summary & Supervisor Verification Checklist

This document serves as the formal academic reproducibility audit trail, explicitly fulfilling the supervisor and examiner requirements set forth in the MSc Computer Science Assessment (Items 1 through 14):

> *"The author should provide a requirements file or environment lockfile, exact package versions, the full scripts outside Word formatting, and execution logs showing that the reported tables were generated from the supplied code."*

### Compliance Matrix

| Requirement | Implementation Artifact | Location |
| :--- | :--- | :--- |
| **Requirements File** | Standard pinned pip manifest | [`requirements.txt`](file:///c:/MSC_Malaria/requirements.txt) |
| **Environment Lockfile** | Cryptographic SHA-256 deterministic lockfile | [`uv.lock`](file:///c:/MSC_Malaria/uv.lock) |
| **Project Definition** | Declarative project metadata & dependencies | [`pyproject.toml`](file:///c:/MSC_Malaria/pyproject.toml) |
| **Full Scripts (Outside Word)** | Standalone, pure `.py` source scripts (Stages 0–11) | Project root (`pre_process.py` through `run_bootstrap_ci_final.py`) |
| **Automated Log Engine** | Single-command master pipeline runner | [`generate_execution_logs.py`](file:///c:/MSC_Malaria/generate_execution_logs.py) |
| **Execution Logs & Table Proofs** | Complete console stdout execution logs (11 files) | [`outputs/logs/`](file:///c:/MSC_Malaria/outputs/logs/) (`step0_data_ingestion.log` through `step10_bootstrap_ci.log`) |
| **Walkthrough Audit Records** | Detailed mathematical walkthrough documents (8 files) | [`outputs/walkthrough_step*.md`](file:///c:/MSC_Malaria/outputs/) |
| **Sensitivity & Uncertainty Data** | Exported empirical CSV deliverables | [`outputs/multiseed_stability_results.csv`](file:///c:/MSC_Malaria/outputs/multiseed_stability_results.csv), [`outputs/grouped_cluster_sensitivity_results.csv`](file:///c:/MSC_Malaria/outputs/grouped_cluster_sensitivity_results.csv), [`outputs/ripper_rule_coverage_support.csv`](file:///c:/MSC_Malaria/outputs/ripper_rule_coverage_support.csv), [`outputs/c45_pathway_coverage_support.csv`](file:///c:/MSC_Malaria/outputs/c45_pathway_coverage_support.csv) |
| **Model & Graphic Deliverables** | Serialized ML models (`.pkl`, `.npz`) & 300 DPI plots | [`outputs/models/`](file:///c:/MSC_Malaria/outputs/models/) and [`outputs/plots/`](file:///c:/MSC_Malaria/outputs/plots/) |
| **Thesis Synchronization Engine** | Programmatic docx patching and figure synchronization | [`build_full_work_revised.py`](file:///c:/MSC_Malaria/build_full_work_revised.py) |
| **Examiner Technical & Rationale Guide** | Dedicated script-by-script and artifact explanation guide | [`EXAMINER_TECHNICAL_GUIDE.md`](file:///c:/MSC_Malaria/EXAMINER_TECHNICAL_GUIDE.md) |

---

## 2. Computational Environment & Exact Package Versions

### A. Core Runtime Specification
* **Python Runtime**: Python 3.14+ (Verified on CPython 3.14.5, x86_64)
* **Virtual Environment**: `.venv` (Managed via `uv` or standard Python `venv`)
* **Operating System**: Platform independent (Tested on Windows 11 / PowerShell; fully POSIX compliant for Linux / macOS)

### B. Pinned Dependency Roster
All package versions are strictly pinned in [`requirements.txt`](file:///c:/MSC_Malaria/requirements.txt):

| Package Name | Exact Version | Role in Research Pipeline |
| :--- | :---: | :--- |
| `pandas` | **3.0.3** | Stata `.DTA` ingestion, survey weight normalization, data aggregation |
| `scikit-learn` | **1.9.0** | Stratified splitting, Logistic Regression, Random Forest, SVM, ROC/PR curves |
| `scipy` | **1.18.0** | Sparse matrix representation, mathematical backend computations |
| `numpy` | **2.5.1** | Vectorized array transformations and continuous threshold arithmetic |
| `imbalanced-learn` (`imblearn`) | **0.14.2** | SMOTE-NC (Synthetic Minority Over-sampling for Nominal and Continuous) |
| `chefboost` | **0.0.19** | Information gain ratio and C4.5 decision tree rule construction |
| `wittgenstein` | **0.3.5** | Repeated Incremental Pruning to Produce Error Reduction (RIPPER) |
| `matplotlib` | **3.11.0** | Publication-grade graphics generation (ROC curves, confusion matrices) |
| `joblib` | **1.5.3** | Serialization of fitted scikit-learn pipeline objects |
| `python-docx` | **1.2.0** | Programmatic Word document compilation and table formatting |

### C. Environment Setup & Replication Instructions

#### Option 1: Fast Setup via `uv` (Recommended)
```powershell
# 1. Installs exact locked dependencies from uv.lock deterministically
uv sync

# 2. Execute the entire pipeline and generate all execution logs in one command:
uv run python generate_execution_logs.py
```

#### Option 2: Standard Virtual Environment via `pip`
```powershell
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate virtual environment
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# 3. Install the exact package versions
pip install -r requirements.txt

# 4. Execute the master pipeline log engine
python generate_execution_logs.py
```

---

## 3. Directory Layout & Pipeline Artifacts

```
c:\MSC_Malaria\
├── NGKR81DT\                             # Raw DHS Stata Child Recode directory
│   └── NGKR81FL.DTA                      # Stata binary file (Children 0-59 months, N=10,988)
├── NGPR81DT\                             # Raw DHS Stata Household Member Recode directory
│   └── NGPR81FL.DTA                      # Stata binary file (Biomarker and bednet records)
│
├── pre_process.py                        # Stage 0: Ingestion, participant filtering & relational merge
├── first_stage.py                        # Stage 1: Categorical stabilization & SMOTE-NC index tracking
├── second_step.py                        # Stage 2: Survey weight normalization & baseline diagnostics
├── third_step.py                         # Stages 3 & 4: Stratified 80/20 partition & training SMOTE-NC
├── fourth_step.py                        # Stage 5: White-Box Induction (C4.5 & RIPPER)
├── fifth_step.py                         # Stage 6: Black-Box Benchmark Baselines (RF & SVM tuning)
├── sixth_step.py                         # Stage 7: Test evaluation, Calibrated ROC & Trade-off matrix
├── step7_multiseed_stability.py          # Stage 8: 10-seed empirical stability analysis (Item 6)
├── step8_grouped_cluster_sensitivity.py  # Stage 9: Household & cluster sensitivity analysis (Item 8)
├── step9_rule_coverage_support.py        # Stage 10: Rule coverage, support & calibrated ROC (Items 10 & 11)
├── run_bootstrap_ci_final.py             # Stage 11: Non-parametric bootstrap evaluation engine (Items 5 & 9)
├── generate_execution_logs.py            # Master automation runner for complete log generation
├── build_full_work_revised.py            # Master thesis document patching & figure synchronization
│
├── requirements.txt                      # Pinned package requirements manifest
├── pyproject.toml                        # Project configuration & dependency declarations
├── uv.lock                               # Exact cryptographic lockfile
├── REPRODUCIBILITY.md                    # This master audit and reproduction document
├── EXAMINER_TECHNICAL_GUIDE.md          # Comprehensive script-by-script & artifact guide for examiners
├── Full_Work.docx                        # Primary synchronized dissertation document
├── Full_Work_Revised.docx                # Synchronized revised dissertation document
├── assessment_full_text.txt              # Examiner review text and 14 evaluation requirements
│
├── pre_processed_malaria_sdoh_matrix.csv # Intermediate merged feature matrix (N=243, 17 cols)
├── final_malaria_sdoh_matrix.csv         # Stabilized ML matrix with integer/string typing (N=243)
├── X_train_resampled.csv                 # SMOTE-NC balanced training predictors (N=316, 12 cols)
├── Y_train_resampled.csv                 # SMOTE-NC balanced training target (50/50 balance)
├── X_test.csv                            # Isolated untouched clinical test predictors (N=49)
├── Y_test.csv                            # Isolated untouched clinical test ground truth
│
└── outputs\
    ├── logs\                             # Authentic console stdout execution logs (11 files)
    │   ├── step0_data_ingestion.log      # Stage 0: NMIS filtering from 10,988 to 243 children
    │   ├── step1_categorical_mapping.log # Stage 1: Categorical stabilization & SMOTE-NC indices
    │   ├── step2_baseline_diagnostics.log# Stage 2: Survey weight normalization (Tables 4.1–4.5)
    │   ├── step3_partition.log           # Stages 3 & 4: 80/20 train/test split & SMOTE-NC (Table 4.6)
    │   ├── step4_rule_induction.log      # Stage 5: C4.5 tree & RIPPER ruleset generation
    │   ├── step5_black_box_baselines.log # Stage 6: 5-fold GridSearchCV for RF and SVM (Table 4.7)
    │   ├── step6_comparative_evaluation.log # Stage 7: Test set evaluation & trade-off matrix (Table 4.8)
    │   ├── step7_multiseed_stability.log # Stage 8: 10-seed empirical stability distributions
    │   ├── step8_grouped_cluster_sensitivity.log # Stage 9: Schemes A, B, C 5-fold CV comparisons
    │   ├── step9_rule_coverage_support.log # Stage 10: Antecedent support, coverage & calibrated ROC
    │   └── step10_bootstrap_ci.log       # Stage 11: Non-parametric bootstrap percentile CIs
    │
    ├── models\                           # Serialized models and test predictions
    │   ├── random_forest_pipeline.pkl    # Serialized tuned Random Forest model
    │   ├── svm_pipeline.pkl              # Serialized tuned Support Vector Machine model
    │   └── test_predictions.npz          # Serialized predictions and probabilities for test cohort
    │
    ├── plots\                            # Publication-grade figures (300 DPI)
    │   ├── decision_tree_plot.png        # Figure 4.5: C4.5 Decision Tree topology
    │   ├── roc_comparison.png            # Figure 4.4: Joint multi-model continuous ROC curve
    │   ├── confusion_matrices.png        # Figure 4.3: Comparative 2x2 confusion matrices
    │   ├── model_performance_comparison.png # Figure 4.2: Bar chart of comparative performance
    │   ├── sdoh_prevalence.png           # Figure 4.1: Bivariate malaria prevalence by SDOH factors
    │   └── interpretability_vs_accuracy.png # Figure 4.6: Complexity vs Macro F1 Pareto frontier
    │
    ├── multiseed_stability_results.csv   # 10-seed replication results (Mean ± SD)
    ├── grouped_cluster_sensitivity_results.csv # Household & cluster 5-fold sensitivity metrics
    ├── ripper_rule_coverage_support.csv  # RIPPER rule support, coverage, precision, and recall
    ├── c45_pathway_coverage_support.csv  # C4.5 pathway support, coverage, precision, and recall
    │
    ├── c45_visualization_discrepancy_note.md # Methodological note on C4.5 vs CART root split
    ├── walkthrough_step2.md              # Academic audit trail for Baseline Survey Diagnostics
    ├── walkthrough_step3.md              # Academic audit trail for Partitioning & SMOTE-NC
    ├── walkthrough_step4.md              # Academic audit trail for C4.5 & RIPPER Rule Induction
    ├── walkthrough_step5.md              # Academic audit trail for Black-Box Baselines
    ├── walkthrough_step6.md              # Academic audit trail for Comparative Evaluation & ROC
    ├── walkthrough_step7_stability.md    # Academic audit trail for 10-Seed Stability Analysis
    ├── walkthrough_step8_cluster_sensitivity.md # Academic audit trail for Grouped CV Sensitivity
    ├── walkthrough_step9_rule_coverage.md# Academic audit trail for Rule Coverage & Support
    └── walkthrough_step10_bootstrap_ci.md# Academic audit trail for Bootstrap Uncertainty
```

---

## 4. Sequential Execution Guide (Outside Word Formatting)

Every stage of the analysis pipeline is encapsulated in a dedicated, pure Python script. All scripts can be executed independently or sequentially via [`generate_execution_logs.py`](file:///c:/MSC_Malaria/generate_execution_logs.py):

### Stage 0: Data Ingestion & Geopolitical Filtering
* **Script**: [`pre_process.py`](file:///c:/MSC_Malaria/pre_process.py)
* **Command**: `python pre_process.py`
* **Log File**: [`outputs/logs/step0_data_ingestion.log`](file:///c:/MSC_Malaria/outputs/logs/step0_data_ingestion.log)
* **Mathematical Operations**:
  - Ingests raw Stata binaries (`NGKR81FL.DTA` and `NGPR81FL.DTA`).
  - Biological survival filter: Alive children only (`b5 == 1`), $10{,}988 \rightarrow 10{,}645$.
  - Household residency filter: De facto household residents (`b16 > 0`), $10{,}645 \rightarrow 10{,}469$.
  - Biological age window: 6 to 59 months (`6 <= b19 <= 59`), $10{,}469 \rightarrow 9{,}510$.
  - Geopolitical spatial filter: Nasarawa State (`v024 == 15`, $n = 330$) and Rural settlements (`v025 == 2`, $n = 245$).
  - Relational Left-Join on composite keys (`hv001`, `hv002`, `hvidx`).
  - Target outcome: Valid microscopy results (`hml32 in [0, 1]`, $n = 243$).
  - Vector control adherence: Valid bed net record (`hml20 in [0, 1]`, $n = 243$).
* **Yield**: Cleaned analytical cohort of **$N = 243$** children saved to `pre_processed_malaria_sdoh_matrix.csv`. Proves Dissertation Chapter 3 Table 3.1 (Participant Flow).

---

### Stage 1: Categorical Mapping & Value Stabilization
* **Script**: [`first_stage.py`](file:///c:/MSC_Malaria/first_stage.py)
* **Command**: `python first_stage.py`
* **Log File**: [`outputs/logs/step1_categorical_mapping.log`](file:///c:/MSC_Malaria/outputs/logs/step1_categorical_mapping.log)
* **Mathematical Operations**:
  - Enforces continuous vector space for child's age (`b19`) to retain variance for binary split thresholds ($\theta$).
  - Preserves ordinal constraint on DHS wealth quintile (`v190`, values 1–5) without manual PCA re-indexing.
  - Sanitizes nominal predictor columns (`b4`, `v106`, `v127`, `v128`, `v129`, `v113`, `v116`, `v119`, `v158`, `hml20`) into clean string representations.
  - Constructs explicit integer index tracking array (`[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]`) strictly required by SMOTE-NC.
* **Yield**: Stabilized dataset saved to `final_malaria_sdoh_matrix.csv`.

---

### Stage 2: Complex Survey Weight Normalization & Baseline Diagnostics
* **Script**: [`second_step.py`](file:///c:/MSC_Malaria/second_step.py)
* **Command**: `python second_step.py`
* **Log File**: [`outputs/logs/step2_baseline_diagnostics.log`](file:///c:/MSC_Malaria/outputs/logs/step2_baseline_diagnostics.log)
* **Mathematical Operations**:
  - Analytical weight calculation: $w_i = \frac{\text{v005}_i}{1{,}000{,}000}$ (Sum of analytical weights = $185.0152$).
  - Target class imbalance calculation on microscopy smear test (`hml32`): 198 Negative (81.24% weighted), 45 Positive (18.76% weighted). Ratio: 4.33:1.
  - Weighted univariate percentage: $\text{Pct}_c = \left( \frac{\sum w_i \cdot I(x_i = c)}{\sum w_i} \right) \times 100$.
  - Weighted bivariate malaria prevalence: $P_c = \left( \frac{\sum w_i \cdot I(\text{hml32}_i = 1)}{\sum w_i} \right) \times 100$.
* **Yield**: Directly proves Dissertation Chapter 4 Tables 4.1, 4.2, 4.3, 4.4, 4.5 and generates Figure 4.1 ([`outputs/plots/sdoh_prevalence.png`](file:///c:/MSC_Malaria/outputs/plots/sdoh_prevalence.png)).

---

### Stages 3 & 4: Stratified Train-Test Partitioning & SMOTE-NC Resampling
* **Script**: [`third_step.py`](file:///c:/MSC_Malaria/third_step.py)
* **Command**: `python third_step.py`
* **Log File**: [`outputs/logs/step3_partition.log`](file:///c:/MSC_Malaria/outputs/logs/step3_partition.log)
* **Mathematical Operations**:
  - Stratified 80/20 train-test partition (`random_state=42`, `stratify=y`), isolating $N_{\text{train}} = 194$ (80%) and $N_{\text{test}} = 49$ (20%).
  - Drops administrative/tracking metadata (`hv001`, `hv002`, `hvidx`, `v005`, `analytical_weight`) from $X$ to prevent target/administrative leakage.
  - Synthetic Minority Over-sampling for Nominal and Continuous (SMOTE-NC) applied **strictly to the training split**.
  - Nominal features assigned local structural mode among $k = 5$ nearest neighbors (no fractional code corruptions).
* **Yield**: Balanced training set of **$N = 316$** (158 Positive / 158 Negative) saved to `X_train_resampled.csv` and `Y_train_resampled.csv`; untouched clinical test split saved to `X_test.csv` and `Y_test.csv`. Proves Dissertation Chapter 4 Table 4.6.

---

### Stage 5: White-Box Rule Induction (C4.5 & RIPPER)
* **Script**: [`fourth_step.py`](file:///c:/MSC_Malaria/fourth_step.py)
* **Command**: `python fourth_step.py`
* **Log File**: [`outputs/logs/step4_rule_induction.log`](file:///c:/MSC_Malaria/outputs/logs/step4_rule_induction.log)
* **Mathematical Operations**:
  - Induces C4.5 Decision Tree optimizing Information Gain Ratio:
    $$H(S) = -\sum p_i \log_2 p_i, \quad \text{GainRatio}(A) = \frac{IG(S, A)}{\text{SplitInfo}(S, A)}$$
  - Bottom-up post-pruning to extract human-readable IF-THEN conditional rules (`outputs/rules/rules.py`).
  - Induces RIPPER classifier using Wittgenstein framework maximizing FOIL Information Gain:
    $$\text{FOIL}_{IG} = p_1 \left( \log_2 \frac{p_1}{p_1 + n_1} - \log_2 \frac{p_0}{p_0 + n_0} \right)$$
* **Yield**: Extracted rule structures proving Dissertation Chapter 4 Tables 4.10 & 4.11 and 300 DPI plot Figure 4.5 ([`outputs/plots/decision_tree_plot.png`](file:///c:/MSC_Malaria/outputs/plots/decision_tree_plot.png)).

---

### Stage 6: Black-Box Benchmark Optimization (Random Forest & SVM)
* **Script**: [`fifth_step.py`](file:///c:/MSC_Malaria/fifth_step.py)
* **Command**: `python fifth_step.py`
* **Log File**: [`outputs/logs/step5_black_box_baselines.log`](file:///c:/MSC_Malaria/outputs/logs/step5_black_box_baselines.log)
* **Mathematical Operations**:
  - Constructs preprocessor pipelines (`OneHotEncoder(handle_unknown='ignore')` + `StandardScaler`).
  - Trains ensemble `RandomForestClassifier` with 5-fold cross-validated grid search over `n_estimators`, `max_depth`, `min_samples_split`.
  - Trains kernel-based `SVC` with grid search over kernel types (`linear`, `rbf`) and regularization parameters ($C$, $\gamma$).
* **Yield**: Optimal hyperparameter configurations proving Dissertation Chapter 4 Table 4.7; serializes pipelines to [`outputs/models/random_forest_pipeline.pkl`](file:///c:/MSC_Malaria/outputs/models/random_forest_pipeline.pkl) and [`outputs/models/svm_pipeline.pkl`](file:///c:/MSC_Malaria/outputs/models/svm_pipeline.pkl).

---

### Stage 7: Multi-Criteria Model Evaluation & Thesis Synthesis
* **Script**: [`sixth_step.py`](file:///c:/MSC_Malaria/sixth_step.py)
* **Command**: `python sixth_step.py`
* **Log File**: [`outputs/logs/step6_comparative_evaluation.log`](file:///c:/MSC_Malaria/outputs/logs/step6_comparative_evaluation.log)
* **Mathematical Operations**:
  - Passes untouched test partition ($N = 49$, 40 Negative / 9 Positive) through Zero-R, Logistic Regression, C4.5, RIPPER, Random Forest, and SVM.
  - Computes exact Confusion Matrices, Global Accuracy, Sensitivity/Recall (Malaria+), Specificity (Malaria-), Precision, and Macro F1-Score.
  - Generates joint Receiver Operating Characteristic (ROC) curve using continuous Laplace-smoothed leaf probabilities for C4.5 and multi-threshold continuous curves for benchmarks.
  - Profiles structural complexity (terminal rules, literals per rule) for accuracy-vs-interpretability trade-off.
* **Yield**: Proves Dissertation Chapter 4 Tables 4.8 & 4.9; generates Figures 4.2, 4.3, 4.4, and 4.6 in `outputs/plots/`; serializes [`outputs/models/test_predictions.npz`](file:///c:/MSC_Malaria/outputs/models/test_predictions.npz).

---

### Stage 8: Empirical Multi-Seed Stability Analysis (Examiner Item 6)
* **Script**: [`step7_multiseed_stability.py`](file:///c:/MSC_Malaria/step7_multiseed_stability.py)
* **Command**: `python step7_multiseed_stability.py`
* **Log File**: [`outputs/logs/step7_multiseed_stability.log`](file:///c:/MSC_Malaria/outputs/logs/step7_multiseed_stability.log)
* **Mathematical Operations**:
  - Partitions $N = 243$ cohort across 10 random seeds (`[10, 20, 30, 42, 50, 60, 70, 80, 90, 100]`) into stratified 80/20 train/test splits.
  - Applies SMOTE-NC strictly to the training partition on each seed.
  - Evaluates Zero-R, Logistic Regression, C4.5, Random Forest, and SVM.
  - Computes Mean $\pm$ Standard Deviation across all 10 seeds.
* **Yield**: Generates [`outputs/multiseed_stability_results.csv`](file:///c:/MSC_Malaria/outputs/multiseed_stability_results.csv) proving sensitivity variance up to $\pm 15.54\%$, verifying Dissertation Chapter 4 Section 4.3 discussion.

---

### Stage 9: Grouped Household & Cluster Sensitivity Analysis (Examiner Item 8)
* **Script**: [`step8_grouped_cluster_sensitivity.py`](file:///c:/MSC_Malaria/step8_grouped_cluster_sensitivity.py)
* **Command**: `python step8_grouped_cluster_sensitivity.py`
* **Log File**: [`outputs/logs/step8_grouped_cluster_sensitivity.log`](file:///c:/MSC_Malaria/outputs/logs/step8_grouped_cluster_sensitivity.log)
* **Mathematical Operations**:
  - Implements 5-Fold Cross-Validation across 3 distinct partitioning regimes:
    1. Scheme A: Standard Stratified Random K-Fold.
    2. Scheme B: Household-Grouped K-Fold (139 unique households; isolates intra-household correlation).
    3. Scheme C: Cluster-Grouped K-Fold (9 rural clusters; tests generalization to unseen villages).
* **Yield**: Generates [`outputs/grouped_cluster_sensitivity_results.csv`](file:///c:/MSC_Malaria/outputs/grouped_cluster_sensitivity_results.csv), documenting SVM sensitivity collapse to 2.50% under unseen village clusters, verifying Dissertation Chapter 4 Section 4.3 discussion.

---

### Stage 10: Rule Coverage, Support & Calibrated ROC (Examiner Items 10 & 11)
* **Script**: [`step9_rule_coverage_support.py`](file:///c:/MSC_Malaria/step9_rule_coverage_support.py)
* **Command**: `python step9_rule_coverage_support.py`
* **Log File**: [`outputs/logs/step9_rule_coverage_support.log`](file:///c:/MSC_Malaria/outputs/logs/step9_rule_coverage_support.log)
* **Mathematical Operations**:
  - Computes Antecedent Support ($n$), Population Coverage (%), Rule Precision/Confidence (%), and Class Recall (%) for RIPPER and C4.5 across training ($N=316$) and test ($N=49$) partitions.
  - Replaces binary $\{0, 1\}$ ROC step curves with continuous Laplace-smoothed leaf probability estimates across 10 operating thresholds.
* **Yield**: Generates [`outputs/ripper_rule_coverage_support.csv`](file:///c:/MSC_Malaria/outputs/ripper_rule_coverage_support.csv) and [`outputs/c45_pathway_coverage_support.csv`](file:///c:/MSC_Malaria/outputs/c45_pathway_coverage_support.csv).

---

### Stage 11: Non-Parametric Empirical Bootstrapping Engine (Examiner Items 5 & 9)
* **Script**: [`run_bootstrap_ci_final.py`](file:///c:/MSC_Malaria/run_bootstrap_ci_final.py)
* **Command**: `python run_bootstrap_ci_final.py`
* **Log File**: [`outputs/logs/step10_bootstrap_ci.log`](file:///c:/MSC_Malaria/outputs/logs/step10_bootstrap_ci.log)
* **Mathematical Operations**:
  - Executes $B = 1{,}000$ non-parametric bootstrap replications on the held-out test cohort ($N = 49$).
  - Derives empirical 2.5th and 97.5th percentiles for Precision (Pos & Neg), F1-Score (Pos & Neg), Macro F1, ROC-AUC, and PR-AUC.
* **Yield**: Directly proves Dissertation Chapter 4 Table 4.8 Panels A and B confidence intervals.

---

## 5. Direct Cross-Reference: Code Execution to Dissertation Chapters

The following matrix directly cross-references every table and figure presented in Chapters 3, 4, and 5 of [`Full_Work.docx`](file:///c:/MSC_Malaria/Full_Work.docx) and [`Full_Work_Revised.docx`](file:///c:/MSC_Malaria/Full_Work_Revised.docx) with its originating script, output execution log, and audit walkthrough document:

| Dissertation Item | Description | Originating Script | Verification / Audit Artifact |
| :--- | :--- | :--- | :--- |
| **Table 3.1** | Sequential Participant Flow & Cohort Attrition Filtering Pipeline | [`pre_process.py`](file:///c:/MSC_Malaria/pre_process.py) | [`outputs/logs/step0_data_ingestion.log`](file:///c:/MSC_Malaria/outputs/logs/step0_data_ingestion.log) |
| **Table 3.2** | Algorithmic Mapping of SDOH Predictor Variables (NMIS 2021) | [`first_stage.py`](file:///c:/MSC_Malaria/first_stage.py) | [`outputs/logs/step1_categorical_mapping.log`](file:///c:/MSC_Malaria/outputs/logs/step1_categorical_mapping.log) |
| **Table 3.3** | Unified Algorithmic Parameters for Transparent Models | [`fourth_step.py`](file:///c:/MSC_Malaria/fourth_step.py) | [`outputs/logs/step4_rule_induction.log`](file:///c:/MSC_Malaria/outputs/logs/step4_rule_induction.log) |
| **Table 3.4** | Algorithmic Evaluation Metrics for Epidemiological Modelling | [`sixth_step.py`](file:///c:/MSC_Malaria/sixth_step.py) | [`outputs/walkthrough_step6.md`](file:///c:/MSC_Malaria/outputs/walkthrough_step6.md) |
| **Figure 3.1** | Geographical Location of Nasarawa State within North-Central Nigeria | Cartographic Ingestion | Chapter 3 Map |
| **Figure 3.2** | Multi-Stage Computational Architecture & Analysis Workflow | Analysis Pipeline | Chapter 3 Flowchart |
| **Table 4.1** | Target Class Diagnostic Profile (Microscopy Smear Test `hml32`) | [`second_step.py`](file:///c:/MSC_Malaria/second_step.py) | [`outputs/logs/step2_baseline_diagnostics.log`](file:///c:/MSC_Malaria/outputs/logs/step2_baseline_diagnostics.log) & [`walkthrough_step2.md`](file:///c:/MSC_Malaria/outputs/walkthrough_step2.md) (Sec 2) |
| **Table 4.2** | Biological and Behavioural Vulnerability Profiles (`b19`, `b4`, `hml20`) | [`second_step.py`](file:///c:/MSC_Malaria/second_step.py) | [`outputs/logs/step2_baseline_diagnostics.log`](file:///c:/MSC_Malaria/outputs/logs/step2_baseline_diagnostics.log) & [`walkthrough_step2.md`](file:///c:/MSC_Malaria/outputs/walkthrough_step2.md) (Sec 3A) |
| **Table 4.3** | Socioeconomic and Maternal Characteristics Profiles (`v106`, `v190`) | [`second_step.py`](file:///c:/MSC_Malaria/second_step.py) | [`outputs/logs/step2_baseline_diagnostics.log`](file:///c:/MSC_Malaria/outputs/logs/step2_baseline_diagnostics.log) & [`walkthrough_step2.md`](file:///c:/MSC_Malaria/outputs/walkthrough_step2.md) (Sec 3B) |
| **Table 4.4** | Structural Built Environment Characteristics Profiles (`v127`, `v128`, `v129`) | [`second_step.py`](file:///c:/MSC_Malaria/second_step.py) | [`outputs/logs/step2_baseline_diagnostics.log`](file:///c:/MSC_Malaria/outputs/logs/step2_baseline_diagnostics.log) & [`walkthrough_step2.md`](file:///c:/MSC_Malaria/outputs/walkthrough_step2.md) (Sec 3C) |
| **Table 4.5** | Drinking Water & Sanitation Infrastructure Profiles (`v113`, `v116`, `v119`, `v158`) | [`second_step.py`](file:///c:/MSC_Malaria/second_step.py) | [`outputs/logs/step2_baseline_diagnostics.log`](file:///c:/MSC_Malaria/outputs/logs/step2_baseline_diagnostics.log) & [`walkthrough_step2.md`](file:///c:/MSC_Malaria/outputs/walkthrough_step2.md) (Sec 3D) |
| **Figure 4.1** | Stratified Vector Profile of Weighted Malaria Prevalence across SDoH Modules | [`second_step.py`](file:///c:/MSC_Malaria/second_step.py) | [`outputs/plots/sdoh_prevalence.png`](file:///c:/MSC_Malaria/outputs/plots/sdoh_prevalence.png) |
| **Table 4.6** | Diagnostic Matrix for Stratified Partitioning & SMOTE-NC Resampling | [`third_step.py`](file:///c:/MSC_Malaria/third_step.py) | [`outputs/logs/step3_partition.log`](file:///c:/MSC_Malaria/outputs/logs/step3_partition.log) & [`walkthrough_step3.md`](file:///c:/MSC_Malaria/outputs/walkthrough_step3.md) |
| **Table 4.7** | Algorithmic Configurations and GridSearchCV Hyperparameter Spaces | [`fifth_step.py`](file:///c:/MSC_Malaria/fifth_step.py) | [`outputs/logs/step5_black_box_baselines.log`](file:///c:/MSC_Malaria/outputs/logs/step5_black_box_baselines.log) & [`walkthrough_step5.md`](file:///c:/MSC_Malaria/outputs/walkthrough_step5.md) |
| **Table 4.8** | Algorithmic Evaluation Matrix (Panels A & B + 95% Bootstrap CIs + Baselines) | [`run_bootstrap_ci_final.py`](file:///c:/MSC_Malaria/run_bootstrap_ci_final.py) | [`outputs/logs/step10_bootstrap_ci.log`](file:///c:/MSC_Malaria/outputs/logs/step10_bootstrap_ci.log) & [`outputs/walkthrough_step10_bootstrap_ci.md`](file:///c:/MSC_Malaria/outputs/walkthrough_step10_bootstrap_ci.md) |
| **In-Text Section** | Evaluation of Small-Sample Estimation Uncertainty (95% Wilson Score CIs) | [`run_bootstrap_ci_final.py`](file:///c:/MSC_Malaria/run_bootstrap_ci_final.py) | Formatted in [`Full_Work.docx`](file:///c:/MSC_Malaria/Full_Work.docx) after Table 4.8 |
| **In-Text Section** | Multi-Seed Stability Profile across 10 Distinct Random Splits (Mean ± SD) | [`step7_multiseed_stability.py`](file:///c:/MSC_Malaria/step7_multiseed_stability.py) | [`outputs/logs/step7_multiseed_stability.log`](file:///c:/MSC_Malaria/outputs/logs/step7_multiseed_stability.log) & [`outputs/multiseed_stability_results.csv`](file:///c:/MSC_Malaria/outputs/multiseed_stability_results.csv) |
| **In-Text Section** | Grouped Household and Spatial Cluster Sensitivity Analysis (5-Fold CV) | [`step8_grouped_cluster_sensitivity.py`](file:///c:/MSC_Malaria/step8_grouped_cluster_sensitivity.py) | [`outputs/logs/step8_grouped_cluster_sensitivity.log`](file:///c:/MSC_Malaria/outputs/logs/step8_grouped_cluster_sensitivity.log) & [`outputs/grouped_cluster_sensitivity_results.csv`](file:///c:/MSC_Malaria/outputs/grouped_cluster_sensitivity_results.csv) |
| **Figure 4.2** | Multi-Metric Performance Comparison Profile Matrix across Evaluated Models | [`sixth_step.py`](file:///c:/MSC_Malaria/sixth_step.py) | [`outputs/plots/model_performance_comparison.png`](file:///c:/MSC_Malaria/outputs/plots/model_performance_comparison.png) |
| **Figure 4.3** | Supervised Learning Confusion Matrix Heatmaps for Isolated Test Split ($N=49$) | [`sixth_step.py`](file:///c:/MSC_Malaria/sixth_step.py) | [`outputs/plots/confusion_matrices.png`](file:///c:/MSC_Malaria/outputs/plots/confusion_matrices.png) |
| **Figure 4.4** | Comparative Receiver Operating Characteristic (ROC) Curves across Evaluated Classifiers | [`sixth_step.py`](file:///c:/MSC_Malaria/sixth_step.py) | [`outputs/plots/roc_comparison.png`](file:///c:/MSC_Malaria/outputs/plots/roc_comparison.png) |
| **Table 4.9** | Model Complexity and Structural Interpretability Comparison | [`sixth_step.py`](file:///c:/MSC_Malaria/sixth_step.py) | [`outputs/logs/step6_comparative_evaluation.log`](file:///c:/MSC_Malaria/outputs/logs/step6_comparative_evaluation.log) & [`walkthrough_step6.md`](file:///c:/MSC_Malaria/outputs/walkthrough_step6.md) |
| **Figure 4.5** | Global Recursive Partitioning Topology of C4.5 Decision Tree | [`fourth_step.py`](file:///c:/MSC_Malaria/fourth_step.py) | [`outputs/plots/decision_tree_plot.png`](file:///c:/MSC_Malaria/outputs/plots/decision_tree_plot.png) |
| **Table 4.10** | Logical Decision Branches Extracted via C4.5 Decision Tree (Support & Coverage) | [`fourth_step.py`](file:///c:/MSC_Malaria/fourth_step.py) & [`step9_rule_coverage_support.py`](file:///c:/MSC_Malaria/step9_rule_coverage_support.py) | [`outputs/c45_pathway_coverage_support.csv`](file:///c:/MSC_Malaria/outputs/c45_pathway_coverage_support.csv) & [`walkthrough_step9_rule_coverage.md`](file:///c:/MSC_Malaria/outputs/walkthrough_step9_rule_coverage.md) |
| **Table 4.11** | Wittgenstein Propositional Ruleset Extracted via RIPPER (Support & Coverage) | [`fourth_step.py`](file:///c:/MSC_Malaria/fourth_step.py) & [`step9_rule_coverage_support.py`](file:///c:/MSC_Malaria/step9_rule_coverage_support.py) | [`outputs/ripper_rule_coverage_support.csv`](file:///c:/MSC_Malaria/outputs/ripper_rule_coverage_support.csv) & [`walkthrough_step9_rule_coverage.md`](file:///c:/MSC_Malaria/outputs/walkthrough_step9_rule_coverage.md) |
| **Figure 4.6** | Optimization Frontier Mapping Model Complexity against Continuous Macro F1-Score | [`sixth_step.py`](file:///c:/MSC_Malaria/sixth_step.py) | [`outputs/plots/interpretability_vs_accuracy.png`](file:///c:/MSC_Malaria/outputs/plots/interpretability_vs_accuracy.png) |
| **Table 5.1** | Synthesis Matrix of Extracted White-Box SDoH Pathways & Literature Alignment | Synthesis Matrix | Chapter 5 Policy Table |

---

## 6. Official Submission & Defense Presentation Package

When transmitting the project deliverables to the supervisor or external examiners:

1. **Self-Contained Code Archive**:
   Provide the root directory containing all standalone `.py` scripts alongside [`requirements.txt`](file:///c:/MSC_Malaria/requirements.txt), [`pyproject.toml`](file:///c:/MSC_Malaria/pyproject.toml), and [`uv.lock`](file:///c:/MSC_Malaria/uv.lock).
2. **Deterministic Guarantee**:
   Direct the examiner to Section 2 of this guide. Running `pip install -r requirements.txt` or `uv sync` followed by `python generate_execution_logs.py` reproduces all outputs and execution logs identically.
3. **Execution Audit Verification**:
   Point to [`outputs/logs/`](file:///c:/MSC_Malaria/outputs/logs/) and [`outputs/walkthrough_step*.md`](file:///c:/MSC_Malaria/outputs/) to demonstrate that every table, count, percentage, confidence interval, and metric reported in the dissertation was directly generated by the codebase.
4. **Dissertation Document Synchronization**:
   Both [`Full_Work.docx`](file:///c:/MSC_Malaria/Full_Work.docx) and [`Full_Work_Revised.docx`](file:///c:/MSC_Malaria/Full_Work_Revised.docx) have been synchronized with the latest high-resolution figures, uncertainty confidence intervals, baseline evaluations, and examiner corrections via [`build_full_work_revised.py`](file:///c:/MSC_Malaria/build_full_work_revised.py).
