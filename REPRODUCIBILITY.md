# Master Reproducibility Guide & Computational Audit Trail

**Project Title**: Machine Learning and Rule Induction for Pediatric Malaria Screening Using Social Determinants of Health  
**Target Study Cohort**: Rural Nasarawa State, Nigeria (2021 Nigeria Malaria Indicator Survey - NMIS)  
**Primary Research Goal**: Comparative evaluation of white-box rule-based models (C4.5 Decision Tree, RIPPER) versus black-box benchmarks (Random Forest, Support Vector Machine) using survey-weighted demographic and environmental exposures.

---

## 1. Executive Summary & Supervisor Verification Checklist

This document serves as the formal academic reproducibility audit trail, explicitly addressing the supervisor and reviewer requirement:

> *"The author should provide a requirements file or environment lockfile, exact package versions, the full scripts outside Word formatting, and execution logs showing that the reported tables were generated from the supplied code."*

### Compliance Matrix

| Requirement | Implementation Artifact | Location |
| :--- | :--- | :--- |
| **Requirements File** | Standard pinned pip manifest | [`requirements.txt`](file:///c:/MSC_Malaria/requirements.txt) |
| **Environment Lockfile** | Cryptographic SHA-256 deterministic lockfile | [`uv.lock`](file:///c:/MSC_Malaria/uv.lock) |
| **Project Definition** | Declarative project metadata & dependencies | [`pyproject.toml`](file:///c:/MSC_Malaria/pyproject.toml) |
| **Full Scripts (Outside Word)** | Standalone, pure `.py` source scripts (Steps 0–6) | Project root (`pre_process.py` through `sixth_step.py`) |
| **Execution Logs & Table Proofs** | Terminal execution logs & mathematical audit trails | [`outputs/logs/`](file:///c:/MSC_Malaria/outputs/logs/) and [`outputs/walkthrough_step*.md`](file:///c:/MSC_Malaria/outputs/) |
| **Model & Graphic Deliverables** | Serialized ML models (`.pkl`) & 300 DPI publication plots | [`outputs/models/`](file:///c:/MSC_Malaria/outputs/models/) and [`outputs/plots/`](file:///c:/MSC_Malaria/outputs/plots/) |

---

## 2. Computational Environment & Exact Package Versions

### A. Core Runtime Specification
* **Python Runtime**: Python 3.14+ (Verified on CPython 3.14.5, x86_64)
* **Operating System**: Platform independent (Tested on Windows 11 / PowerShell; fully POSIX compliant for Linux / macOS)

### B. Pinned Dependency Roster
All package versions are strictly pinned in [`requirements.txt`](file:///c:/MSC_Malaria/requirements.txt):

| Package Name | Exact Version | Role in Research Pipeline |
| :--- | :---: | :--- |
| `pandas` | **3.0.3** | Stata `.DTA` ingestion, survey weight normalization, data aggregation |
| `scikit-learn` | **1.9.0** | Stratified train/test splitting, Random Forest, SVM, evaluation metrics |
| `scipy` | **1.18.0** | Sparse matrix representation, mathematical backend computations |
| `numpy` | **2.5.1** | Vectorized array transformations and continuous threshold arithmetic |
| `imbalanced-learn` (`imblearn`) | **0.14.2** | SMOTE-NC (Synthetic Minority Over-sampling for Nominal and Continuous) |
| `chefboost` | **0.0.19** | Information gain ratio and C4.5 decision tree rule construction |
| `wittgenstein` | **0.3.5** | Repeated Incremental Pruning to Produce Error Reduction (RIPPER) |
| `matplotlib` | **3.11.0** | Publication-grade graphics generation (ROC curves, confusion matrices) |
| `joblib` | **1.5.3** | Serialization of fitted scikit-learn pipeline objects |

### C. Environment Setup & Replication Instructions

#### Option 1: Fast Setup via `uv` (Recommended)
```powershell
# Installs exact locked dependencies from uv.lock deterministically
uv sync

# Run any script in the managed environment:
uv run python second_step.py
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
```

---

## 3. Directory Layout & Pipeline Artifacts

```
c:\MSC_Malaria\
├── NGKR81DT\                             # Raw DHS Stata Child Recode folder
│   └── NGKR81FL.DTA                      # Raw Stata binary data (Children 0-59 months)
├── NGPR81DT\                             # Raw DHS Stata Household Member Recode folder
│   └── NGPR81FL.DTA                      # Raw Stata binary data (Household members)
│
├── pre_process.py                        # Step 0: Ingestion, subsetting & relational merge
├── first_stage.py                        # Step 1: Categorical stabilization & SMOTE-NC tracking
├── second_step.py                        # Step 2: Complex survey weight normalization & diagnostics
├── third_step.py                         # Step 3 & 4: Stratified 80/20 train/test split & SMOTE-NC
├── fourth_step.py                        # Step 4: White-Box Induction (C4.5 & RIPPER)
├── fifth_step.py                         # Step 5: Black-Box Benchmark Models (RF & SVM)
├── sixth_step.py                         # Step 6: Test evaluation, ROC-AUC & Table 4.13 synthesis
│
├── requirements.txt                      # Pinned package requirements manifest
├── pyproject.toml                        # Project configuration & dependency declarations
├── uv.lock                               # Exact cryptographic lockfile
├── REPRODUCIBILITY.md                    # This master audit and reproduction document
│
├── pre_processed_malaria_sdoh_matrix.csv # Intermediate merged feature matrix (N=243, 17 cols)
├── final_malaria_sdoh_matrix.csv         # Stabilized ML matrix with integer/string typing
├── X_train_resampled.csv                 # SMOTE-NC balanced training predictors (N=316, 12 cols)
├── Y_train_resampled.csv                 # SMOTE-NC balanced training target (50/50 balance)
├── X_test.csv                            # Isolated untouched clinical test predictors (N=49)
├── Y_test.csv                            # Isolated untouched clinical test ground-truth target
│
└── outputs\
    ├── logs\                             # Plaintext console execution logs
    │   └── step2_baseline_diagnostics.log# Complete Step 2 stdout output proving dissertation tables
    ├── models\                           # Trained and serialized machine learning pipelines
    │   ├── random_forest_pipeline.pkl    # Serialized tuned Random Forest model
    │   └── svm_pipeline.pkl              # Serialized tuned Support Vector Machine model
    ├── plots\                            # Publication-grade figures (300 DPI)
    │   ├── decision_tree_plot.png        # C4.5 Decision Tree visualization
    │   ├── roc_comparison.png            # Joint 4-model ROC/AUC curve
    │   ├── confusion_matrices.png        # Comparative 2x2 confusion matrices
    │   ├── model_performance_comparison.png # Bar chart of accuracy, sensitivity, specificity
    │   ├── sdoh_prevalence.png           # Bivariate malaria prevalence by SDOH factors
    │   └── interpretability_vs_accuracy.png # Pareto trade-off curve
    ├── c45_visualization_discrepancy_note.md # Methodological note explaining C4.5 vs CART split
    ├── walkthrough_step2.md              # Academic audit trail for Baseline Survey Diagnostics
    ├── walkthrough_step3.md              # Academic audit trail for Partitioning & SMOTE-NC
    ├── walkthrough_step4.md              # Academic audit trail for C4.5 & RIPPER Rule Induction
    ├── walkthrough_step5.md              # Academic audit trail for Black-Box Baselines
    └── walkthrough_step6.md              # Academic audit trail for Final Evaluation & Table 4.13
```

---

## 4. Sequential Execution Guide (Outside Word Formatting)

Every stage of the analysis pipeline is encapsulated in a dedicated, pure Python script. All scripts can be executed sequentially in the console:

### Stage 0: Data Ingestion & Geopolitical Filtering
* **Script**: [`pre_process.py`](file:///c:/MSC_Malaria/pre_process.py)
* **Command**: `python pre_process.py`
* **Mathematical Operations**:
  - Ingests raw Stata binaries (`NGKR81FL.DTA` and `NGPR81FL.DTA`).
  - Biological survival filter: Alive children only (`b5 == 1`).
  - Household residency filter: De facto household residents (`b16 > 0`).
  - Biological age window: 6 to 59 months (`b19 >= 6` & `b19 <= 59`).
  - Geopolitical spatial filter: Nasarawa State (`v024 == 15`) and Rural communities (`v025 == 2`).
  - Relational Left-Join on composite keys (`hv001`, `hv002`, `hvidx`).
  - Dropped missing microscopy results (`hml32`).
* **Yield**: Cleaned analytical cohort of **$N=243$** children saved to `pre_processed_malaria_sdoh_matrix.csv`.

---

### Stage 1: Categorical Mapping & Value Stabilization
* **Script**: [`first_stage.py`](file:///c:/MSC_Malaria/first_stage.py)
* **Command**: `python first_stage.py`
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
* **Mathematical Operations**:
  - Analytical weight calculation: $w_i = \frac{\text{v005}_i}{1,000,000}$ (Sum of analytical weights = $185.0152$).
  - Target class imbalance calculation on microscopy smear test (`hml32`).
  - Weighted univariate percentage: $\text{Pct}_c = \left( \frac{\sum w_i \cdot I(x_i = c)}{\sum w_i} \right) \times 100$.
  - Weighted bivariate malaria prevalence: $P_c = \left( \frac{\sum w_i \cdot I(\text{hml32}_i = 1)}{\sum w_i} \right) \times 100$.
* **Execution Log**: Saved at [`outputs/logs/step2_baseline_diagnostics.log`](file:///c:/MSC_Malaria/outputs/logs/step2_baseline_diagnostics.log).

---

### Stage 3 & 4: Stratified Train-Test Splitting & SMOTE-NC Resampling
* **Script**: [`third_step.py`](file:///c:/MSC_Malaria/third_step.py)
* **Command**: `python third_step.py`
* **Mathematical Operations**:
  - Stratified 80/20 train-test partition (`random_state=42`, `stratify=y`), isolating $N_{\text{train}}=194$ (80%) and $N_{\text{test}}=49$ (20%).
  - Drops administrative/tracking metadata (`hv001`, `hv002`, `hvidx`, `v005`, `analytical_weight`) from $X$ to prevent target/administrative leakage.
  - Synthetic Minority Over-sampling for Nominal and Continuous (SMOTE-NC) applied **strictly to the training split**.
  - Nominal features assigned local structural mode among $k=5$ nearest neighbors (no fractional code corruptions).
* **Yield**: Balanced training set of **$N=316$** (158 Positive / 158 Negative) saved to `X_train_resampled.csv` and `Y_train_resampled.csv`; untouched clinical test split saved to `X_test.csv` and `Y_test.csv`.

---

### Stage 5: White-Box Rule Induction (C4.5 & RIPPER)
* **Script**: [`fourth_step.py`](file:///c:/MSC_Malaria/fourth_step.py)
* **Command**: `python fourth_step.py`
* **Mathematical Operations**:
  - Induces C4.5 Decision Tree optimizing Information Gain Ratio:
    $$H(S) = -\sum p_i \log_2 p_i, \quad \text{GainRatio}(A) = \frac{IG(S, A)}{\text{SplitInfo}(S, A)}$$
  - Bottom-up post-pruning to extract human-readable IF-THEN conditional rules.
  - Induces RIPPER classifier using Wittgenstein framework maximizing FOIL Information Gain:
    $$\text{FOIL}_{IG} = p_1 \left( \log_2 \frac{p_1}{p_1 + n_1} - \log_2 \frac{p_0}{p_0 + n_0} \right)$$
* **Yield**: Human-readable rule outputs, tree topology rules, and 300 DPI plot `outputs/plots/decision_tree_plot.png`.

---

### Stage 6: Black-Box Benchmark Baselines (Random Forest & SVM)
* **Script**: [`fifth_step.py`](file:///c:/MSC_Malaria/fifth_step.py)
* **Command**: `python fifth_step.py`
* **Mathematical Operations**:
  - Constructs preprocessor pipelines (`OneHotEncoder(handle_unknown='ignore')` + `StandardScaler`).
  - Trains ensemble `RandomForestClassifier` with 5-fold cross-validated grid search over `n_estimators`, `max_depth`, `min_samples_split`.
  - Trains kernel-based `SVC` with grid search over kernel types (`linear`, `rbf`) and regularization parameters ($C$, $\gamma$).
* **Yield**: Serialized production pipelines saved to `outputs/models/random_forest_pipeline.pkl` and `outputs/models/svm_pipeline.pkl`.

---

### Stage 7: Multi-Criteria Model Evaluation & Thesis Synthesis
* **Script**: [`sixth_step.py`](file:///c:/MSC_Malaria/sixth_step.py)
* **Command**: `python sixth_step.py`
* **Mathematical Operations**:
  - Passes untouched test partition ($N=49$, 40 Negative / 9 Positive) through all four finalized models.
  - Computes exact Confusion Matrices, Global Accuracy, Sensitivity/Recall (Malaria+), Specificity (Malaria-), and Macro F1-Score.
  - Generates joint Receiver Operating Characteristic (ROC) coordinate projections and Area Under Curve (AUC-ROC).
  - Profiles structural complexity (terminal rules, literals per rule) for accuracy-vs-interpretability trade-off.
* **Yield**: Generates thesis Table 4.13, Pareto plots, and publication-ready graphic figures in `outputs/plots/`.

---

## 5. Direct Cross-Reference: Code Execution to Dissertation Chapter 4

The following matrix directly cross-references every table and figure presented in Chapter 4 of the dissertation with its originating script, output log, and audit document:

| Dissertation Item | Description | Originating Script | Verification / Audit Artifact |
| :--- | :--- | :--- | :--- |
| **Table 4.1** | Target Class Distribution (Raw vs Weighted Imbalance) | [`second_step.py`](file:///c:/MSC_Malaria/second_step.py) | [`outputs/logs/step2_baseline_diagnostics.log`](file:///c:/MSC_Malaria/outputs/logs/step2_baseline_diagnostics.log) & [`walkthrough_step2.md`](file:///c:/MSC_Malaria/outputs/walkthrough_step2.md) (Sec 2) |
| **Table 4.2** | Child Demographic Profile (`b19` Age & `b4` Sex) | [`second_step.py`](file:///c:/MSC_Malaria/second_step.py) | [`outputs/logs/step2_baseline_diagnostics.log`](file:///c:/MSC_Malaria/outputs/logs/step2_baseline_diagnostics.log) & [`walkthrough_step2.md`](file:///c:/MSC_Malaria/outputs/walkthrough_step2.md) (Sec 3A) |
| **Table 4.3** | Maternal & Socioeconomic Profile (`v106`, `v190`) | [`second_step.py`](file:///c:/MSC_Malaria/second_step.py) | [`outputs/logs/step2_baseline_diagnostics.log`](file:///c:/MSC_Malaria/outputs/logs/step2_baseline_diagnostics.log) & [`walkthrough_step2.md`](file:///c:/MSC_Malaria/outputs/walkthrough_step2.md) (Sec 3B) |
| **Table 4.4** | Household Infrastructure Profile (`v119` Electricity, `v158` Radio) | [`second_step.py`](file:///c:/MSC_Malaria/second_step.py) | [`outputs/logs/step2_baseline_diagnostics.log`](file:///c:/MSC_Malaria/outputs/logs/step2_baseline_diagnostics.log) & [`walkthrough_step2.md`](file:///c:/MSC_Malaria/outputs/walkthrough_step2.md) (Sec 3C) |
| **Table 4.5** | Housing Construction Material Profile (`v127`, `v128`, `v129`) | [`second_step.py`](file:///c:/MSC_Malaria/second_step.py) | [`outputs/logs/step2_baseline_diagnostics.log`](file:///c:/MSC_Malaria/outputs/logs/step2_baseline_diagnostics.log) & [`walkthrough_step2.md`](file:///c:/MSC_Malaria/outputs/walkthrough_step2.md) (Sec 3D) |
| **Table 4.6** | WASH & Vector Intervention Profile (`v113`, `v116`, `hml20`) | [`second_step.py`](file:///c:/MSC_Malaria/second_step.py) | [`outputs/logs/step2_baseline_diagnostics.log`](file:///c:/MSC_Malaria/outputs/logs/step2_baseline_diagnostics.log) & [`walkthrough_step2.md`](file:///c:/MSC_Malaria/outputs/walkthrough_step2.md) (Sec 3E) |
| **Table 4.9** | Train-Test Partition & SMOTE-NC Resampling Matrix | [`third_step.py`](file:///c:/MSC_Malaria/third_step.py) | [`walkthrough_step3.md`](file:///c:/MSC_Malaria/outputs/walkthrough_step3.md) (Sec 2) |
| **Table 4.10** | C4.5 Decision Tree Extracted Rule Paths | [`fourth_step.py`](file:///c:/MSC_Malaria/fourth_step.py) | [`walkthrough_step4.md`](file:///c:/MSC_Malaria/outputs/walkthrough_step4.md) & [`c45_visualization_discrepancy_note.md`](file:///c:/MSC_Malaria/outputs/c45_visualization_discrepancy_note.md) |
| **Table 4.11** | RIPPER Sequential Covering Induced Rule Statements | [`fourth_step.py`](file:///c:/MSC_Malaria/fourth_step.py) | [`walkthrough_step4.md`](file:///c:/MSC_Malaria/outputs/walkthrough_step4.md) (Sec 3) |
| **Table 4.12** | Black-Box Benchmark Optimal Hyperparameters | [`fifth_step.py`](file:///c:/MSC_Malaria/fifth_step.py) | [`walkthrough_step5.md`](file:///c:/MSC_Malaria/outputs/walkthrough_step5.md) (Sec 2) |
| **Table 4.13** | Comparative Performance & Complexity Evaluation Matrix | [`sixth_step.py`](file:///c:/MSC_Malaria/sixth_step.py) | [`walkthrough_step6.md`](file:///c:/MSC_Malaria/outputs/walkthrough_step6.md) (Sec 2 & 4) |
| **Figure 4.1** | Decision Tree Diagram (Root Split: Floor Material `v127`) | [`fourth_step.py`](file:///c:/MSC_Malaria/fourth_step.py) | [`outputs/plots/decision_tree_plot.png`](file:///c:/MSC_Malaria/outputs/plots/decision_tree_plot.png) |
| **Figure 4.2** | Joint ROC Curve Comparison (4 Models) | [`sixth_step.py`](file:///c:/MSC_Malaria/sixth_step.py) | [`outputs/plots/roc_comparison.png`](file:///c:/MSC_Malaria/outputs/plots/roc_comparison.png) |
| **Figure 4.3** | 2x2 Test Partition Confusion Matrices | [`sixth_step.py`](file:///c:/MSC_Malaria/sixth_step.py) | [`outputs/plots/confusion_matrices.png`](file:///c:/MSC_Malaria/outputs/plots/confusion_matrices.png) |
| **Figure 4.4** | Bivariate Malaria Prevalence Across SDOH Categories | [`second_step.py`](file:///c:/MSC_Malaria/second_step.py) | [`outputs/plots/sdoh_prevalence.png`](file:///c:/MSC_Malaria/outputs/plots/sdoh_prevalence.png) |

---

## 6. Official Submission & Defense Presentation Package

When transmitting the project deliverables to the supervisor or external examiners:

1. **Self-Contained Code Archive**:
   Provide the root directory containing all `.py` scripts alongside [`requirements.txt`](file:///c:/MSC_Malaria/requirements.txt), [`pyproject.toml`](file:///c:/MSC_Malaria/pyproject.toml), and [`uv.lock`](file:///c:/MSC_Malaria/uv.lock).
2. **Deterministic Guarantee**:
   Direct the supervisor to Section 2 of this guide. Running `pip install -r requirements.txt` or `uv sync` reproduces the exact virtual environment without missing or conflicting dependencies.
3. **Execution Audit Verification**:
   Point to [`outputs/logs/`](file:///c:/MSC_Malaria/outputs/logs/) and [`outputs/walkthrough_step*.md`](file:///c:/MSC_Malaria/outputs/) to demonstrate that every table, count, percentage, and metric reported in the thesis was directly generated by the code base.
