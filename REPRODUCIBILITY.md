# Reproducibility Guide

**Thesis:** Transparent Rule-Based Modelling of Childhood Malaria Using Social Determinants of Health in Rural Nasarawa State, Nigeria (MSc Computer Science)
**Data:** 2021 Nigeria Malaria Indicator Survey (NMIS), rural Nasarawa State, children aged 6–59 months (N = 243)
**Revised thesis file:** `Full_Work_revised_3.docx` (not stored in this repository)
**Companion document:** [EXAMINER_TECHNICAL_GUIDE.md](EXAMINER_TECHNICAL_GUIDE.md) explains the methods and results stage by stage.

This guide explains how to regenerate every table and figure in Chapter 4 of the revised thesis from the code in this repository.

---

## 1. What the repository contains

| Item | Location |
| :--- | :--- |
| Pipeline scripts (Stages 0–11) | `pre_process.py`, `first_stage.py`, `second_step.py`, `third_step.py`, `fourth_step.py`, `fifth_step.py`, `sixth_step.py`, `step7_multiseed_stability.py`, `step8_grouped_cluster_sensitivity.py`, `step9_rule_coverage_support.py`, `run_bootstrap_ci_final.py` |
| Standalone diagnostics (run separately, see Section 4) | `class_weight_comparison.py`, `rule_stability.py`, `ripper_trace.py`, `wealth_housing_association.py` |
| Shared C4.5 utilities | `c45_tree_utils.py` (parses the ChefBoost rules file, routes records, computes leaf scores, draws Figure 4.5 and Figure I.1) |
| One-command runner | `generate_execution_logs.py` (runs all stages in order and writes one log per stage) |
| Pinned environment | `requirements.txt`, `pyproject.toml`, `uv.lock` |
| Execution logs | `outputs/logs/step0_data_ingestion.log` … `outputs/logs/step10_bootstrap_ci.log` |
| Result tables (CSV) | `outputs/*.csv` (see Section 5) |
| Fitted models and test predictions | `outputs/models/` |
| Figures | `outputs/plots/` |
| Derived analysis data | `final_malaria_sdoh_matrix.csv`, `pre_processed_malaria_sdoh_matrix.csv`, `X_train_resampled.csv`, `Y_train_resampled.csv`, `X_test.csv`, `Y_test.csv` |

The raw NMIS files are **not** in the repository (see Section 2).

---

## 2. Data access

The pipeline starts from two Stata files of the 2021 NMIS:

| File | Recode | Used for |
| :--- | :--- | :--- |
| `NGKR81DT/NGKR81FL.DTA` | Children's Recode (KR) | Child, maternal and household characteristics |
| `NGPR81DT/NGPR81FL.DTA` | Household Member Recode (PR) | Microscopy result (`hml32`) and bed-net use (`hml20`) |

These files are distributed by The DHS Program (https://dhsprogram.com) to registered users and may not be redistributed, so they are excluded by `.gitignore`. To run Stages 0–2, register with The DHS Program, request the Nigeria 2021 MIS datasets, and place the two folders above in the repository root.

Without the raw files you can still run Stages 3–11, which start from the committed `final_malaria_sdoh_matrix.csv` (Section 4, Option B).

---

## 3. Environment

The results in the thesis were produced with:

* **Python 3.14.5** on **Windows 11** (`requires-python = ">=3.14"` in `pyproject.toml`).
* The exact package versions pinned in `requirements.txt` and `uv.lock`. The main ones are:

| Package | Version | Used for |
| :--- | :---: | :--- |
| pandas | 3.0.3 | Reading the Stata files, data handling |
| numpy | 2.5.1 | Numerical work |
| scikit-learn | 1.9.0 | Splitting, logistic regression, Random Forest, SVM, metrics |
| imbalanced-learn | 0.14.2 | SMOTE-NC |
| chefboost | 0.0.19 | C4.5 decision tree |
| wittgenstein | 0.3.5 | RIPPER rule learner |
| scipy | 1.18.0 | Back end for scikit-learn and imbalanced-learn |
| matplotlib | 3.11.0 | Figures |

Other operating systems were not tested.

**Setup with uv (uses the lockfile):**

```bash
uv sync
```

**Setup with pip:**

```bash
python -m venv .venv
# Windows PowerShell:  .\.venv\Scripts\Activate.ps1
# Linux/macOS:         source .venv/bin/activate
pip install -r requirements.txt
```

---

## 4. Running the pipeline

### Option A: full run (requires the raw NMIS files)

```bash
python generate_execution_logs.py
```

This runs Stages 0–11 in order. For each stage it captures the console output, together with the Python version and exit code, in `outputs/logs/`. It stops at the first stage that fails.

### Option B: without the raw NMIS files (Stages 3–11)

```bash
python third_step.py
python fourth_step.py
python fifth_step.py
python sixth_step.py
python step7_multiseed_stability.py
python step8_grouped_cluster_sensitivity.py
python step9_rule_coverage_support.py
python run_bootstrap_ci_final.py
```

`third_step.py` reads the committed `final_malaria_sdoh_matrix.csv` and recreates the train/test files, including `X_train_original.csv` and `Y_train_original.csv` (the 194 training rows before oversampling, which later stages need). Run the scripts in this order: each stage reads files written by the stages before it.

### Stages and their outputs

| Stage | Script | Log | Main outputs |
| :--- | :--- | :--- | :--- |
| 0 | `pre_process.py` | `step0_data_ingestion.log` | `pre_processed_malaria_sdoh_matrix.csv` |
| 1 | `first_stage.py` | `step1_categorical_mapping.log` | `final_malaria_sdoh_matrix.csv` |
| 2 | `second_step.py` | `step2_baseline_diagnostics.log` | weighted descriptive tables (log), `plots/sdoh_prevalence.png` |
| 3–4 | `third_step.py` | `step3_partition.log` | `X/Y_train_original.csv`, `X/Y_train_resampled.csv`, `X/Y_test.csv` |
| 5 | `fourth_step.py` | `step4_rule_induction.log` | `outputs/rules/rules.py`, `plots/decision_tree_plot.png`, `plots/decision_tree_full.png` |
| 6 | `fifth_step.py` | `step5_black_box_baselines.log` | `models/random_forest_pipeline.pkl`, `models/svm_pipeline.pkl`, `models/best_params.json` |
| 7 | `sixth_step.py` | `step6_comparative_evaluation.log` | `test_point_metrics.csv`, `tradeoff_matrix.csv`, `models/test_predictions.npz`, Figures 4.2–4.4 and 4.6 |
| 8 | `step7_multiseed_stability.py` | `step7_multiseed_stability.log` | `multiseed_stability_results.csv`, `multiseed_stability_per_seed.csv` |
| 9 | `step8_grouped_cluster_sensitivity.py` | `step8_grouped_cluster_sensitivity.log` | `grouped_cluster_sensitivity_results.csv`, `grouped_cluster_sensitivity_per_fold.csv` |
| 10 | `step9_rule_coverage_support.py` | `step9_rule_coverage_support.log` | `c45_pathway_coverage_support.csv`, `c45_leaf_support_all.csv`, `ripper_rule_coverage_support.csv` |
| 11 | `run_bootstrap_ci_final.py` | `step10_bootstrap_ci.log` | `table_4_8_bootstrap_ci.csv` |

Paths in the last column are relative to `outputs/` unless they start at the repository root.

### Standalone diagnostics

Four further scripts produce results reported in the thesis. `generate_execution_logs.py` does not run them, so run them yourself after Stages 3–10 (Option A or Option B):

```bash
python class_weight_comparison.py
python rule_stability.py
python ripper_trace.py
python wealth_housing_association.py
```

| Script | Log | Main outputs | Needs |
| :--- | :--- | :--- | :--- |
| `class_weight_comparison.py` | `class_weight_comparison.log` | `class_weight_comparison.csv`, `class_weight_multiseed.csv` | `X/Y_train_original.csv`, `X/Y_test.csv`, `final_malaria_sdoh_matrix.csv`, `test_point_metrics.csv`, `multiseed_stability_results.csv` |
| `rule_stability.py` | `rule_stability.log` | `rule_stability_summary.csv`, `rule_stability_per_seed.csv`, `rule_precision_wilson.csv` | `final_malaria_sdoh_matrix.csv`, `c45_pathway_coverage_support.csv`, `ripper_rule_coverage_support.csv` |
| `ripper_trace.py` | `ripper_trace.log` | The log only | `X/Y_train_resampled.csv` |
| `wealth_housing_association.py` | `wealth_housing_association.log` | `wealth_housing_association.csv` | `final_malaria_sdoh_matrix.csv` |

Each script writes its own log to `outputs/logs/`. `X/Y_train_original.csv` are not stored in the repository; `third_step.py` creates them.

---

## 5. Where each thesis result comes from

| Thesis item | Content | Script | Source file |
| :--- | :--- | :--- | :--- |
| Table 3.1 | Participant flow, 10,988 → 243 children | `pre_process.py` | `step0_data_ingestion.log` |
| Tables 4.1–4.5 | Weighted descriptive profiles | `second_step.py` | `step2_baseline_diagnostics.log` |
| Figure 4.1 | Weighted prevalence by wealth, education, net use, floor | `second_step.py` | `plots/sdoh_prevalence.png` |
| Table 4.6 | Train/test split and SMOTE-NC balance | `third_step.py` | `step3_partition.log` |
| Table 4.7 | Hyperparameter search spaces and selected values | `fifth_step.py` | `models/best_params.json`, `step5_black_box_baselines.log` |
| Section 4.2.1 | Association of wealth quintile with housing, water and sanitation variables (bias-corrected Cramér's V) | `wealth_housing_association.py` | `wealth_housing_association.csv` |
| Table 4.8 | Test-set metrics with Wilson and bootstrap 95% intervals | `sixth_step.py`, `run_bootstrap_ci_final.py` | `test_point_metrics.csv`, `table_4_8_bootstrap_ci.csv` |
| Table 4.8.1 | 10-seed stability (mean, SD, range) | `step7_multiseed_stability.py` | `multiseed_stability_results.csv` |
| Table 4.8.2 | Stratified, household-grouped and cluster-grouped 5-fold CV | `step8_grouped_cluster_sensitivity.py` | `grouped_cluster_sensitivity_results.csv` |
| Table 4.8.3 | SMOTE-NC versus class weighting (logistic regression, Random Forest, SVM) | `class_weight_comparison.py` | `class_weight_comparison.csv`, `class_weight_multiseed.csv` |
| Table 4.9 | Model complexity | `sixth_step.py` | `tradeoff_matrix.csv` |
| Table 4.10 | C4.5 named pathways C1–C5: support, coverage, precision, recall | `step9_rule_coverage_support.py` | `c45_pathway_coverage_support.csv` |
| Table 4.11 | RIPPER rules: support, coverage, precision, recall | `step9_rule_coverage_support.py` | `ripper_rule_coverage_support.csv` |
| Tables 4.10 and 4.11, Prec. column | Wilson 95% intervals for rule precision (Real and Test rows) | `rule_stability.py` | `rule_precision_wilson.csv` |
| Table 4.11.1 and Section 4.2.4.2 | Recurrence of the reported rules across ten partitions | `rule_stability.py` | `rule_stability_summary.csv`, `rule_stability_per_seed.csv`, `logs/rule_stability.log` |
| Algorithm 3.2 and Section 4.2.4.3 | RIPPER induction phases (15 rules grown, 13 deleted by the final description-length step) | `ripper_trace.py` | `logs/ripper_trace.log` |
| Figure 4.2 | Performance bar chart | `sixth_step.py` | `plots/model_performance_comparison.png` |
| Figure 4.3 | Confusion matrices | `sixth_step.py` | `plots/confusion_matrices.png` |
| Figure 4.4 | ROC curves | `sixth_step.py` | `plots/roc_comparison.png` |
| Figure 4.5 | C4.5 tree, named pathways | `fourth_step.py` | `plots/decision_tree_plot.png` |
| Figure 4.6 | Complexity versus macro F1 | `sixth_step.py` | `plots/interpretability_vs_accuracy.png` |
| Figure I.1 | Complete C4.5 tree | `fourth_step.py` | `plots/decision_tree_full.png` |

Appendices A–J of the thesis reproduce the scripts. Appendix A covers Stages 0–2, Appendices B–H cover `third_step.py` to `step9_rule_coverage_support.py`, Appendix I is `c45_tree_utils.py`, and Appendix J is `run_bootstrap_ci_final.py`. The four standalone diagnostics are not reproduced in the appendices; the thesis cites them by name.

---

## 6. Determinism

All random steps use fixed seeds: the train/test split, SMOTE-NC, model fitting, cross-validation shuffling and the bootstrap (seed 42 throughout; the multi-seed analysis and the ten-partition parts of the standalone diagnostics use seeds 10, 20, 30, 42, 50, 60, 70, 80, 90 and 100). Repeated full runs on the environment above reproduced byte-identical cohort and split files and an identical C4.5 rules file. Each log also records the run's duration and the Python path, which naturally change between runs and machines. Results on other platforms or package versions have not been checked.

---

## 7. Files that are not part of the pipeline

These files are not run by `generate_execution_logs.py`. Treat `outputs/*.csv` and `outputs/logs/` as the authoritative results.

| File | Status |
| :--- | :--- |
| `app.py` | Utility for parsing DHS `.MAP` layout files; not used by the pipeline |
| `scratch_test_c45_complexity.py`, `scratch_test_ripper.py` | Development scratch tests |
| `outputs/decision_tree_plot.png` | Copy of `outputs/plots/decision_tree_plot.png` |
| `docx_content.txt` | Earlier drafting material |

Notes and scripts from earlier drafts (step walkthroughs, a C4.5/CART discrepancy note, an earlier bootstrap script and a thesis-patching script) were removed because their figures predated the current results. They remain in the git history.
