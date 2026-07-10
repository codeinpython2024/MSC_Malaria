# Implementation Plan: Comprehensive Graphical Representations of Tabular Data

To provide a richer visual report rather than relying solely on text tables, we will implement graphical representations of the primary datasets and evaluation results. All plots will be saved in high resolution (300 DPI) under `outputs/plots/`.

## Proposed Changes

### 1. [second_step.py](file:///c:/MSc_malaria/second_step.py)
We will add code at the end of `second_step.py` to generate and export:
- **`outputs/plots/sdoh_prevalence.png`**: A 2x2 grid of bar charts showing weighted malaria prevalence across four key Social Determinants of Health (SDOH):
  1. Wealth Quintile (`v190`)
  2. Maternal Education Level (`v106`)
  3. Child Slept Under LLIN (`hml20`)
  4. Main Floor Material (`v127`)

### 2. [sixth_step.py](file:///c:/MSc_malaria/sixth_step.py)
We will add code in `sixth_step.py` to generate and export the following plots:
- **`outputs/plots/model_performance_comparison.png`**: A grouped bar chart showing Accuracy, Macro F1, Sensitivity, and Specificity side-by-side for all four models (C4.5, RIPPER, Random Forest, SVM).
- **`outputs/plots/confusion_matrices.png`**: A 2x2 subplot containing labeled confusion matrix heatmaps (visual grids) with counts for all four models.
- **`outputs/plots/interpretability_vs_accuracy.png`**: A scatter plot representing the trade-off between model complexity (Rules Count / Mean Literals) and predictive performance (Macro F1-score).

---

## Verification Plan

### Automated Verification
- Run the steps sequentially:
  ```powershell
  .venv\Scripts\python.exe second_step.py
  .venv\Scripts\python.exe fourth_step.py
  .venv\Scripts\python.exe sixth_step.py
  ```
- Verify that the new plots exist under `outputs/plots/`:
  - `sdoh_prevalence.png`
  - `decision_tree_plot.png`
  - `roc_comparison.png`
  - `model_performance_comparison.png`
  - `confusion_matrices.png`
  - `interpretability_vs_accuracy.png`
