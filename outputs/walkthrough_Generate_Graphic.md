# Walkthrough: Comprehensive Graphical Outputs

We have implemented graphical representations for all baseline socio-demographic statistics and evaluation matrices.

## New Generated Graphics

All plots have been saved as high-resolution (300 DPI) images under `outputs/plots/`:

1. **SDOH Prevalence Plot (`sdoh_prevalence.png`)**:
   - A 2x2 grid of bar charts displaying weighted malaria microscopy prevalence percentages across key social determinants:
     - Wealth Quintile (`v190`)
     - Maternal Education Level (`v106`)
     - Net usage (`hml20`)
     - Main Floor Material (`v127`)
   - Shows weighted percentages with text annotations on top of each bar.

2. **Model Performance Comparison (`model_performance_comparison.png`)**:
   - A grouped bar chart displaying Accuracy, Precision, Sensitivity, Specificity, and Macro F1-score side-by-side for each model (C4.5, RIPPER, RF, SVM).

3. **Confusion Matrix Heatmaps (`confusion_matrices.png`)**:
   - A 2x2 grid containing color-mapped grids with count labels representing True Positives, False Positives, False Negatives, and True Negatives for each classifier.

4. **Accuracy-vs-Interpretability Trade-off Plot (`interpretability_vs_accuracy.png`)**:
   - A 2D scatter plot charting model complexity (derived from rules/trees count size proxies) against Macro F1 performance, visually illustrating the trade-off frontier.

5. **Decision Tree Topology (`decision_tree_plot.png`)**:
   - An detailed layout of the C4.5-like decision tree rules for clinical reference.
