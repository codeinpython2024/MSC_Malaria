# Walkthrough: Step 7 Black-Box Benchmarking Baselines

This document serves as the comprehensive walkthrough and academic audit trail for **Step 7: Construct Black-Box Benchmarking Baselines (Random Forest & Support Vector Machine)** as implemented in [fifth_step.py](file:///c:/MSc_malaria/fifth_step.py).

---

## 1. Theoretical & Algorithmic Foundations

To validate the clinical necessity of our intrinsically interpretable white-box models within a rigorous machine learning framework, we must establish optimized, high-dimensional black-box empirical baselines. 

We selected two distinct architectures: **Random Forest (an ensemble bagging algorithm)** and **Support Vector Machine (a kernelized margin classifier)**.

### A. Preprocessing Pipeline & Mathematical Scaling
Scikit-Learn estimators require purely numerical representations. Direct consumption of nominal integer category codes introduces incorrect ordering assumptions. To handle a mixed feature space correctly, we implement a Scikit-Learn `ColumnTransformer` embedded in a unified `Pipeline`:

1. **Continuous Feature (`b19` - Child's Age)**:
   Scaled using standard normal scaling:
   $$z = \frac{x - \mu}{\sigma}$$
   This ensures that the continuous feature (scaled $0$ to $59$ months) does not mathematically dominate the binary one-hot encoded vectors (scaled $0$ to $1$) in SVM kernel distance calculations.
   
2. **Nominal Features (11 Variables)**:
   Transformed using **One-Hot Encoding** with `handle_unknown='ignore'`. This maps each categorical code to a distinct binary dimension (e.g., $v158 \in \{0, 1, 2, 3\}$ becomes four independent sparse dimensions), preserving structural distance.

---

### B. Baseline 1: Random Forest Ensemble Classifier
Random Forest is a bootstrap-aggregated (bagged) ensemble of decision trees. It introduces random feature subsets at each split node to decorrelate individual trees and reduce generalization variance:
- **Bagging**: Each tree is trained on a distinct bootstrap sample drawn with replacement from $\tilde{X}_{\text{train}}$.
- **Random Subspace Projection**: For a split on a node, only a random subset of features (typically $\sqrt{d}$) is evaluated.
- **Hyperparameter Grid**:
  - `n_estimators`: $[50, 100, 200, 300]$ (scale of the ensemble)
  - `max_depth`: $[\text{None}, 5, 10, 15]$ (structural depth limit)
  - `min_samples_split`: $[2, 5, 10]$ (minimum samples required to split a node)

---

### C. Baseline 2: Support Vector Machine (SVC)
Support Vector Classifiers (SVC) search for the optimal separating hyperplane that maximizes the margin between two target classes. For non-linear relationships, SVC projects the feature vectors into an infinite-dimensional Hilbert space using the radial basis function (RBF) kernel:

$$K(x_i, x_j) = \exp(-\gamma ||x_i - x_j||^2)$$

- **Hyperparameters Optimized**:
  - $C$: $[0.1, 1, 10, 100]$ (Regularization parameter that controls the trade-off between maximizing the margin and minimizing classification errors on training instances).
  - $\gamma$: $[\text{'scale'}, \text{'auto'}, 0.01, 0.1, 1]$ (Kernel width factor).
  - Kernel: $[\text{'rbf'}, \text{'linear'}, \text{'poly'}]$ (Kernel topology type).
- **Clinical Configuration**: We set `probability=True` during SVC instantiation. This guarantees the model computes continuous posterior probabilities via Platt scaling, which is strictly required to plot downstream Receiver Operating Characteristic (ROC) curves in Step 8.

---

## 2. Deliverable 1: Cross-Validated Hyperparameter Selection

We executed a **5-fold Stratified GridSearchCV** optimizing for the **Macro F1-Score** to combat the clinical target imbalance:

### A. Random Forest Best Parameters
- **Optimal Hyperparameters**: `{'classifier__max_depth': 10, 'classifier__min_samples_split': 5, 'classifier__n_estimators': 300}`
- **Best Cross-Validation Score (Macro F1)**: **0.8119**

### B. Support Vector Machine Best Parameters
- **Optimal Hyperparameters**: `{'classifier__C': 100, 'classifier__gamma': 1, 'classifier__kernel': 'rbf'}`
- **Best Cross-Validation Score (Macro F1)**: **0.8139**

---

## 3. Deliverable 2: Performance Evaluation on Unseen Test Partition

Both optimized pipelines were evaluated against the isolated, un-resampled clinical test split ($N=49$, Malaria+ prevalence of $18.37\%$):

### A. Random Forest Test Set Evaluation
* **Confusion Matrix**:
  $$\begin{bmatrix} 34 & 6 \\ 7 & 2 \end{bmatrix}$$
* **Classification Report**:
  - Malaria Negative (0): Precision: **83%**, Recall: **85%**, F1: **84%**
  - Malaria Positive (1): Precision: **25%**, Recall: **22%**, F1: **24%**
  - **Global Accuracy**: **73%**
  - **Macro Average F1**: **54%**

### B. Support Vector Machine Test Set Evaluation
* **Confusion Matrix**:
  $$\begin{bmatrix} 34 & 6 \\ 6 & 3 \end{bmatrix}$$
* **Classification Report**:
  - Malaria Negative (0): Precision: **85%**, Recall: **85%**, F1: **85%**
  - Malaria Positive (1): Precision: **33%**, Recall: **33%**, F1: **33%**
  - **Global Accuracy**: **76%**
  - **Macro Average F1**: **59%**

---

## 4. Verification, Audit Trail & Serialization

### Persistent Outputs
The grid-search routine successfully serialized the complete `Pipeline` objects (inclusive of the numerical scale transformers, one-hot category encoders, and optimal hyperparameter weights) under the `outputs/models/` directory:
1. `outputs/models/random_forest_pipeline.pkl`
2. `outputs/models/svm_pipeline.pkl`

These files can be dynamically unpickled downstream for programmatic prediction and validation.

### How to Run and Audit
Run the baseline optimization script from the workspace root:
```powershell
.venv\Scripts\python.exe -X utf8 fifth_step.py
```
* **Dependency Safe-Guards**: Embedded preprocessors prevent data leakage by only fitting on the training partition and applying transformations (scaling/encoding) to the testing set in validation.
