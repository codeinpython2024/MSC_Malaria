# Technical Note: Algorithmic Discrepancy between C4.5 (ChefBoost) and CART (Scikit-Learn Replica) in Decision Tree Visualization

This note documents the technical and mathematical reasons for the split-structure discrepancy observed during visual verification of the Decision Tree plot, and explains why **`b19` (Child's Age)** is the true mathematical root node of the inducted C4.5 model, whereas **`v116` (Toilet Facility Type)** was chosen by the scikit-learn CART replica.

---

## 1. Background
In the thesis analysis, the primary white-box decision tree model was trained using the **C4.5 algorithm** (via the `ChefBoost` framework). C4.5 is a standard machine learning algorithm that constructs decision trees by recursively splitting the dataset based on Shannon’s Information Entropy and **Gain Ratio**.

Because `ChefBoost` outputs its decision rules as nested programming logic (`outputs/rules/rules.py`) rather than a visual object, a plotting utility was introduced using `scikit-learn`'s `plot_tree` function. This created a visual model discrepancy that was subsequently identified and corrected.

---

## 2. The Core Divergence: C4.5 vs. CART
The table below contrasts the actual C4.5 model and the scikit-learn plotting replica:

| Aspect | C4.5 Model (ChefBoost - Actual Model) | CART Model (Scikit-Learn - Plotting Replica) |
| :--- | :--- | :--- |
| **Core Algorithm** | C4.5 | CART (Classification and Regression Trees) |
| **Nominal Feature Treatment** | **Native Categorical Splitting**: Treats nominal features (e.g., `v116`, `v113`) as unordered categorical strings. | **Continuous / Ordinal Assumption**: Requires all features to be numeric. Treated nominal codes (e.g., `11`, `12`, `21`, `22`, `23`, `31`) as ordered continuous integers. |
| **Splitting Metric** | **Gain Ratio**: Adjusts for the number of categories to prevent bias toward high-cardinality features. | **Entropy / Gini Impurity**: Maximizes binary impurity reduction without adjusting for category numbers. |
| **Splitting Mechanism** | Multi-way or binary splits based on string categories. | Strict binary splits at numeric cutoffs (e.g., `v116 <= 27.0`). |
| **Mathematical Root Node** | **`b19`** (Child's Age in months, continuous) | **`v116`** (Type of Toilet Facility, nominal treated as continuous) |

---

## 3. Why the Root Nodes Differed

### The C4.5 Selection (`b19 > 6`)
C4.5 evaluates nominal columns by partitioning their categorical levels. Because some nominal variables (like `v113` or `v116`) have several unique category strings, the **Gain Ratio** penalized their raw information gain to avoid overfitting. Consequently, **`b19` (Age)** emerged as the partition yielding the highest normalized Gain Ratio.

### The CART Selection (`v116 <= 27.0`)
To render the tree using scikit-learn's plotting tool, the code was forced to convert string-based categories (e.g., `'11'`, `'21'`, `'31'`) into numerical integers (`11, 21, 31`). 
* Because scikit-learn only accepts numeric input, it treated these categories as an **ordered scale** ($11 < 12 < 21 < \dots$).
* It found that a binary threshold of **`v116 <= 27.0`** mathematically separated the data into two highly clean subsets:
  - Improved/semi-improved sanitation facilities (`11`, `12`, `21`, `22`, `23`)
  - No sanitation facility (`31` - Open Defecation)
* Because CART is binary-only and does not penalize feature cardinality with a Gain Ratio, this arbitrary numeric threshold of `27.0` produced the largest mathematical reduction in entropy, causing CART to select it as the root node.

---

## 4. Resolution
To maintain absolute scientific integrity and avoid confusing examiners, **the plotting script was updated to bypass the scikit-learn CART approximation**. 

We replaced the replica with a custom visualizer that maps the **actual C4.5 logical branches** from the generated rules (`outputs/rules/rules.py`). This guarantees that:
1. **`b19` (Age)** is displayed as the root node of the decision tree.
2. The tree diagram (`outputs/plots/decision_tree_plot.png`) is a 1:1 visual match for the logical rules described in your thesis text and Table 4.10.
3. The visual presentation represents the authentic, unsupervised output of the C4.5 algorithm.
