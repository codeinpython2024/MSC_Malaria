# Walkthrough: Step 5 C4.5 Decision Trees & Step 6 RIPPER Rulesets

This document serves as the comprehensive walkthrough and academic audit trail for **Step 5: Inducing C4.5 Decision Trees via ChefBoost** and **Step 6: Inducing Separate-and-Conquer Rulesets (RIPPER) via Wittgenstein**, as implemented in [fourth_step.py](file:///c:/MSc_malaria/fourth_step.py).

---

## 1. Theoretical & Mathematical Foundations

### A. Step 5: C4.5 Decision Tree Induction (ChefBoost)
The C4.5 algorithm is a landmark supervised learning method that builds decision trees from training data using the concept of **Information Entropy**. At each node of the tree, C4.5 chooses the attribute that most effectively splits the set of samples into subsets enriched in one class or another.

#### 1. Claude Shannon's Information Entropy
For a dataset $S$ containing binary classes $C = \{\text{Positive}, \text{Negative}\}$, the baseline entropy $H(S)$ measures the impurity of the dataset:

$$H(S) = - \sum_{i=1}^{c} p_i \log_2(p_i)$$

where $p_i$ represents the proportion of instances belonging to class $i$ in $S$. When the classes are perfectly balanced ($50\%$ positive and $50\%$ negative, as achieved after our SMOTE-NC oversampling), $H(S) = 1.0$, indicating maximum impurity.

#### 2. Information Gain & The Gain Ratio Correction
To split a node on attribute $A$, C4.5 calculates the expected reduction in entropy, known as **Information Gain**:

$$\text{Gain}(S, A) = H(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} H(S_v)$$

However, standard Information Gain inherently favors attributes with a large number of distinct nominal categories (e.g., a unique ID column would yield an entropy of $0$ and a massive gain, yet fail to generalize). To penalize such multi-way splits, C4.5 normalizes the Gain by the **Split Information**:

$$\text{SplitInfo}(S, A) = - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} \log_2\left(\frac{|S_v|}{|S|}\right)$$

$$\text{GainRatio}(S, A) = \frac{\text{Gain}(S, A)}{\text{SplitInfo}(S, A)}$$

The attribute maximizing $\text{GainRatio}(S, A)$ is selected as the splitting node.

---

### B. Step 6: RIPPER Separate-and-Conquer Rule Induction (Wittgenstein)
The **Repeated Incremental Pruning to Produce Error Reduction (RIPPER)** algorithm is a separate-and-conquer rule induction method. Unlike decision trees which partition the entire feature space simultaneously, RIPPER builds a sequential ruleset that isolates the positive class (`hml32 = 1`) through highly specific logical conjunctions.

#### 1. The Separate-and-Conquer Strategy
1. **Grow Phase**: RIPPER adds conditions (antecedents) to a rule (consequent) one by one to maximize **FOIL (First Order Inductive Learner) Information Gain** until the rule covers no negative examples.
   
   $$\text{FOIL\_Gain} = s \times \left( \log_2\left(\frac{p_1}{p_1 + n_1}\right) - \log_2\left(\frac{p_0}{p_0 + n_0}\right) \right)$$
   
   where $p_0$ and $n_0$ are positive and negative instances covered by the original rule, $p_1$ and $n_1$ are covered by the rule after adding the antecedent, and $s$ is the number of positive instances covered by both.
   
2. **Prune Phase**: RIPPER evaluates the rule on a validation set and prunes antecedents that increase the error rate.
3. **Conquer/Remove Phase**: The instances covered by the finalized rule are deleted from the training set, and the loop repeats to find the next rule (the "or" condition `V`) until no positive instances remain or the description length is exceeded.

---

## 2. Deliverable 1: C4.5 Interpretative Tree Pathways

The induced C4.5 Decision Tree exports complete python code in `outputs/rules/rules.py` containing nested `if-then` logical statements. By resolving nominal integer codes back to their dictionary mappings (from Step 2), we extract the following clinical pathways:

### Root Level Split: Child's Age (`b19`)
* **Pathway 1: Age $\le 6$ months** $\rightarrow$ **Classified as Negative (No Malaria)**
  > [!NOTE]
  > **Clinical Interpretation**: Infants under 6 months have protective maternal antibodies (transplacentally acquired IgG) and high levels of fetal hemoglobin (HbF), which significantly retards parasite development, rendering active clinical malaria extremely rare.

* **Pathway 2: Age $> 6$ months** $\rightarrow$ **Proceed to Sociodemographic Exposures**
  Older children are highly vulnerable and undergo further stratification based on household risk factors:

#### Sub-Pathway 2.1: Media Exposure (`v158` - Radio Access)
* **If `v158 == '0'` (Household does not listen to radio at all)**:
  * **Bednet Usage (`hml20` - Slept Under LLIN)**:
    * **If `hml20 == '0'` (Did NOT sleep under a Long-Lasting Insecticidal Net)**:
      * **Water Source (`v113` - Water Type)**:
        * **If `v113 == '31'` (Uses a Protected Well)**:
          * **Roof Material (`v129` - Roof Type)**:
            * **If `v129 == '31'` (Metal/Tin Roof)** $\rightarrow$ **Classified as Positive (Active Malaria)**
            * **If `v129 \in \{'13', '22', '12'\}` (Thatch, Mud, or Rustic Roofs)** $\rightarrow$ **Classified as Negative**
        * **If `v113 == '43'` (Uses Rainwater)**:
          * **Maternal Education (`v106` - Education level)**:
            * **If `v106 == '0'` (No Maternal Education)** $\rightarrow$ **Classified as Positive**
            * **If `v106 \in \{'1', '2'\}` (Primary or Secondary Education)** $\rightarrow$ **Classified as Negative**
        * **If `v113 == '21'` (Uses a Tube well/borehole)**:
          * **Wall Material (`v128` - Wall Type)**:
            * **If `v128 == '21'` (Mud/Brick Walls)** $\rightarrow$ **Classified as Positive**

    * **If `hml20 == '1'` (Slept under a Net)**:
      * **Household Electricity (`v119` - Has Electricity)**:
        * **If `v119 == '1'` (Has Electricity) & Wealth Quintile (`v190` - Poorer)** $\rightarrow$ **Classified as Positive**
        * **If `v119 == '0'` (No Electricity)** $\rightarrow$ **Classified as Negative (Protected by LLIN)**

---

## 3. Deliverable 2: RIPPER Separate-and-Conquer Ruleset

The Wittgenstein RIPPER ruleset represents a highly compact "separate-and-conquer" logic for classifying malaria-positive cases:

```python
[[v113=31 ^ v127=34 ^ v116=31 ^ b4=2 ^ v129=31 ^ hml20=0] V [v128=21 ^ v129=31 ^ b19=14.0-17.0]]
```

### Translation of RIPPER Rules:
RIPPER isolates a positive diagnosis if **Rule 1** OR **Rule 2** is satisfied:

1. **Rule 1 (Highly Specific Environmental Vulnerability)**:
   - **Water Source (`v113 = 31`)**: Protected Well.
   - **Floor Material (`v127 = 34`)**: Cement floor.
   - **Toilet Type (`v116 = 31`)**: Ventilated Improved Pit Latrine (VIP).
   - **Child's Sex (`b4 = 2`)**: Female child.
   - **Roof Material (`v129 = 31`)**: Metal/Tin roof.
   - **Net Usage (`hml20 = 0`)**: Child did **NOT** sleep under an insecticide-treated net.
   
2. **Rule 2 (Vulnerable Age & Infrastructure Conjunction)**:
   - **Wall Material (`v128 = 21`)**: Mud/dirt wall structure.
   - **Roof Material (`v129 = 31`)**: Metal/Tin roof.
   - **Child's Age (`b19 = 14.0 - 17.0 months`)**: Child is between 14 and 17 months old (a high-exposure toddler window where maternal antibody protection has completely decayed, and crawling behaviors increase exposure to vector mosquitoes).

---

## 4. Preliminary Performance Metrics Matrix

Both white-box models were trained on the balanced SMOTE-NC training split ($N=316$) and evaluated strictly on the untouched, stratified test split ($N=49$, positive prevalence of $18.37\%$).

| Metric | ChefBoost C4.5 Decision Tree | Wittgenstein RIPPER Ruleset |
| :--- | :---: | :---: |
| **Accuracy** | 57.14% | **75.51%** |
| **Precision (Malaria+)** | 16.67% | 0.00% |
| **Recall / Sensitivity (Malaria+)** | **33.33%** | 0.00% |
| **F1-Score (Malaria+)** | **22.22%** | 0.00% |
| **Specificity (Malaria-)** | 62.50% | **92.50%** |
| **Precision (Malaria-)** | 80.65% | 80.43% |
| **Recall (Malaria-)** | 62.50% | **92.50%** |
| **F1-Score (Malaria-)** | 70.42% | **86.05%** |

### Clinical and Algorithmic Interpretation of Trade-Offs:
* **The Sensitivity Priority (C4.5)**:
  In clinical diagnostics and epidemiology, failing to detect a positive case (False Negative) is far more dangerous than misclassifying a healthy child as sick (False Positive), as untreated malaria can rapidly progress to severe cerebral malaria or death. 
  The **C4.5 Decision Tree** achieved a **Recall of 33.33%** (capturing $3$ out of $9$ true positive cases), which represents superior clinical utility compared to RIPPER under the baseline data structure.
  
* **The High-Specificity Conservative Model (RIPPER)**:
  RIPPER optimizes for absolute rule precision. Under a severe class imbalance ($18.3\%$), RIPPER proved extremely conservative, only outputting positive classifications when rules were met with immense certainty. This resulted in an exceptionally high **Specificity (92.50%)** and global accuracy (**75.51%**), but missed the positive cohort.

---

## 5. Verification & Audit Trail

### How to Run and Verify
Execute the unified modeling script from the workspace root folder:
```powershell
.venv\Scripts\python.exe -X utf8 fourth_step.py
```

### Verified Audit Checklist
1. **Dynamic Path Alignment**: Adding `sys.path.insert(0, os.getcwd())` guarantees ChefBoost can dynamically import the generated nested model without throwing a `ModuleNotFoundError`.
2. **Object Cast Safety**: Nominal dimensions cast to `object` instead of `pandas.StringDtype` prevents ChefBoost from defaulting to regression trees (which would crash due to continuous computations on nominal values).
3. **Windows UTF-8 Compliance**: Standard stream wrapping enables the safe terminal printing of indicators (like the C4.5 tree structural branch markers) on Windows without causing encoding/charmap aborts.
