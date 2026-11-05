import os
import sys
import io
import pickle
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc, average_precision_score, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

# Guarantee dynamic path discovery for ChefBoost C4.5 rules.py and c45_tree_utils
sys.path.insert(0, os.getcwd())
from c45_tree_utils import C45Tree

os.makedirs(os.path.join("outputs", "plots"), exist_ok=True)
os.makedirs(os.path.join("outputs", "models"), exist_ok=True)

# Every number reported, plotted or saved by this script is computed from the
# fitted models and the test partition. No metric values are typed in.

# ------------------------------------------------------------------------------
# 1. INGEST DATASETS
# ------------------------------------------------------------------------------
print("=" * 80)
print("STAGE 8: JOINT PROGRAMMATIC EVALUATION AND STRUCTURAL COMPLEXITY PROFILING")
print("=" * 80)

print("\n[1/8] Ingesting datasets...")
try:
    X_train = pd.read_csv("X_train_resampled.csv")
    Y_train = pd.read_csv("Y_train_resampled.csv")['hml32'].astype(int)
    X_test = pd.read_csv("X_test.csv")
    Y_test = pd.read_csv("Y_test.csv")['hml32'].astype(int)
    print("Loaded successfully:")
    print(f" - Train resampled instances: {len(X_train)}")
    print(f" - Test clinical instances: {len(X_test)} ({int(Y_test.sum())} positive)")
except Exception as e:
    print(f"Error loading datasets: {e}")
    sys.exit(1)

y_test = Y_test.values
prevalence = float(y_test.mean())

# ------------------------------------------------------------------------------
# 2. DATA PREPARATION FOR DISTINCT MODELS
# ------------------------------------------------------------------------------
print("\n[2/8] Customizing datatypes for model spaces...")

continuous_feature = ['b19']
nominal_features = ['b4', 'v106', 'v190', 'v119', 'v158', 'v127', 'v128', 'v129', 'v113', 'v116', 'hml20']

# RIPPER
X_train_ripper = X_train.copy()
X_test_ripper = X_test.copy()
X_train_ripper['b19'] = X_train_ripper['b19'].astype(int)
X_test_ripper['b19'] = X_test_ripper['b19'].astype(int)
for col in nominal_features:
    X_train_ripper[col] = X_train_ripper[col].astype(int).astype(str).astype(object)
    X_test_ripper[col] = X_test_ripper[col].astype(int).astype(str).astype(object)

# scikit-learn pipelines (Logistic Regression, RF, SVM)
X_train_bb = X_train.copy()
X_test_bb = X_test.copy()
X_train_bb['b19'] = X_train_bb['b19'].astype(float)
X_test_bb['b19'] = X_test_bb['b19'].astype(float)
for col in nominal_features:
    X_train_bb[col] = X_train_bb[col].astype(int).astype(str)
    X_test_bb[col] = X_test_bb[col].astype(int).astype(str)

print(" - Datatypes partitioned and stabilized.")

# ------------------------------------------------------------------------------
# 3. MODEL INGESTION & TEST SET PREDICTIONS
# ------------------------------------------------------------------------------
print("\n[3/8] Loading models and generating predictions...")

preds, scores = {}, {}

# --- Majority class (Zero-R) ---
preds['Zero-R'] = np.zeros(len(y_test), dtype=int)
scores['Zero-R'] = np.full(len(y_test), float(Y_train.mean()))   # constant score
print(" - Zero-R: predicts the negative class for every child.")

# --- C4.5 Decision Tree (ChefBoost rules.py) ---
# Hard labels come from rules.py. The continuous score is the Laplace-corrected
# positive frequency of the training records in the leaf each child reaches
# (see c45_tree_utils.py). C45Tree.leaf_ids() asserts that its routing agrees
# with ChefBoost's own findDecision() for every row.
c45_tree = C45Tree.from_file(os.path.join("outputs", "rules", "rules.py"))
preds['C4.5'] = c45_tree.predict(X_test)
c45_tree.fit_leaf_scores(X_train, Y_train)
scores['C4.5'] = c45_tree.predict_score(X_test)
c45_hard_score = preds['C4.5'].astype(float)
print(f" - C4.5: {len(c45_tree.leaves)} leaves parsed from rules.py; "
      f"{len(np.unique(scores['C4.5']))} distinct leaf scores on the test set.")

# --- Logistic Regression (linear baseline) ---
lr_pipe = Pipeline([
    ('prep', ColumnTransformer(transformers=[
        ('num', StandardScaler(), continuous_feature),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), nominal_features)
    ])),
    ('clf', LogisticRegression(C=1.0, random_state=42, max_iter=1000))
])
lr_pipe.fit(X_train_bb, Y_train)
preds['Logistic Regression'] = lr_pipe.predict(X_test_bb)
scores['Logistic Regression'] = lr_pipe.predict_proba(X_test_bb)[:, 1]
lr_n_coef = int(lr_pipe.named_steps['clf'].coef_.shape[1])
print(f" - Logistic Regression: fitted on {lr_n_coef} encoded inputs "
      f"(1 scaled age column + one-hot columns for 11 nominal variables).")

# --- RIPPER Ruleset (Wittgenstein) ---
import wittgenstein as lw
ripper_model = lw.RIPPER(random_state=42)
ripper_model.fit(X_train_ripper, Y_train, pos_class=1)
preds['RIPPER'] = np.array([1 if p else 0 for p in ripper_model.predict(X_test_ripper)])
scores['RIPPER'] = ripper_model.predict_proba(X_test_ripper)[:, 1]
print(f" - RIPPER: {len(ripper_model.ruleset_.rules)} rules induced.")

# --- Random Forest & SVM (pipelines tuned and saved by fifth_step.py) ---
with open(os.path.join("outputs", "models", "random_forest_pipeline.pkl"), 'rb') as f:
    rf_pipeline = pickle.load(f)
preds['Random Forest'] = rf_pipeline.predict(X_test_bb)
scores['Random Forest'] = rf_pipeline.predict_proba(X_test_bb)[:, 1]

with open(os.path.join("outputs", "models", "svm_pipeline.pkl"), 'rb') as f:
    svm_pipeline = pickle.load(f)
preds['SVM'] = svm_pipeline.predict(X_test_bb)
scores['SVM'] = svm_pipeline.predict_proba(X_test_bb)[:, 1]
print(" - Random Forest and SVM pipelines loaded from outputs/models/.")

MODEL_ORDER = ['Zero-R', 'Logistic Regression', 'C4.5', 'RIPPER', 'Random Forest', 'SVM']

# ------------------------------------------------------------------------------
# 4. COMPUTE METRICS
# ------------------------------------------------------------------------------
print("\n[4/8] Generating comparative evaluation metrics...")


def compute_metrics(y_true, y_pred, y_score, name, constant_score=False):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1_pos = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
    f1_neg = 2 * npv * specificity / (npv + specificity) if (npv + specificity) > 0 else 0.0
    macro_f1 = (f1_pos + f1_neg) / 2
    # A constant score carries no ranking information: ROC-AUC = 0.5 and
    # average precision = prevalence by definition.
    roc_auc = 0.5 if constant_score else roc_auc_score(y_true, y_score)
    pr_auc = float(np.mean(y_true)) if constant_score else average_precision_score(y_true, y_score)
    return {
        "Model": name, "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "Accuracy": accuracy, "Sensitivity": sensitivity, "Specificity": specificity,
        "Precision": precision, "NPV": npv, "F1_Pos": f1_pos, "F1_Neg": f1_neg,
        "Macro_F1": macro_f1, "ROC_AUC": roc_auc, "PR_AUC_AP": pr_auc,
    }


results = {m: compute_metrics(y_test, preds[m], scores[m], m, constant_score=(m == 'Zero-R'))
           for m in MODEL_ORDER}

# C4.5 hard-label (single operating point) AUC, reported alongside the leaf-score AUC
c45_hard_auc = roc_auc_score(y_test, c45_hard_score)
c45_hard_ap = average_precision_score(y_test, c45_hard_score)

print("\n" + "=" * 132)
print("PERFORMANCE EVALUATION MATRIX (CLINICAL TEST SET, N=49) -- point estimates")
print("=" * 132)
print(f"{'Model':<22} | {'CM [[TN,FP],[FN,TP]]':<20} | {'Acc':<7} | {'Sens':<7} | {'Spec':<7} | "
      f"{'Prec':<7} | {'NPV':<7} | {'F1+':<7} | {'F1-':<7} | {'MacroF1':<7} | {'ROC-AUC':<7} | {'PR-AUC(AP)':<9}")
print("-" * 132)
for m in MODEL_ORDER:
    r = results[m]
    cm_str = f"[[{r['TN']},{r['FP']}],[{r['FN']},{r['TP']}]]"
    print(f"{m:<22} | {cm_str:<20} | {r['Accuracy']:.4f} | {r['Sensitivity']:.4f} | {r['Specificity']:.4f} | "
          f"{r['Precision']:.4f} | {r['NPV']:.4f} | {r['F1_Pos']:.4f} | {r['F1_Neg']:.4f} | "
          f"{r['Macro_F1']:.4f}  | {r['ROC_AUC']:.4f}  | {r['PR_AUC_AP']:.4f}")
print("=" * 132)
print(f"PR-AUC is average precision (sklearn.metrics.average_precision_score). "
      f"No-skill baseline = test prevalence = {prevalence:.4f}.")
print(f"C4.5 ROC-AUC above uses Laplace leaf scores. Hard-label (single operating point) "
      f"ROC-AUC = {c45_hard_auc:.4f}, hard-label AP = {c45_hard_ap:.4f}.")

pd.DataFrame([results[m] for m in MODEL_ORDER]).to_csv(
    os.path.join("outputs", "test_point_metrics.csv"), index=False)
print("Point metrics written to outputs/test_point_metrics.csv")

# Save the real predictions and scores for the bootstrap (run_bootstrap_ci_final.py)
np.savez(
    os.path.join("outputs", "models", "test_predictions.npz"),
    Y_test=y_test,
    pred_zero=preds['Zero-R'], prob_zero=scores['Zero-R'],
    pred_lr=preds['Logistic Regression'], prob_lr=scores['Logistic Regression'],
    pred_c45=preds['C4.5'], prob_c45=scores['C4.5'], prob_c45_hard=c45_hard_score,
    pred_ripper=preds['RIPPER'], prob_ripper=scores['RIPPER'],
    pred_rf=preds['Random Forest'], prob_rf=scores['Random Forest'],
    pred_svm=preds['SVM'], prob_svm=scores['SVM'],
)
print("Test-set predictions and scores saved to outputs/models/test_predictions.npz")

# ------------------------------------------------------------------------------
# 5. ROC CURVES
# ------------------------------------------------------------------------------
print("\n[5/8] Graphing comparative ROC curves...")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

colors = {'Logistic Regression': '#17becf', 'Random Forest': '#2ca02c', 'SVM': '#9467bd',
          'C4.5': '#1f77b4', 'RIPPER': '#ff7f0e'}
curve_labels = {'Logistic Regression': 'Logistic Regression',
                'Random Forest': 'Random Forest',
                'SVM': 'SVM (RBF)',
                'C4.5': 'C4.5 (Laplace leaf scores)',
                'RIPPER': 'RIPPER (rule-confidence scores)'}

plt.figure(figsize=(10, 8), dpi=300)
for m in ['Logistic Regression', 'Random Forest', 'SVM', 'C4.5', 'RIPPER']:
    fpr, tpr, _ = roc_curve(y_test, scores[m])
    plt.plot(fpr, tpr, lw=2.2, color=colors[m], label=f"{curve_labels[m]} (AUC = {auc(fpr, tpr):.4f})")
    print(f" - ROC for {curve_labels[m]}: AUC = {auc(fpr, tpr):.4f}")

fpr_h, tpr_h, _ = roc_curve(y_test, c45_hard_score)
plt.plot(fpr_h, tpr_h, lw=1.5, color='#1f77b4', linestyle=':', alpha=0.7,
         label=f"C4.5 hard labels (AUC = {c45_hard_auc:.4f})")
r = results['C4.5']
plt.scatter([1 - r['Specificity']], [r['Sensitivity']], color='#1f77b4', s=60, zorder=5,
            label=f"C4.5 operating point (Sens={r['Sensitivity']:.2f}, FPR={1 - r['Specificity']:.2f})")
plt.plot([0, 1], [0, 1], color='grey', linestyle='--', lw=1.5, label='No discrimination (AUC = 0.5)')
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12, fontweight='bold')
plt.title('ROC curves on the test partition (N = 49, 9 positive)\nRural Nasarawa State, microscopy outcome',
          fontsize=13, fontweight='bold')
plt.legend(loc="lower right", fontsize=10, frameon=True)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join("outputs", "plots", "roc_comparison.png"), dpi=300)
plt.close()
print(" - ROC figure written to outputs/plots/roc_comparison.png")

# ------------------------------------------------------------------------------
# 6. WHITE-BOX STRUCTURAL COMPLEXITY
# ------------------------------------------------------------------------------
print("\n[6/8] Profiling structural complexity...")
leaf_counts_train = c45_tree.leaf_counts(X_train, Y_train)
c45_n_leaves = len(c45_tree.leaves)
c45_n_leaves_nonempty = int((leaf_counts_train['n'] > 0).sum())
c45_mean_literals = float(np.mean([len(l.path) for l in c45_tree.leaves]))
c45_mean_literals_nonempty = float(np.mean(
    [len(l.path) for l in c45_tree.leaves if leaf_counts_train.loc[l.id, 'n'] > 0]))

ripper_rules_count = len(ripper_model.ruleset_.rules)
ripper_mean_literals = float(np.mean([len(rule.conds) for rule in ripper_model.ruleset_.rules])) \
    if ripper_rules_count else 0.0

rf_n_trees = len(rf_pipeline[-1].estimators_)
svm_n_sv = int(np.sum(svm_pipeline[-1].n_support_))

print(f"C4.5: {c45_n_leaves} leaves in rules.py ({c45_n_leaves_nonempty} reached by training records); "
      f"mean conditions per leaf = {c45_mean_literals:.2f} (all) / {c45_mean_literals_nonempty:.2f} (reached)")
print(f"RIPPER: {ripper_rules_count} rules + default; mean conditions per rule = {ripper_mean_literals:.2f}")
for i, rule in enumerate(ripper_model.ruleset_.rules, 1):
    print(f"   R{i}: {rule}")
print(f"Logistic Regression: {lr_n_coef} coefficients + intercept")
print(f"Random Forest: {rf_n_trees} trees")
print(f"SVM: {svm_n_sv} support vectors (kernel = {svm_pipeline[-1].kernel})")

# ------------------------------------------------------------------------------
# 7. ACCURACY-VERSUS-INTERPRETABILITY TABLE
# ------------------------------------------------------------------------------
print("\n[7/8] Accuracy-versus-interpretability matrix...")
complexity = {
    'RIPPER': (ripper_rules_count, f"{ripper_rules_count} rules"),
    'Logistic Regression': (lr_n_coef, f"{lr_n_coef} coefficients"),
    'C4.5': (c45_n_leaves, f"{c45_n_leaves} leaves"),
    'Random Forest': (rf_n_trees, f"{rf_n_trees} trees"),
    'SVM': (svm_n_sv, f"{svm_n_sv} support vectors"),
}
tradeoff_rows = []
for m in ['RIPPER', 'Logistic Regression', 'C4.5', 'Random Forest', 'SVM']:
    r = results[m]
    tradeoff_rows.append({"Model": m, "Complexity": complexity[m][1], "Complexity_N": complexity[m][0],
                          "Accuracy": r['Accuracy'], "Sensitivity": r['Sensitivity'],
                          "Macro_F1": r['Macro_F1'], "ROC_AUC": r['ROC_AUC'], "PR_AUC_AP": r['PR_AUC_AP']})
tradeoff_df = pd.DataFrame(tradeoff_rows)
print(tradeoff_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
tradeoff_df.to_csv(os.path.join("outputs", "tradeoff_matrix.csv"), index=False)

# ------------------------------------------------------------------------------
# 8. ADDITIONAL FIGURES (all values taken from `results`)
# ------------------------------------------------------------------------------
print("\n[8/8] Generating remaining evaluation plots...")
display = {'Zero-R': 'Majority-Class\n(Zero-R)', 'Logistic Regression': 'Logistic\nRegression',
           'C4.5': 'C4.5 Decision\nTree', 'RIPPER': 'RIPPER\nRuleset', 'Random Forest': 'Random\nForest',
           'SVM': 'Support Vector\nMachine'}

# 8A. Grouped bar chart
metrics = [('Accuracy', 'Accuracy'), ('Precision', 'Precision'), ('Sensitivity', 'Sensitivity'),
           ('Specificity', 'Specificity'), ('Macro_F1', 'Macro F1')]
metric_colors = ['#2b5c8f', '#e67e22', '#27ae60', '#c0392b', '#8e44ad']
x = np.arange(len(MODEL_ORDER))
width = 0.14
plt.figure(figsize=(14, 7.5), dpi=300)
for idx, (key, label) in enumerate(metrics):
    plt.bar(x + (idx - 2) * width, [results[m][key] for m in MODEL_ORDER], width, label=label,
            color=metric_colors[idx], alpha=0.9, edgecolor='white', lw=0.6)
plt.title("Performance comparison on the test partition (N = 49)", fontsize=14, fontweight='bold')
plt.xticks(x, [display[m] for m in MODEL_ORDER], fontsize=10)
plt.ylabel("Score", fontsize=11, fontweight='bold')
plt.ylim(0, 1.12)
plt.legend(loc='upper right', fontsize=10, ncol=5)
plt.grid(True, linestyle=':', alpha=0.5, axis='y')
plt.tight_layout()
plt.savefig(os.path.join("outputs", "plots", "model_performance_comparison.png"), dpi=300)
plt.close()
print(" - outputs/plots/model_performance_comparison.png")

# 8B. Confusion matrices (all six models)
fig, axes = plt.subplots(2, 3, figsize=(14, 9.5), dpi=300)
for ax, m in zip(axes.flatten(), MODEL_ORDER):
    r = results[m]
    cm = np.array([[r['TN'], r['FP']], [r['FN'], r['TP']]])
    ax.matshow(cm, cmap='Blues', alpha=0.6)
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, f'{val}', ha='center', va='center', fontsize=16, fontweight='bold')
    ax.set_title(display[m].replace('\n', ' '), fontsize=12, fontweight='bold', pad=10)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Negative', 'Positive'])
    ax.set_yticks([0, 1]); ax.set_yticklabels(['Negative', 'Positive'])
    ax.tick_params(axis="both", bottom=True, top=False, labelbottom=True, labeltop=False)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
plt.suptitle("Confusion matrices on the test partition (N = 49)", fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join("outputs", "plots", "confusion_matrices.png"), dpi=300)
plt.close()
print(" - outputs/plots/confusion_matrices.png")

# 8C. Figure 4.6: complexity vs macro F1 (log-scaled complexity axis, real counts)
point_colors = {'RIPPER': '#e67e22', 'Logistic Regression': '#16a085', 'C4.5': '#2980b9',
                'Random Forest': '#27ae60', 'SVM': '#8e44ad'}
plt.figure(figsize=(11, 6.5), dpi=300)
ax = plt.gca()
# Labels for C4.5 and RIPPER go below their points (C4.5 shares its x value with
# Logistic Regression; RIPPER sits just under the Zero-R reference line).
label_below = {'C4.5', 'RIPPER'}
for row in tradeoff_rows:
    m = row['Model']
    plt.scatter(row['Complexity_N'], row['Macro_F1'], color=point_colors[m], s=200, zorder=6,
                edgecolors='black', lw=0.8)
    below = m in label_below
    plt.annotate(f"{row['Complexity']}\n({m}, macro F1 = {row['Macro_F1']:.4f})",
                 (row['Complexity_N'], row['Macro_F1']), textcoords="offset points",
                 xytext=(0, -16) if below else (0, 14), va='top' if below else 'bottom',
                 ha='center', fontsize=8.5, fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=point_colors[m], lw=1.2, alpha=0.9))
plt.axhline(results['Zero-R']['Macro_F1'], color='grey', linestyle='--', lw=1.2,
            label=f"Zero-R macro F1 = {results['Zero-R']['Macro_F1']:.4f}")
ax.set_xscale('log')
plt.xlabel("Model size (rules / coefficients / leaves / trees / support vectors; log scale)",
           fontsize=11, fontweight='bold')
plt.ylabel("Macro F1 (test partition)", fontsize=11, fontweight='bold')
plt.title("Accuracy-versus-interpretability trade-off (test partition, N = 49)",
          fontsize=13, fontweight='bold')
f1s = [r['Macro_F1'] for r in tradeoff_rows] + [results['Zero-R']['Macro_F1']]
plt.ylim(min(f1s) - 0.05, max(f1s) + 0.08)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join("outputs", "plots", "interpretability_vs_accuracy.png"), dpi=300)
plt.close()
print(" - outputs/plots/interpretability_vs_accuracy.png")

print("\n" + "=" * 80)
print("STAGE 8 CLINICAL TESTING AND PROFILING SUCCESSFULLY COMPLETED")
print("=" * 80)
