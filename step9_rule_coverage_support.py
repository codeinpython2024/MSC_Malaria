import os
import sys
import io
import numpy as np

# Ensure UTF-8 stdout encoding
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass
sys.path.insert(0, os.getcwd())
import pandas as pd
import wittgenstein as lw
from sklearn.metrics import roc_auc_score, average_precision_score

from c45_tree_utils import C45Tree, PATHWAY_NAMES

print("="*90)
print("STEP 9: RULE COVERAGE AND SUPPORT AUDIT (C4.5 LEAVES FROM rules.py, RIPPER RULES)")
print("="*90)

# Load data
X_train_res = pd.read_csv("X_train_resampled.csv")
Y_train_res = pd.read_csv("Y_train_resampled.csv")['hml32'].astype(int).values
X_test = pd.read_csv("X_test.csv")
Y_test = pd.read_csv("Y_test.csv")['hml32'].astype(int).values
# third_step.py writes the original rows first; synthetic SMOTE-NC rows follow.
N_ORIG = len(pd.read_csv("Y_train_original.csv"))

nominal_features = ['b4', 'v106', 'v190', 'v119', 'v158', 'v127', 'v128', 'v129', 'v113', 'v116', 'hml20']

# Format for RIPPER
X_train_rip = X_train_res.copy()
X_test_rip = X_test.copy()
X_train_rip['b19'] = X_train_rip['b19'].astype(int)
X_test_rip['b19'] = X_test_rip['b19'].astype(int)
for col in nominal_features:
    X_train_rip[col] = X_train_rip[col].astype(int).astype(str).astype(object)
    X_test_rip[col] = X_test_rip[col].astype(int).astype(str).astype(object)

# Train RIPPER
print("\n[1/3] Fitting RIPPER and parsing individual propositional rules...")
ripper = lw.RIPPER(random_state=42)
ripper.fit(X_train_rip, Y_train_res, pos_class=1)

print(f"Induced RIPPER Ruleset ({len(ripper.ruleset_.rules)} target rules):")
for idx, r in enumerate(ripper.ruleset_.rules):
    print(f"  Rule {idx+1}: {r}")

# Function to audit a rule condition mask on a dataframe
def audit_rule(df_feat, y_true, mask, rule_name, dataset_name, target_class=1):
    n_cohort = len(df_feat)
    n_target_cohort = np.sum(y_true == target_class)
    
    # Support: instances satisfying antecedent
    support = int(np.sum(mask))
    coverage = (support / n_cohort) * 100.0 if n_cohort > 0 else 0.0
    
    tp = int(np.sum((mask == True) & (y_true == target_class)))
    fp = int(np.sum((mask == True) & (y_true != target_class)))
    
    rule_precision = (tp / support) * 100.0 if support > 0 else 0.0
    rule_recall = (tp / n_target_cohort) * 100.0 if n_target_cohort > 0 else 0.0
    
    return {
        "Rule": rule_name,
        "Dataset": dataset_name,
        "Cohort_N": n_cohort,
        "Support_n": support,
        "Coverage_%": coverage,
        "TP": tp,
        "FP": fp,
        "Precision_Confidence_%": rule_precision,
        "Recall_Yield_%": rule_recall
    }

# ------------------------------------------------------------------------------
# 1. AUDIT RIPPER RULES
# ------------------------------------------------------------------------------
print("\n[2/3] Auditing Rule Support, Coverage, Confidence, and Yield for RIPPER...")

ripper_audit_rows = []

# Evaluate each RIPPER rule on Training (N=316) and Test (N=49)
ripper_cohorts = [
    (f"Training (resampled, N={len(X_train_rip)})", X_train_rip, Y_train_res),
    (f"Training (original rows only, N={N_ORIG})", X_train_rip.iloc[:N_ORIG].reset_index(drop=True), Y_train_res[:N_ORIG]),
    (f"Test (N={len(X_test_rip)})", X_test_rip, Y_test),
]
for d_name, d_feat, d_y in ripper_cohorts:
    # Cumulative coverage tracker for sequential covering
    covered_so_far = np.zeros(len(d_feat), dtype=bool)
    
    # Transform continuous/binned features using RIPPER's bin_transformer_ so interval conditions match
    d_feat_binned = ripper.bin_transformer_.transform(d_feat)
    
    for r_idx, rule in enumerate(ripper.ruleset_.rules):
        # Build boolean mask for rule
        cond_masks = []
        for cond in rule.conds:
            col = cond.feature
            val = cond.val
            cond_masks.append(d_feat_binned[col] == val)
        
        # Rule antecedent mask
        rule_mask = np.logical_and.reduce(cond_masks)
        
        # In sequential covering, a rule applies only to instances not already covered
        active_mask = rule_mask & (~covered_so_far)
        covered_so_far = covered_so_far | rule_mask
        
        # Rule printed exactly as Wittgenstein reports it (b19 bins appear as 'lo-hi')
        r_name = f"RIPPER Rule R{r_idx+1}: {rule}"
        res = audit_rule(d_feat, d_y, active_mask, r_name, d_name, target_class=1)
        ripper_audit_rows.append(res)
        
    # Consistency check: the union of rule masks must equal RIPPER's own positive predictions
    rip_pos = np.array([bool(p) for p in ripper.predict(d_feat)])
    if not np.array_equal(rip_pos, covered_so_far):
        raise AssertionError(f"RIPPER rule masks disagree with ripper.predict() on {d_name}")

    # Default Rule: covers remaining instances -> predicts Negative (class 0)
    def_mask = ~covered_so_far
    def_res = audit_rule(d_feat, d_y, def_mask, "RIPPER Default Rule: [ELSE => Class 0]", d_name, target_class=0)
    ripper_audit_rows.append(def_res)

df_ripper_audit = pd.DataFrame(ripper_audit_rows)
print("\n" + "="*130)
print("RIPPER PROPOSITIONAL RULE SUPPORT & COVERAGE AUDIT MATRIX")
print("="*130)
print(f"{'Rule Statement':<50} | {'Dataset':<36} | {'Support (n)':<11} | {'Coverage (%)':<13} | {'TP':<4} | {'FP':<4} | {'Confidence (%)':<14} | {'Recall (%)':<10}")
print("-" * 150)
for _, r in df_ripper_audit.iterrows():
    print(f"{r['Rule'][:50]:<50} | {r['Dataset']:<36} | {r['Support_n']:<11} | {r['Coverage_%']:<12.2f}% | {r['TP']:<4} | {r['FP']:<4} | {r['Precision_Confidence_%']:<13.2f}% | {r['Recall_Yield_%']:<9.2f}%")
print("="*130)


for i, rule in enumerate(ripper.ruleset_.rules, 1):
    print(f"R{i} (full): {rule}")

df_ripper_audit.to_csv("outputs/ripper_rule_coverage_support.csv", index=False)
print("\nExported outputs/ripper_rule_coverage_support.csv")

# ------------------------------------------------------------------------------
# 2. AUDIT C4.5 LEAVES AND NAMED PATHWAYS, DIRECTLY FROM outputs/rules/rules.py
# ------------------------------------------------------------------------------
print("\n[3/3] Auditing C4.5 leaves and named pathways (parsed from outputs/rules/rules.py)...")

tree = C45Tree.from_file(os.path.join("outputs", "rules", "rules.py"))
pathways = tree.resolve_pathways()
cohorts = [
    (f"Training (resampled, N={len(X_train_res)})", X_train_res, Y_train_res),
    (f"Training (original rows only, N={N_ORIG})", X_train_res.iloc[:N_ORIG].reset_index(drop=True), Y_train_res[:N_ORIG]),
    (f"Test (N={len(X_test)})", X_test, Y_test),
]

# 2A. Every leaf on every cohort
leaf_rows = []
leaf_ids_by_cohort = {}
for d_name, d_feat, d_y in cohorts:
    ids = tree.leaf_ids(d_feat)
    leaf_ids_by_cohort[d_name] = ids
    for leaf in tree.leaves:
        mask = ids == leaf.id
        target = 1 if leaf.label == 'Positive' else 0
        row = audit_rule(d_feat, d_y, mask, f"Leaf {leaf.id}", d_name, target_class=target)
        row.update({"Leaf_ID": leaf.id, "Leaf_Class": leaf.label,
                    "Positives_in_leaf": int(np.sum(mask & (d_y == 1))),
                    "Path": tree.describe_path(leaf)})
        leaf_rows.append(row)
df_leaves = pd.DataFrame(leaf_rows)
df_leaves.to_csv("outputs/c45_leaf_support_all.csv", index=False)
print(f" - {len(tree.leaves)} leaves audited on {len(cohorts)} cohorts -> outputs/c45_leaf_support_all.csv")

# 2B. Named pathways (Table 4.10). Each pathway is one or more leaves of rules.py.
c45_audit_rows = []
for name, ids in pathways.items():
    target_label = tree.leaves[ids[0]].label
    target = 1 if target_label == 'Positive' else 0
    paths = " OR ".join(f"[{tree.describe_path(tree.leaves[i])}]" for i in ids)
    for d_name, d_feat, d_y in cohorts:
        mask = np.isin(leaf_ids_by_cohort[d_name], ids)
        row = audit_rule(d_feat, d_y, mask, f"Pathway {name}: {PATHWAY_NAMES[name]}", d_name, target_class=target)
        row.update({"Pathway": name, "Leaf_IDs": ",".join(map(str, ids)), "Predicted_Class": target_label,
                    "Positives_in_pathway": int(np.sum(mask & (d_y == 1))), "Conditions": paths})
        c45_audit_rows.append(row)
df_c45_audit = pd.DataFrame(c45_audit_rows)

print("\n" + "=" * 150)
print("C4.5 NAMED PATHWAY SUPPORT & COVERAGE AUDIT (Precision = share of covered children in the predicted class)")
print("=" * 150)
print(f"{'Pathway':<8} | {'Leaves':<6} | {'Class':<8} | {'Dataset':<36} | {'Support':<7} | {'Coverage':<9} | "
      f"{'TP':<4} | {'FP':<4} | {'Precision':<9} | {'Recall':<8} | {'Pos in path':<11}")
print("-" * 150)
for _, r in df_c45_audit.iterrows():
    print(f"{r['Pathway']:<8} | {r['Leaf_IDs']:<6} | {r['Predicted_Class']:<8} | {r['Dataset']:<36} | "
          f"{r['Support_n']:<7} | {r['Coverage_%']:<8.2f}% | {r['TP']:<4} | {r['FP']:<4} | "
          f"{r['Precision_Confidence_%']:<8.2f}% | {r['Recall_Yield_%']:<7.2f}% | {r['Positives_in_pathway']:<11}")
print("=" * 150)
for name, ids in pathways.items():
    for i in ids:
        print(f"{name} = Leaf {i} ({tree.leaves[i].label}): {tree.describe_path(tree.leaves[i])}")

df_c45_audit.to_csv("outputs/c45_pathway_coverage_support.csv", index=False)
print("\nExported outputs/c45_pathway_coverage_support.csv")

# 2C. Positive leaves not among the named pathways (reported for completeness)
named = {i for ids in pathways.values() for i in ids}
train_counts = tree.leaf_counts(X_train_res, Y_train_res)
print("\nOther leaves predicting 'Positive' (not in the named pathways):")
for leaf in tree.leaves:
    if leaf.label == 'Positive' and leaf.id not in named:
        n, k = int(train_counts.loc[leaf.id, 'n']), int(train_counts.loc[leaf.id, 'pos'])
        print(f"  Leaf {leaf.id}: train n={n}, positives={k} :: {tree.describe_path(leaf)}")

# ------------------------------------------------------------------------------
# 3. C4.5 DISCRIMINATION: LEAF SCORES vs HARD LABELS (from sixth_step.py outputs)
# ------------------------------------------------------------------------------
npz = np.load(os.path.join("outputs", "models", "test_predictions.npz"))
print("\nC4.5 discrimination on the test partition:")
print(f" - Laplace leaf-score ROC-AUC = {roc_auc_score(Y_test, npz['prob_c45']):.4f}, "
      f"AP = {average_precision_score(Y_test, npz['prob_c45']):.4f} "
      f"({len(np.unique(npz['prob_c45']))} distinct scores)")
print(f" - Hard-label ROC-AUC         = {roc_auc_score(Y_test, npz['prob_c45_hard']):.4f}, "
      f"AP = {average_precision_score(Y_test, npz['prob_c45_hard']):.4f}")
