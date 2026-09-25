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
import pandas as pd
import wittgenstein as lw
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, precision_recall_curve, auc, roc_auc_score

print("="*90)
print("STEP 9: RULE COVERAGE, SUPPORT AUDIT & CALIBRATED ROC IMPLEMENTATION (ITEMS 10 & 11)")
print("="*90)

# Load data
X_train_res = pd.read_csv("X_train_resampled.csv")
Y_train_res = pd.read_csv("Y_train_resampled.csv")['hml32'].astype(int).values
X_test = pd.read_csv("X_test.csv")
Y_test = pd.read_csv("Y_test.csv")['hml32'].astype(int).values

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
for d_name, d_feat, d_y in [("Training (Resampled, N=316)", X_train_rip, Y_train_res), 
                            ("Clinical Test (N=49)", X_test_rip, Y_test)]:
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
        
        # Format rule string: clarify Wittgenstein's internal 'b19=14.0-17.0' bin syntax as mathematical interval '14 < b19 <= 17'
        rule_str = str(rule).replace("b19=14.0-17.0", "14 < b19 <= 17").replace("b19=14.0 - 17.0", "14 < b19 <= 17")
        r_name = f"RIPPER Rule R{r_idx+1}: {rule_str}"
        res = audit_rule(d_feat, d_y, active_mask, r_name, d_name, target_class=1)
        ripper_audit_rows.append(res)
        
    # Default Rule: covers remaining instances -> predicts Negative (class 0)
    def_mask = ~covered_so_far
    def_res = audit_rule(d_feat, d_y, def_mask, "RIPPER Default Rule: [ELSE => Class 0]", d_name, target_class=0)
    ripper_audit_rows.append(def_res)

df_ripper_audit = pd.DataFrame(ripper_audit_rows)
print("\n" + "="*130)
print("RIPPER PROPOSITIONAL RULE SUPPORT & COVERAGE AUDIT MATRIX")
print("="*130)
print(f"{'Rule Statement':<50} | {'Dataset':<28} | {'Support (n)':<11} | {'Coverage (%)':<13} | {'TP':<4} | {'FP':<4} | {'Confidence (%)':<14} | {'Recall (%)':<10}")
print("-" * 150)
for _, r in df_ripper_audit.iterrows():
    print(f"{r['Rule'][:50]:<50} | {r['Dataset']:<28} | {r['Support_n']:<11} | {r['Coverage_%']:<12.2f}% | {r['TP']:<4} | {r['FP']:<4} | {r['Precision_Confidence_%']:<13.2f}% | {r['Recall_Yield_%']:<9.2f}%")
print("="*130)

# ------------------------------------------------------------------------------
# 2. AUDIT C4.5 PRINCIPAL DECISION BRANCHES (HARMONIZED WITH FIGURE 4.5 & RULES.PY)
# ------------------------------------------------------------------------------
print("\n[3/3] Auditing C4.5 Principal Decision Branches (Harmonized with Figure 4.5 & rules.py)...")

c45_audit_rows = []

# Define key empirical pathways matching Figure 4.5 (decision_tree_plot.png) and rules.py:
# Pathway C1: Infancy Protection [b19 <= 6.0 months => Class 0]
# Pathway C2: Protected Well x Metal Roof [b19 > 6 ^ v158 = 0 ^ hml20 = 0 ^ v113 = 31 ^ v129 = 31 => Class 1]
# Pathway C3: Rainwater x Maternal Illiteracy [b19 > 6 ^ v158 = 0 ^ hml20 = 0 ^ v113 = 43 ^ v106 = 0 => Class 1]
# Pathway C4: Borehole/Well x Mud Walls [b19 > 6 ^ v158 = 0 ^ hml20 = 0 ^ v113 in {21, 32} ^ v128 = 21 => Class 1]
# Pathway C5: Intervention Paradox [b19 > 6 ^ v158 = 0 ^ hml20 = 1 ^ v119 = 1 ^ v190 <= 2 => Class 1]

for d_name, d_feat, d_y in [("Training (Resampled, N=316)", X_train_res, Y_train_res), 
                            ("Clinical Test (N=49)", X_test, Y_test)]:
    
    # Pathway C1: b19 <= 6 => Class 0
    mask_c1 = (d_feat['b19'] <= 6)
    c45_audit_rows.append(audit_rule(
        d_feat, d_y, mask_c1, 
        "Pathway C1: Infancy Protection [b19 <= 6 => Class 0]", 
        d_name, target_class=0
    ))
    
    # Pathway C2: b19 > 6 ^ v158 = 0 ^ hml20 = 0 ^ v113 = 31 ^ v129 = 31 => Class 1
    mask_c2 = (
        (d_feat['b19'] > 6) & 
        (d_feat['v158'] == 0) & 
        (d_feat['hml20'] == 0) & 
        (d_feat['v113'] == 31) & 
        (d_feat['v129'] == 31)
    )
    c45_audit_rows.append(audit_rule(
        d_feat, d_y, mask_c2, 
        "Pathway C2: Protected Well x Metal Roof [b19>6 ^ v158=0 ^ hml20=0 ^ v113=31 ^ v129=31 => Class 1]", 
        d_name, target_class=1
    ))
    
    # Pathway C3: b19 > 6 ^ v158 = 0 ^ hml20 = 0 ^ v113 = 43 ^ v106 = 0 => Class 1
    mask_c3 = (
        (d_feat['b19'] > 6) & 
        (d_feat['v158'] == 0) & 
        (d_feat['hml20'] == 0) & 
        (d_feat['v113'] == 43) & 
        (d_feat['v106'] == 0)
    )
    c45_audit_rows.append(audit_rule(
        d_feat, d_y, mask_c3, 
        "Pathway C3: Rainwater x Maternal Illiteracy [b19>6 ^ v158=0 ^ hml20=0 ^ v113=43 ^ v106=0 => Class 1]", 
        d_name, target_class=1
    ))
    
    # Pathway C4: b19 > 6 ^ v158 = 0 ^ hml20 = 0 ^ v113 in {21, 32} ^ v128 = 21 => Class 1
    mask_c4 = (
        (d_feat['b19'] > 6) & 
        (d_feat['v158'] == 0) & 
        (d_feat['hml20'] == 0) & 
        (d_feat['v113'].isin([21, 32])) & 
        (d_feat['v128'] == 21)
    )
    c45_audit_rows.append(audit_rule(
        d_feat, d_y, mask_c4, 
        "Pathway C4: Borehole/Well x Mud Walls [b19>6 ^ v158=0 ^ hml20=0 ^ v113 in {21,32} ^ v128=21 => Class 1]", 
        d_name, target_class=1
    ))
    
    # Pathway C5: b19 > 6 ^ v158 = 0 ^ hml20 = 1 ^ v119 = 1 ^ v190 <= 2 => Class 1
    mask_c5 = (
        (d_feat['b19'] > 6) & 
        (d_feat['v158'] == 0) & 
        (d_feat['hml20'] == 1) & 
        (d_feat['v119'] == 1) & 
        (d_feat['v190'] <= 2)
    )
    c45_audit_rows.append(audit_rule(
        d_feat, d_y, mask_c5, 
        "Pathway C5: Intervention Paradox [b19>6 ^ v158=0 ^ hml20=1 ^ v119=1 ^ v190<=2 => Class 1]", 
        d_name, target_class=1
    ))

df_c45_audit = pd.DataFrame(c45_audit_rows)
print("\n" + "="*130)
print("C4.5 PRINCIPAL DECISION PATHWAY SUPPORT & COVERAGE AUDIT MATRIX (HARMONIZED)")
print("="*130)
print(f"{'Pathway Description':<50} | {'Dataset':<28} | {'Support (n)':<11} | {'Coverage (%)':<13} | {'TP':<4} | {'FP':<4} | {'Confidence (%)':<14} | {'Recall (%)':<10}")
print("-" * 150)
for _, r in df_c45_audit.iterrows():
    print(f"{r['Rule'][:50]:<50} | {r['Dataset']:<28} | {r['Support_n']:<11} | {r['Coverage_%']:<12.2f}% | {r['TP']:<4} | {r['FP']:<4} | {r['Precision_Confidence_%']:<13.2f}% | {r['Recall_Yield_%']:<9.2f}%")
print("="*130)

# Save audit tables safely (handling possible Windows file locks if open in Excel)
# Save audit tables safely (handling possible Windows file locks if open in Excel)
df_ripper_audit.to_csv("outputs/ripper_rule_coverage_support_updated.csv", index=False)
try:
    df_ripper_audit.to_csv("outputs/ripper_rule_coverage_support.csv", index=False)
    print("\nExported RIPPER rule support matrix to outputs/ripper_rule_coverage_support.csv and updated.csv.")
except PermissionError:
    print(f"\nNote: outputs/ripper_rule_coverage_support.csv is locked by another application. Saved to outputs/ripper_rule_coverage_support_updated.csv.")

df_c45_audit.to_csv("outputs/c45_pathway_coverage_support_updated.csv", index=False)
try:
    df_c45_audit.to_csv("outputs/c45_pathway_coverage_support.csv", index=False)
    print("Exported C4.5 pathway support matrix to outputs/c45_pathway_coverage_support.csv and updated.csv.")
except PermissionError:
    print(f"Note: outputs/c45_pathway_coverage_support.csv is locked by another application. Saved to outputs/c45_pathway_coverage_support_updated.csv.")

# ------------------------------------------------------------------------------
# 3. ITEM 11: CALIBRATED ROC EVALUATION
# ------------------------------------------------------------------------------
print("\n" + "="*90)
print("ITEM 11: CALIBRATED CONTINUOUS ROC IMPLEMENTATION VS DISCRETE HARD PROJECTIONS")
print("="*90)

# Compare discrete C4.5 score vs leaf-calibrated score
# Using DecisionTreeClassifier with Entropy (C4.5 equivalent) to extract continuous probability estimates
dt_calibrated = DecisionTreeClassifier(criterion='entropy', random_state=42, min_samples_leaf=3)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

prep = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['b19']),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), nominal_features)
    ]
)
pipe_dt_cal = Pipeline([('prep', prep), ('clf', dt_calibrated)])

X_tr_cl = X_train_res.copy()
X_te_cl = X_test.copy()
X_tr_cl['b19'] = X_tr_cl['b19'].astype(float)
X_te_cl['b19'] = X_te_cl['b19'].astype(float)
for col in nominal_features:
    X_tr_cl[col] = X_tr_cl[col].astype(int).astype(str)
    X_te_cl[col] = X_te_cl[col].astype(int).astype(str)

pipe_dt_cal.fit(X_tr_cl, Y_train_res)

# Hard predictions vs continuous probability scores
c45_hard_probs = np.load("outputs/models/test_predictions.npz")['prob_c45']
c45_cal_probs = pipe_dt_cal.predict_proba(X_te_cl)[:, 1]

fpr_hard, tpr_hard, _ = roc_curve(Y_test, c45_hard_probs)
auc_hard = auc(fpr_hard, tpr_hard)

fpr_cal, tpr_cal, _ = roc_curve(Y_test, c45_cal_probs)
auc_cal = auc(fpr_cal, tpr_cal)

p_hard, r_hard, _ = precision_recall_curve(Y_test, c45_hard_probs)
prauc_hard = auc(r_hard, p_hard)

p_cal, r_cal, _ = precision_recall_curve(Y_test, c45_cal_probs)
prauc_cal = auc(r_cal, p_cal)

print(f"\nC4.5 ROC & PR Implementation Comparison:")
print(f" - Hard Binary Step Curve (Old)       : ROC-AUC = {auc_hard:.4f}, PR-AUC = {prauc_hard:.4f}")
print(f" - Continuous Calibrated Leaf Curves  : ROC-AUC = {auc_cal:.4f}, PR-AUC = {prauc_cal:.4f}")
print(f" - Number of Operating Thresholds    : Hard = {len(fpr_hard)} points, Calibrated = {len(fpr_cal)} points")
print(f"\nConclusion for Item 11: Continuous probability calibration resolves the stepped curve artifact while confirming weak discriminative signal ({auc_cal:.4f}).")
